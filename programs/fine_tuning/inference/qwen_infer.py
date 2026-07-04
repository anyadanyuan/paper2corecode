"""
qwen_infer.py — 调用微调后的 Qwen2.5-Coder-7B 模型，生成核心代码

两种运行模式（自动检测）：
  1. 本地加载模式：直接用 transformers 加载合并后的模型权重（需 GPU）
  2. API 模式：若 QWEN_API_BASE 环境变量存在，走 OpenAI-compatible HTTP 调用
     （需先启动 vLLM 服务，然后通过 SSH 端口转发或直接本地访问）

训练配置还原（来自 train_qwen.yaml）：
  - 底座：Qwen2.5-Coder-7B-Instruct
  - 微调：LoRA → 已合并导出为完整权重
  - 模型路径：/root/autodl-tmp/models/Qwen2.5-7B-paper2Xcode
  - 训练数据格式：Alpaca（instruction / input / output 三字段）

Usage（批量处理）：
  python qwen_infer.py \
      --input_dir  data/training_set_txts \
      --output_dir outputs/qwen \
      --model_path /root/autodl-tmp/models/Qwen2.5-7B-paper2Xcode

Usage（单文件推理）：
  python qwen_infer.py --single data/training_set_txts/ACT.txt

Usage（API 模式，需先启动 vLLM）：
  QWEN_API_BASE=http://localhost:8000/v1 \
  python qwen_infer.py --input_dir data/training_set_txts --output_dir outputs/qwen/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Alpaca instruction，与训练数据中一致
INSTRUCTION = (
    "You are an expert AI researcher. "
    "Implement the core PyTorch model architecture based on the following paper excerpts. "
    "Output the code using <file> tags."
)

# ── 推理后端 ──────────────────────────────────────────────────────────────────

class LocalInferencer:
    """
    直接用 transformers 加载合并后的模型权重。
    在服务器上运行，模型路径指向 /root/autodl-tmp/models/Qwen2.5-7B-paper2Xcode。
    """

    def __init__(self, model_path: str, max_new_tokens: int = 2048):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[LocalInferencer] Loading model from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens
        print("[LocalInferencer] Model loaded.")

    def generate(self, paper_text: str) -> str:
        import torch

        # 与训练时相同的 Alpaca → ChatML 映射：
        #   instruction + "\n\n" + input  →  user 消息
        user_content = f"{INSTRUCTION}\n\n{paper_text}"
        messages = [{"role": "user", "content": user_content}]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,          # 代码生成用贪心
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 只保留新生成的 token
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result


class APIInferencer:
    """
    通过 OpenAI-compatible API 调用（vLLM serve 后 SSH 端口转发到本地）。
    base_url 来自环境变量 QWEN_API_BASE，如 http://localhost:8000/v1。
    """

    def __init__(self, base_url: str, max_tokens: int = 2048):
        from openai import OpenAI

        self.model_name = "qwen-paper2code"  # 与 serve_vllm.sh 中 --served-model-name 一致
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key="sk-local-placeholder",   # vLLM 不校验 key
            base_url=base_url,
        )
        print(f"[APIInferencer] Using API at {base_url}")

    def generate(self, paper_text: str) -> str:
        user_content = f"{INSTRUCTION}\n\n{paper_text}"
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": user_content}],
            max_tokens=self.max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""


def get_inferencer(
    model_path: Optional[str] = None,
    max_tokens: int = 2048,
) -> LocalInferencer | APIInferencer:
    """自动选择推理后端"""
    api_base = os.environ.get("QWEN_API_BASE", "").strip()
    if api_base:
        return APIInferencer(base_url=api_base, max_tokens=max_tokens)

    if model_path is None:
        model_path = os.environ.get(
            "QWEN_MODEL_PATH", "/root/autodl-tmp/models/Qwen2.5-7B-paper2Xcode"
        )

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model path not found: {model_path}\n"
            "Either:\n"
            "  1. Set QWEN_API_BASE=http://localhost:8000/v1 (API mode)\n"
            "  2. Set QWEN_MODEL_PATH=<path> (local mode)\n"
            "  3. Pass --model_path to the script"
        )

    return LocalInferencer(model_path=model_path, max_new_tokens=max_tokens)


# ── 批量处理 ──────────────────────────────────────────────────────────────────

def process_directory(
    input_dir: Path,
    output_dir: Path,
    inferencer: LocalInferencer | APIInferencer,
    skip_existing: bool = True,
) -> None:
    """
    批量处理 cleaned_output/*.txt → qwen_outputs/*.xml
    支持增量（跳过已处理的文件）。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(input_dir.glob("*.txt"))

    if not txt_files:
        print(f"[WARN] No .txt files found in {input_dir}")
        return

    print(f"Found {len(txt_files)} paper(s) to process")
    skipped, processed, failed = 0, 0, 0

    for txt_path in txt_files:
        paper_name = txt_path.stem
        out_path = output_dir / f"{paper_name}.xml"

        if skip_existing and out_path.exists():
            skipped += 1
            continue

        print(f"\n[{processed + 1 + skipped}/{len(txt_files)}] {paper_name}")

        paper_text = txt_path.read_text(encoding="utf-8")
        # 截断：训练时 cutoff_len=2048 tokens，约 6000 字符
        paper_text = paper_text[:6000]

        try:
            output = inferencer.generate(paper_text)

            # 简单校验：确认有 <file> 标签
            if "<file" not in output:
                print(f"  [WARN] No <file> tags in output, saving raw text")

            out_path.write_text(output, encoding="utf-8")
            print(f"  [OK] Saved to {out_path}")
            processed += 1

        except Exception as e:
            print(f"  [FAIL] Error: {e}")
            failed += 1
            continue

    print(f"\n{'='*50}")
    print(f"Done: {processed} processed, {skipped} skipped, {failed} failed")
    print(f"Output dir: {output_dir}")


# ── 单文件推理（供 build_xkg.py 直接导入调用）──────────────────────────────

def generate_core_code(
    paper_text: str,
    model_path: Optional[str] = None,
    max_tokens: int = 2048,
    _inferencer_cache: dict = {},
) -> str:
    """
    单篇论文推理接口，带模块级缓存避免重复加载模型。

        from qwen_infer import generate_core_code
        output = generate_core_code(paper_text, model_path="/root/autodl-tmp/models/...")
    """
    cache_key = model_path or os.environ.get("QWEN_API_BASE", "local")
    if cache_key not in _inferencer_cache:
        _inferencer_cache[cache_key] = get_inferencer(model_path=model_path, max_tokens=max_tokens)
    return _inferencer_cache[cache_key].generate(paper_text)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Qwen2.5-Coder-7B fine-tuned model on cleaned paper texts"
    )
    parser.add_argument(
        "--input_dir",
        default="data/training_set_txts",
        help="Directory of cleaned .txt files",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/qwen",
        help="Directory to save generated .xml files",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to merged model weights (default: /root/autodl-tmp/models/Qwen2.5-7B-paper2Xcode)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max new tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Re-process files even if output already exists",
    )
    parser.add_argument(
        "--single",
        default=None,
        help="Process a single .txt file instead of a directory",
    )
    args = parser.parse_args()

    inferencer = get_inferencer(model_path=args.model_path, max_tokens=args.max_tokens)

    if args.single:
        txt_path = Path(args.single)
        paper_text = txt_path.read_text(encoding="utf-8")[:6000]
        output = inferencer.generate(paper_text)
        print(output)
    else:
        process_directory(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            inferencer=inferencer,
            skip_existing=not args.no_skip,
        )


if __name__ == "__main__":
    main()
