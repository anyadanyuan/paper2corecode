"""
model_loader.py - 模型加载器

统一加载基座模型或微调模型，支持：
- 本地 transformers 加载
- vLLM / OpenAI-compatible API 调用
"""

from __future__ import annotations

import os
from typing import Optional

import torch


class ModelLoader:
    """统一模型加载器"""

    def __init__(
        self,
        model_path: str,
        use_api: bool = False,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        初始化模型加载器。

        Args:
            model_path: 本地路径或 HuggingFace model_id
            use_api: 是否通过 API 调用（vLLM/OpenAI-compatible）
            api_base: API 地址（如 http://localhost:8000/v1）
            api_key: API 密钥（可选）
        """
        self.model_path = model_path
        self.use_api = use_api
        self.api_base = api_base or "http://localhost:8000/v1"
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")

        self.model = None
        self.tokenizer = None
        self.client = None

        if use_api:
            self._init_api()
        else:
            self._init_local()

    def _init_local(self):
        """初始化本地模型加载"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[ModelLoader] Loading model from {self.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        print("[ModelLoader] Model loaded successfully.")

    def _init_api(self):
        """初始化 API 客户端"""
        from openai import OpenAI

        print(f"[ModelLoader] Using API at {self.api_base}")

        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )

    def generate(
        self,
        paper_text: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_context_length: int = 4096,
    ) -> str:
        """
        生成代码。使用与训练时一致的 ChatML 格式 + Alpaca instruction。
        """
        paper_text = self._truncate_paper_text(
            paper_text, max_tokens, max_context_length
        )

        instruction = (
            "You are an expert AI researcher. "
            "Implement the core PyTorch model architecture based on "
            "the following paper excerpts. Output the code using <file> tags."
        )
        user_content = f"{instruction}\n\n{paper_text}"
        messages = [{"role": "user", "content": user_content}]

        if self.use_api:
            return self._generate_api(messages, max_tokens, temperature, top_p)
        else:
            return self._generate_local(messages, max_tokens, temperature, top_p)

    def _truncate_paper_text(
        self, paper_text: str, max_tokens: int, max_context_length: int
    ) -> str:
        """
        智能截断论文文本以适应上下文限制。

        策略：优先保留 Abstract 和 Method 部分的开头。
        """
        # 预留 tokens：instruction (~80) + format overhead (~20) = 100
        max_input_tokens = max_context_length - max_tokens - 100

        # 粗略估算：1 token ≈ 4 chars（英文），保守起见用 3
        max_input_chars = max_input_tokens * 3

        if len(paper_text) <= max_input_chars:
            return paper_text

        print(
            f"  [WARNING] Paper text too long ({len(paper_text)} chars), truncating to {max_input_chars} chars"
        )

        # 尝试智能截断：保留 Abstract 和 Method 的前半部分
        import re

        # 查找 Abstract 和 Method 章节
        abstract_match = re.search(
            r"\[Abstract\](.*?)(?=\[|$)", paper_text, re.DOTALL | re.IGNORECASE
        )
        method_match = re.search(
            r"\[Method\](.*?)(?=\[|$)", paper_text, re.DOTALL | re.IGNORECASE
        )

        if abstract_match and method_match:
            # 两个都找到，优先保留它们
            abstract_text = abstract_match.group(0)
            method_text = method_match.group(0)

            # 如果两者都太长，按比例截断
            total_len = len(abstract_text) + len(method_text)
            if total_len > max_input_chars:
                abstract_ratio = len(abstract_text) / total_len
                abstract_limit = int(
                    max_input_chars * abstract_ratio * 0.8
                )  # Abstract 占 80% 比例
                method_limit = max_input_chars - abstract_limit

                abstract_text = abstract_text[:abstract_limit]
                method_text = method_text[:method_limit]

            return f"{abstract_text}\n\n{method_text}\n\n[...truncated]"
        else:
            # 没有明确章节，简单截断
            return paper_text[:max_input_chars] + "\n\n[...truncated]"

    def _generate_local(
        self,
        messages: list,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """本地模型生成（ChatML 格式）"""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def _generate_api(
        self,
        messages: list,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """API 模式生成（ChatML / chat completions）"""
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content or ""
