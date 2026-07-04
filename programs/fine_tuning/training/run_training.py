#!/usr/bin/env python3
"""
run_training.py — SFT 训练编排器（LLaMA-Factory + TaskMonitor 监控）。

服务器目录约定：
    /root/paper2XAgent/         程序代码根目录
    /root/autodl-tmp/           数据与输出目录
    /root/LLaMA-Factory/        LLaMA-Factory 源码/运行目录（无自定义代码）

流程：
    1. 环境检查          — Python、关键包、GPU、llamafactory-cli、模型缓存、训练数据
    2. 数据集注册        — 复制 training_set.json 到 LLaMA-Factory/data/，注册 dataset_info.json
    3. 训练配置生成      — 从 train_qwen.yaml 模板生成运行时配置（替换路径变量）
    4. 执行训练          — 用 TaskMonitor 记录显存与耗时
    5. 结果汇总          — 输出适配器路径、监控摘要

用法：
    python run_training.py --check-only                        # 仅检查环境
    python run_training.py                                     # 完整训练
    python run_training.py \
        --train-json /root/autodl-tmp/data/training_set.json \
        --llamafactory-dir /root/LLaMA-Factory \
        --output-dir /root/autodl-tmp/lora_adapter_output \
        --log-dir /root/autodl-tmp/monitor_logs \
        --hf-cache /root/autodl-tmp/huggingface_cache
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 确保能导入项目根目录下的 utils
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils.nvidia_utils import TaskMonitor, get_memory_snapshot

# ═══════════════════════════════════════════════════════════════════════════════
#  默认路径（基于服务器目录约定）
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    "llamafactory_dir": "/root/LLaMA-Factory",
    "train_json": str(_PROJECT_ROOT / "data" / "train_dataset.json"),
    "output_dir": "/root/autodl-tmp/lora_adapter_output",
    "log_dir": "/root/autodl-tmp/monitor_logs",
    "hf_cache": "/root/autodl-tmp/huggingface_cache",
    "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "dataset_key": "paper2CoreCode_sft",
    "hf_tokenizer_parallelism": "false",
}

# 期望的关键包及最低版本
REQUIRED_PACKAGES: List[Tuple[str, str]] = [
    ("torch", "2.0.0"),
    ("transformers", "4.40.0"),
    ("peft", "0.10.0"),
    ("bitsandbytes", "0.43.0"),
    ("accelerate", "0.29.0"),
]

# 训练配置模板中的占位符
CONFIG_TEMPLATE = Path(__file__).with_name("train_qwen.yaml")

# ═══════════════════════════════════════════════════════════════════════════════
#  日志
# ═══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("run_training")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(name)s]  %(message)s",
        datefmt="%H:%M:%S",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  环境检查
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EnvCheckResult:
    name: str
    ok: bool
    message: str


def check_python_version(min_major: int = 3, min_minor: int = 9) -> EnvCheckResult:
    ok = sys.version_info >= (min_major, min_minor)
    return EnvCheckResult(
        name="Python version",
        ok=ok,
        message=f"{sys.version.split()[0]} (require >= {min_major}.{min_minor})",
    )


def check_installed_package(name: str, min_version: str) -> EnvCheckResult:
    try:
        import importlib.metadata as metadata

        ver = metadata.version(name)
        ok = _version_satisfies(ver, min_version)
        return EnvCheckResult(
            name=f"package: {name}",
            ok=ok,
            message=f"{ver} (require >= {min_version})",
        )
    except Exception as exc:
        return EnvCheckResult(
            name=f"package: {name}",
            ok=False,
            message=f"not installed or import error: {exc}",
        )


def check_llamafactory_cli() -> EnvCheckResult:
    cmd = shutil.which("llamafactory-cli")
    if not cmd:
        return EnvCheckResult(
            name="llamafactory-cli",
            ok=False,
            message="not found in PATH; please activate the LLaMA-Factory environment",
        )
    try:
        out = subprocess.run(
            ["llamafactory-cli", "version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if out.returncode == 0:
            return EnvCheckResult(
                name="llamafactory-cli",
                ok=True,
                message=f"{cmd} (v{out.stdout.strip()})",
            )
    except Exception:
        pass
    return EnvCheckResult(
        name="llamafactory-cli",
        ok=True,
        message=f"{cmd} (binary found, --help skipped due to known argparse bug)",
    )


def check_gpu() -> EnvCheckResult:
    snap = get_memory_snapshot()
    if snap:
        return EnvCheckResult(
            name="GPU (CUDA)",
            ok=True,
            message=f"GPU 0: {snap['reserved_mb'] / 1024:.2f}GB / {snap['total_mb'] / 1024:.2f}GB used",
        )
    return EnvCheckResult(
        name="GPU (CUDA)",
        ok=False,
        message="nvidia-smi / PyTorch CUDA not available",
    )


def check_model_cache(hf_cache: Path, model_name: str) -> EnvCheckResult:
    escaped = model_name.replace("/", "--")
    cache_dir = hf_cache / "hub" / f"models--{escaped}"
    if cache_dir.is_dir():
        return EnvCheckResult(
            name="Model cache",
            ok=True,
            message=f"found {cache_dir}",
        )
    return EnvCheckResult(
        name="Model cache",
        ok=False,
        message=f"not found at {cache_dir}; model will be downloaded during training",
    )


def check_training_data(train_json: Path) -> EnvCheckResult:
    if not train_json.exists():
        return EnvCheckResult(
            name="Training data",
            ok=False,
            message=f"not found: {train_json}",
        )
    try:
        with open(train_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return EnvCheckResult(
                name="Training data",
                ok=False,
                message="JSON root is not a list",
            )
        if len(data) == 0:
            return EnvCheckResult(
                name="Training data",
                ok=False,
                message="JSON list is empty",
            )
        # 简单校验 Alpaca 字段
        required = {"instruction", "input", "output"}
        bad = [i for i, item in enumerate(data[:5]) if not required.issubset(item.keys())]
        if bad:
            return EnvCheckResult(
                name="Training data",
                ok=False,
                message=f"samples missing instruction/input/output keys: {bad}",
            )
        return EnvCheckResult(
            name="Training data",
            ok=True,
            message=f"{len(data)} samples at {train_json}",
        )
    except Exception as exc:
        return EnvCheckResult(
            name="Training data",
            ok=False,
            message=f"failed to read {train_json}: {exc}",
        )


def run_all_checks(args: argparse.Namespace) -> Tuple[bool, List[EnvCheckResult]]:
    results: List[EnvCheckResult] = []
    results.append(check_python_version())
    for pkg, ver in REQUIRED_PACKAGES:
        results.append(check_installed_package(pkg, ver))
    results.append(check_gpu())
    results.append(check_llamafactory_cli())
    results.append(check_model_cache(Path(args.hf_cache), args.model))
    results.append(check_training_data(Path(args.train_json)))

    all_ok = all(r.ok for r in results)
    return all_ok, results


def print_check_report(all_ok: bool, results: List[EnvCheckResult]) -> None:
    status = "PASS" if all_ok else "FAIL"
    logger.info("=" * 60)
    logger.info("Environment check report: %s", status)
    logger.info("=" * 60)
    for r in results:
        symbol = "[OK]" if r.ok else "[FAIL]"
        logger.info("  %s %-22s %s", symbol, r.name + ":", r.message)
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
#  数据集注册
# ═══════════════════════════════════════════════════════════════════════════════

def register_dataset(
    llamafactory_dir: Path,
    dataset_key: str,
    train_json: Path,
) -> Path:
    """
    将 training_set.json 复制到 LLaMA-Factory/data/，并在 dataset_info.json 中注册。
    幂等：若已存在同名数据集且配置一致则跳过。
    """
    lf_data_dir = llamafactory_dir / "data"
    lf_data_dir.mkdir(parents=True, exist_ok=True)

    target_json = lf_data_dir / f"{dataset_key}.json"
    shutil.copy2(train_json, target_json)
    logger.info("copied training data: %s → %s", train_json, target_json)

    info_path = lf_data_dir / "dataset_info.json"
    dataset_info: Dict[str, Any] = {}
    if info_path.exists():
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                dataset_info = json.load(f)
        except Exception as exc:
            logger.warning("failed to parse %s: %s; overwriting", info_path, exc)
            dataset_info = {}

    expected_entry = {
        "file_name": target_json.name,
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
        },
    }

    existing = dataset_info.get(dataset_key)
    if existing == expected_entry:
        logger.info("dataset '%s' already registered with same config", dataset_key)
    else:
        if existing:
            logger.info("updating existing dataset registration '%s'", dataset_key)
        dataset_info[dataset_key] = expected_entry
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        logger.info("registered dataset '%s' in %s", dataset_key, info_path)

    return target_json


# ═══════════════════════════════════════════════════════════════════════════════
#  训练配置生成
# ═══════════════════════════════════════════════════════════════════════════════

def generate_train_config(
    template_path: Path,
    output_path: Path,
    model_name: str,
    dataset_key: str,
    output_dir: str,
) -> Path:
    if not template_path.exists():
        raise FileNotFoundError(f"training config template not found: {template_path}")

    template = template_path.read_text(encoding="utf-8")

    # 简单占位符替换
    mapping = {
        "{MODEL_NAME}": model_name,
        "{DATASET_KEY}": dataset_key,
        "{OUTPUT_DIR}": output_dir,
    }
    for placeholder, value in mapping.items():
        template = template.replace(placeholder, value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(template, encoding="utf-8")
    logger.info("generated training config: %s", output_path)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
#  训练执行
# ═══════════════════════════════════════════════════════════════════════════════

def run_training(
    llamafactory_dir: Path,
    config_path: Path,
    log_dir: Path,
    log_file: Path,
) -> int:
    """
    调用 llamafactory-cli train 并启用 TaskMonitor 监控。
    返回命令退出码。
    """
    cmd = ["llamafactory-cli", "train", str(config_path)]

    logger.info("starting SFT training")
    logger.info("  command: %s", " ".join(cmd))
    logger.info("  cwd: %s", llamafactory_dir)
    logger.info("  monitor log dir: %s", log_dir)
    logger.info("  training log file: %s", log_file)

    start = time.time()
    with TaskMonitor(
        name="SFT_paper2CoreCode",
        log_dir=str(log_dir),
        interval=30.0,
    ) as mon:
        mon.note("training started")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            proc = subprocess.run(
                cmd,
                cwd=str(llamafactory_dir),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )
        mon.note(f"training exited with code {proc.returncode}")

    elapsed_min = (time.time() - start) / 60
    logger.info("training finished in %.2f minutes (exit_code=%d)", elapsed_min, proc.returncode)
    return proc.returncode


# ═══════════════════════════════════════════════════════════════════════════════
#  结果汇总
# ═══════════════════════════════════════════════════════════════════════════════

def print_training_summary(
    output_dir: Path,
    log_dir: Path,
    monitor_dir: Optional[Path] = None,
) -> None:
    logger.info("=" * 60)
    logger.info("Training summary")
    logger.info("=" * 60)
    logger.info("  Adapter output: %s", output_dir)

    adapter_config = output_dir / "adapter_config.json"
    if adapter_config.exists():
        logger.info("  [OK] adapter_config.json found")
    else:
        logger.warning("  [FAIL] adapter_config.json not found")

    loss_png = output_dir / "training_loss.png"
    if loss_png.exists():
        logger.info("  Loss plot:       %s", loss_png)

    if monitor_dir and monitor_dir.is_dir():
        summary = monitor_dir / "summary.json"
        records = monitor_dir / "records.jsonl"
        if summary.exists():
            logger.info("  Monitor summary: %s", summary)
        if records.exists():
            logger.info("  Monitor records: %s", records)

    logger.info("=" * 60)


def _version_satisfies(current: str, minimum: str) -> bool:
    """非常简化的版本比较；返回 current >= minimum。"""

    def _to_tuple(v: str) -> Tuple[int, ...]:
        parts = []
        for p in v.split("."):
            # 取纯数字前缀
            digits = ""
            for ch in p:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            parts.append(int(digits) if digits else 0)
        return tuple(parts)

    try:
        return _to_tuple(current) >= _to_tuple(minimum)
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Orchestrate SFT training with LLaMA-Factory and TaskMonitor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--train-json",
        type=str,
        default=DEFAULTS["train_json"],
        help="Path to Alpaca-format training_set.json",
    )
    ap.add_argument(
        "--llamafactory-dir",
        type=str,
        default=DEFAULTS["llamafactory_dir"],
        help="Path to LLaMA-Factory directory",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULTS["output_dir"],
        help="Directory for LoRA adapter output",
    )
    ap.add_argument(
        "--log-dir",
        type=str,
        default=DEFAULTS["log_dir"],
        help="Directory for monitor logs",
    )
    ap.add_argument(
        "--hf-cache",
        type=str,
        default=DEFAULTS["hf_cache"],
        help="HF_HOME cache directory for models/datasets",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=DEFAULTS["model_name"],
        help="Base model name or path",
    )
    ap.add_argument(
        "--dataset-key",
        type=str,
        default=DEFAULTS["dataset_key"],
        help="Dataset key to register in LLaMA-Factory dataset_info.json",
    )
    ap.add_argument(
        "--config-template",
        type=str,
        default=str(CONFIG_TEMPLATE),
        help="Path to train_qwen.yaml template",
    )
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Only run environment checks and exit",
    )
    ap.add_argument(
        "--skip-env-check",
        action="store_true",
        help="Skip environment checks before training",
    )
    return ap.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    logger.info("=" * 60)
    logger.info("paper2CoreCode SFT Training")
    logger.info("=" * 60)
    logger.info("dataset key: %s", args.dataset_key)
    logger.info("train json:  %s", args.train_json)
    logger.info("output dir:  %s", args.output_dir)
    logger.info("log dir:     %s", args.log_dir)
    logger.info("hf cache:    %s", args.hf_cache)
    logger.info("=" * 60)

    # ── 环境检查 ─────────────────────────────────────────────────────────────
    if not args.skip_env_check:
        all_ok, results = run_all_checks(args)
        print_check_report(all_ok, results)
        if args.check_only:
            return 0 if all_ok else 1
        if not all_ok:
            logger.error("environment check failed; aborting. Use --skip-env-check to bypass.")
            return 1
    else:
        logger.warning("environment checks skipped")
        if args.check_only:
            logger.info("--check-only with --skip-env-check: nothing to do")
            return 0

    # ── 路径对象化 ─────────────────────────────────────────────────────────────
    llamafactory_dir = Path(args.llamafactory_dir)
    train_json = Path(args.train_json)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    template_path = Path(args.config_template)
    runtime_config_path = llamafactory_dir / "data" / f"train_{args.dataset_key}.yaml"

    # ── 准备数据 ───────────────────────────────────────────────────────────────
    register_dataset(llamafactory_dir, args.dataset_key, train_json)

    # ── 生成配置 ───────────────────────────────────────────────────────────────
    generate_train_config(
        template_path=template_path,
        output_path=runtime_config_path,
        model_name=args.model,
        dataset_key=args.dataset_key,
        output_dir=args.output_dir,
    )

    # ── 准备输出目录 ───────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 设置训练所需环境变量，并保留旧值用于恢复
    env_backup = {}
    env_overrides = {
        "HF_HOME": args.hf_cache,
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": DEFAULTS["hf_tokenizer_parallelism"],
    }
    for k, v in env_overrides.items():
        env_backup[k] = os.environ.get(k)
        os.environ[k] = v

    # ── 执行训练 ───────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{args.dataset_key}_{timestamp}.log"

    try:
        exit_code = run_training(llamafactory_dir, runtime_config_path, log_dir, log_file)
    finally:
        # 恢复环境变量
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ── 结果汇总 ───────────────────────────────────────────────────────────────
    if log_dir.is_dir():
        # 找到最新的 monitor 目录
        monitor_dirs = sorted(
            [d for d in log_dir.iterdir() if d.is_dir() and d.name.startswith("SFT_paper2CoreCode_")],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        latest_monitor_dir = monitor_dirs[0] if monitor_dirs else None
    else:
        latest_monitor_dir = None

    print_training_summary(output_dir, log_dir, latest_monitor_dir)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
