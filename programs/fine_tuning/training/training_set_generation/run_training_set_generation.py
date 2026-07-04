"""
run_training_set_generation.py — 训练集生成编排器

基于 ACPP (Algorithm Code Purification Pipeline) 生成 Alpaca 格式训练集：
  1. clone 论文对应代码仓库
  2. AST 粗筛提取候选核心代码块
  3. 单次 CoT LLM 调用：提取算法规格 → 重构为纯净 PyTorch
  4. 沙盒执行 Dummy Test，失败则自反思修复
  5. 与论文文本组装并序列化为 Alpaca 格式

支持增量生成，支持两种测试模式：
  • --test      : 完整测试流程（读取测试 txt → ACPP 管线 → 组装）
  • --mock-test : 模拟测试流程（读取测试 txt → 占位 output → 组装）
  • （无标志）  : 生产流程（读取 data/training_set_txts/ → ACPP 管线 → 组装）

输出路径：
  生产模式  → data/training_set.json
  --test    → data/test/outputs/training_set.json
  --mock-test → data/test/outputs/mock_training_set.json

进度文件（自动创建）：
  生产模式  → data/.training_set_progress.json
  测试模式  → data/test/outputs/.{output_stem}_progress.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_PATH)

from distill_code_repository import (
    cloned_repo,
    ASTFilter,
    CodePurifier,
    SandboxRunner,
    strip_main_block,
    wrap_file_tags,
)
from training_set_assemblage import (
    load_paper_data,
    assemble_alpaca_item,
    load_progress,
    save_progress,
    load_existing_dataset,
    save_dataset_incremental,
)


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def run_mock_assembly(
    paper_data_list: List[Dict[str, str]],
    output_file: Path,
    progress_file: Path,
) -> int:
    """
    模拟组装模式：跳过 clone / ACPP，直接用占位 output 组装 Alpaca 条目。
    返回本次新增条数。
    """
    dataset = load_existing_dataset(output_file)
    processed_urls = load_progress(progress_file)
    start_count = len(dataset)

    remaining = [item for item in paper_data_list if item["github_url"] not in processed_urls]
    logger.info("mock-test: %d papers to assemble", len(remaining))

    for idx, item in enumerate(remaining, 1):
        repo_name = item["repo_name"]
        github_url = item["github_url"]
        paper_text = item["paper_text"]

        logger.info("[%d/%d] mock-assemble %s", idx, len(remaining), repo_name)

        placeholder = (
            f'<file name="{repo_name}_mock.py">\n'
            f'  # Placeholder distilled code for {repo_name}\n'
            f'  # Source: {github_url}\n'
            f'</file>'
        )

        alpaca_item = assemble_alpaca_item(paper_text, placeholder)
        dataset.append(alpaca_item)
        save_dataset_incremental(dataset, output_file)
        save_progress(progress_file, github_url)

        logger.info("  saved - total: %d records", len(dataset))

    return len(dataset) - start_count


def run_full_pipeline(
    paper_data_list: List[Dict[str, str]],
    output_file: Path,
    progress_file: Path,
    model_name: str,
    tmp_dir: Path,
) -> int:
    """
    完整 ACPP 流程：clone → AST 粗筛 → CoT 净化 → 沙盒验证 → 组装 → 增量保存。
    返回本次新增条数。
    """
    dataset = load_existing_dataset(output_file)
    processed_urls = load_progress(progress_file)
    start_count = len(dataset)

    remaining = [item for item in paper_data_list if item["github_url"] not in processed_urls]
    logger.info("full pipeline: %d papers to process", len(remaining))

    if not remaining:
        logger.info("nothing to do, exiting")
        return 0

    tmp_dir.mkdir(parents=True, exist_ok=True)
    purifier = CodePurifier(model_name=model_name)

    for idx, item in enumerate(remaining, 1):
        repo_name = item["repo_name"]
        github_url = item["github_url"]
        paper_text = item["paper_text"]

        logger.info("[%d/%d] %s", idx, len(remaining), repo_name)

        try:
            with cloned_repo(github_url, tmp_dir) as repo_path:
                candidates = ASTFilter.extract_candidates(repo_path)
        except Exception as exc:
            logger.error("  clone failed: %s", exc)
            continue

        if not candidates:
            logger.warning("  no candidate core code blocks found, skipping")
            continue

        logger.info("  AST filter: %d candidate blocks", len(candidates))

        # 合并候选块传给 CoT 净化器
        combined_code = "\n\n".join(candidates)

        try:
            logger.info("  purifying ...")
            clean_code = purifier.purify(combined_code)
        except Exception as exc:
            logger.error("  purification failed: %s", exc)
            continue

        try:
            logger.info("  sandbox verifying ...")
            verified_code, passed = SandboxRunner.verify_and_reflect(
                clean_code, purifier, tmp_dir=tmp_dir
            )
        except Exception as exc:
            logger.error("  sandbox runner failed: %s", exc)
            continue

        if not passed:
            logger.warning("  sandbox verification failed after retries, skipping")
            continue

        # 后处理：移除测试块，包裹 <file> 标签
        final_code = strip_main_block(verified_code)
        final_code = wrap_file_tags(final_code, filename=f"{repo_name}_core.py")

        alpaca_item = assemble_alpaca_item(paper_text, final_code)
        dataset.append(alpaca_item)
        save_dataset_incremental(dataset, output_file)
        save_progress(progress_file, github_url)

        logger.info("  saved - total: %d records", len(dataset))

    return len(dataset) - start_count


def main():
    ap = argparse.ArgumentParser(
        description="Generate Alpaca-format training set from cleaned paper texts and GitHub repos."
    )
    ap.add_argument(
        "--info-json", type=str,
        default=str(_PROJECT_ROOT / "data" / "dataset_2024.json"),
        help="Path to dataset info JSON (default: data/dataset_2024.json)",
    )
    ap.add_argument(
        "--txt-dir", type=str,
        default=str(_PROJECT_ROOT / "data" / "training_set_txts"),
        help="Directory containing cleaned .txt files (default: data/training_set_txts)",
    )
    ap.add_argument(
        "--output", type=str,
        default=str(_PROJECT_ROOT / "data" / "training_set.json"),
        help="Output Alpaca JSON path (default: data/training_set.json)",
    )
    ap.add_argument(
        "--progress-file", type=str,
        help="Progress tracking JSON path (auto-derived from --output if omitted)",
    )
    ap.add_argument(
        "--tmp-dir", type=str,
        default=str(_PROJECT_ROOT / "data" / "tmp"),
        help="Temporary directory for git clones and sandbox files (default: data/tmp)",
    )
    ap.add_argument(
        "--model", type=str,
        default="deepseek/deepseek-v4-pro",
        help="OpenRouter model name (default: deepseek/deepseek-v4-pro)",
    )
    ap.add_argument(
        "--test", action="store_true",
        help="Run in test mode: use data/test/examples/ and run full ACPP pipeline",
    )
    ap.add_argument(
        "--mock-test", action="store_true",
        help="Run in mock-test mode: use data/test/examples/ and skip LLM distillation",
    )
    args = ap.parse_args()

    # ── 路径解析 ─────────────────────────────────────────
    info_json = Path(args.info_json)

    if args.test or args.mock_test:
        txt_dir = _PROJECT_ROOT / "data" / "test" / "examples"
        output_file = _PROJECT_ROOT / "data" / "test" / "outputs" / (
            "mock_training_set.json" if args.mock_test else "training_set.json"
        )
    else:
        txt_dir = Path(args.txt_dir)
        output_file = Path(args.output)

    if args.progress_file:
        progress_file = Path(args.progress_file)
    else:
        progress_file = output_file.with_name("." + output_file.stem + "_progress.json")

    tmp_dir = Path(args.tmp_dir)

    # ── 前置检查 ─────────────────────────────────────────
    if not info_json.exists():
        logger.error("dataset info JSON not found: %s", info_json)
        sys.exit(1)
    if not txt_dir.is_dir():
        logger.error("txt directory not found: %s", txt_dir)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Training Set Generation")
    logger.info("  mode:   %s", "mock-test" if args.mock_test else ("test" if args.test else "production"))
    logger.info("  json:   %s", info_json)
    logger.info("  txt:    %s", txt_dir)
    logger.info("  out:    %s", output_file)
    logger.info("  tmp:    %s", tmp_dir)
    logger.info("  model:  %s", args.model)
    logger.info("=" * 60)

    # ── 加载数据 ─────────────────────────────────────────
    try:
        paper_data_list = load_paper_data(info_json, txt_dir)
    except Exception as exc:
        logger.error("failed to load paper data: %s", exc)
        sys.exit(1)

    if not paper_data_list:
        logger.warning("no papers to process (all missing or unmatched), exiting")
        sys.exit(0)

    # ── 执行 ─────────────────────────────────────────────
    if args.mock_test:
        new_count = run_mock_assembly(paper_data_list, output_file, progress_file)
    else:
        new_count = run_full_pipeline(
            paper_data_list,
            output_file,
            progress_file,
            model_name=args.model,
            tmp_dir=tmp_dir,
        )

    logger.info("=" * 60)
    logger.info("done: %d new records", new_count)
    logger.info("output: %s", output_file)


if __name__ == "__main__":
    main()
