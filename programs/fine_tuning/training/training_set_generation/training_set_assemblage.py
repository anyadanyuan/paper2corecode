"""
training_set_assemblage.py — 纯组装模块：将论文文本与蒸馏后的代码组装成 Alpaca 格式训练集。

Alpaca 格式：
  {
    "instruction": "You are an expert AI researcher...",
    "input":     "[论文文本]",
    "output":    "<file name=\"...\">...</file>"
  }
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


def load_paper_data(info_json: Path, txt_dir: Path) -> List[Dict[str, str]]:
    """
    读取 dataset_info.json 和 txt_dir 中的 .txt 文件，组装论文数据列表。

    规则：
      1. dataset_info.json 中 repo_name → repo_url 建立映射
      2. 遍历 txt_dir 中的 *.txt，stem 作为 repo_name
      3. 若该 repo_name 在 json 中无对应记录 → 跳过
      4. 若 txt 文件读取失败 → 跳过
      5. 返回列表元素：{"repo_name": str, "paper_text": str, "github_url": str}
    """
    if not info_json.exists():
        raise FileNotFoundError(f"dataset info JSON not found: {info_json}")
    if not txt_dir.is_dir():
        raise NotADirectoryError(f"txt directory not found: {txt_dir}")

    with open(info_json, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    # repo_name → github_url
    url_map: Dict[str, str] = {}
    for conference_papers in dataset_info.values():
        for paper_data in conference_papers:
            repo_name = paper_data.get("repo_name")
            repo_url = paper_data.get("repo_url")
            if repo_name and repo_url:
                url_map[repo_name] = repo_url

    logger.info("dataset info loaded: %d repo entries", len(url_map))

    paper_data_list = []
    for filepath in sorted(txt_dir.glob("*.txt")):
        repo_name = filepath.stem
        github_url = url_map.get(repo_name)

        if not github_url:
            logger.warning(
                "skip '%s': no matching repo_url in %s", repo_name, info_json.name
            )
            continue

        try:
            paper_text = filepath.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("failed to read %s: %s", filepath, exc)
            continue

        paper_data_list.append({
            "repo_name": repo_name,
            "paper_text": paper_text,
            "github_url": github_url,
        })

    logger.info(
        "assembled %d paper entries (skipped %d missing/invalid)",
        len(paper_data_list),
        len(list(txt_dir.glob("*.txt"))) - len(paper_data_list),
    )
    return paper_data_list


def assemble_alpaca_item(paper_text: str, distilled_code: str) -> Dict[str, str]:
    """
    将单篇论文文本与蒸馏代码组装为 Alpaca 格式条目。
    注意：最终输出不含 github_url，该字段仅用于流程编排。
    """
    return {
        "instruction": (
            "You are an expert AI researcher. Implement the core PyTorch model "
            "architecture based on the following paper excerpts. Output the code "
            "using <file> tags."
        ),
        "input": paper_text,
        "output": distilled_code,
    }


def load_progress(progress_file: Path) -> Set[str]:
    """
    读取进度文件，返回已处理的 github_url 集合。
    若文件不存在则返回空集合。
    """
    if not progress_file.exists():
        return set()

    try:
        with open(progress_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        processed = set(data) if isinstance(data, list) else set()
        logger.info("progress file loaded: %d processed repos", len(processed))
        return processed
    except Exception as exc:
        logger.warning("failed to read progress file %s: %s", progress_file, exc)
        return set()


def save_progress(progress_file: Path, github_url: str):
    """
    将单个 github_url 追加到进度文件中。
    """
    processed = load_progress(progress_file)
    if github_url in processed:
        return

    processed.add(github_url)
    try:
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(sorted(processed), f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.error("failed to save progress file %s: %s", progress_file, exc)


def load_existing_dataset(output_file: Path) -> List[Dict]:
    """
    加载已生成的 Alpaca 数据集。返回纯列表（元素不含 github_url）。
    """
    if not output_file.exists():
        return []

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            logger.info("existing dataset: %d records", len(data))
            return data
        else:
            logger.warning("existing dataset format invalid, starting fresh")
            return []
    except Exception as exc:
        logger.warning("failed to read existing dataset %s: %s", output_file, exc)
        return []


def save_dataset_incremental(dataset: List[Dict], output_file: Path):
    """
    将完整数据集增量写入 JSON 文件。
    """
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.error("failed to write dataset to %s: %s", output_file, exc)
