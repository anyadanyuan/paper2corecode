"""
Data Cleaner - 从 pdf_reader.py 输出的 JSON 中提取并结构化正文文本。

输出 Markdown 格式，仅保留 body 字段（丢弃 title/authors/abstract/references）。
Heading 格式: # {sec_num} {section_name}
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


# ──────────────────────────────────────────────
#  Section Key Parsing & Markdown Rendering
# ──────────────────────────────────────────────

def _parse_section_key(key: str):
    """解析 body dict 的 key，返回 (heading_level, heading_text)。

    - 普通 key "1_Introduction" → level=1, heading="1 Introduction"
    - 数字 key "2.1_Architecture" → level=2, heading="2.1 Architecture"
    - fallback key "3_Section" → level=1+1, heading="3 Section"
    - _text key → level=0, heading="" (无 heading，纯段落)
    """
    if key == "_text":
        return 0, ""

    parts = key.split("_", 1)
    sec_num = parts[0]
    name = parts[1] if len(parts) > 1 else ""

    level = sec_num.count(".") + 1

    if name:
        heading = f"{sec_num} {name.replace('_', ' ')}"
    else:
        heading = sec_num

    return level, heading


def _render_body_tree(body: dict) -> str:
    """递归将嵌套 body dict 渲染为 markdown 字符串。

    规则：
      - str 值 → heading + 段落文本
      - dict 值 → heading + 先渲染 _text（若有），再递归渲染子节点
      - _text 键 → 直接输出段落文本（不产生 heading）
    """
    lines = []

    # 先处理 _text（若存在）
    if "_text" in body and isinstance(body["_text"], str) and body["_text"].strip():
        lines.append(body["_text"].strip())
        lines.append("")

    for key, value in body.items():
        if key == "_text":
            continue

        level, heading_text = _parse_section_key(key)

        if level == 0:
            continue

        if isinstance(value, str):
            heading_marker = "#" * level
            lines.append(f"{heading_marker} {heading_text}")
            lines.append("")
            text = value.strip()
            if text:
                lines.append(text)
                lines.append("")

        elif isinstance(value, dict):
            heading_marker = "#" * level
            lines.append(f"{heading_marker} {heading_text}")
            lines.append("")
            lines.append(_render_body_tree(value))

    return "\n".join(lines)


# ──────────────────────────────────────────────
#  PaperDataCleaner
# ──────────────────────────────────────────────

class PaperDataCleaner:
    """论文正文清洗器，将 pdf_reader.py 的 JSON 输出转换为结构化 Markdown。

    --- Future: 关键词过滤体系（暂未启用）---
    # keywords = {
    #     "high_importance": {
    #         "method", "methodology", "approach", "model", "architecture",
    #         "framework", "implementation", "algorithm", "network", "encoder",
    #         "decoder", "attention", "transformer", "layer", "module",
    #         "embedding", "loss", "objective", "training", "optimizer", "inference",
    #     },
    #     "medium_importance": {
    #         "experiment", "experiments", "result", "results", "evaluation",
    #         "benchmark", "dataset", "datasets", "baseline", "baselines",
    #         "ablation", "metric", "metrics", "performance", "accuracy",
    #         "sensitivity", "comparison", "setup", "setting",
    #     },
    #     "low_importance": {
    #         "introduction", "related work", "background", "conclusion",
    #         "discussion", "preliminaries", "limitation", "limitations",
    #         "future work", "appendix", "supplementary", "acknowledgement",
    #         "ethics", "impact", "safeguards", "reference",
    #     },
    # }
    #
    # def _is_useful_section(self, section_name: str) -> bool:
    #     if not section_name:
    #         return False
    #     sec_lower = section_name.lower()
    #     is_target = any(kw in sec_lower for kw in self.target_keywords)
    #     is_drop = any(kw in sec_lower for kw in self.drop_keywords)
    #     return is_target and not is_drop
    """

    def clean_paper(self, raw_json: Dict) -> str:
        """将论文 JSON 转换为结构化 Markdown 文本。

        Args:
            raw_json: pdf_reader.py 输出的标准 JSON

        Returns:
            Markdown 格式的正文文本（仅 body 字段）
        """
        body = raw_json.get("body", {})
        if not body:
            return ""

        return _render_body_tree(body).strip() + "\n"


# ──────────────────────────────────────────────
#  Top-level convenience functions
# ──────────────────────────────────────────────

def convert_paper_to_text(
    json_path: str,
    output_path: Optional[str] = None,
) -> str:
    """将单个论文 JSON 转换为结构化 Markdown 并保存。

    Args:
        json_path: 输入 JSON 文件路径
        output_path: 输出 .txt 路径（可选，默认与输入同目录同名 .txt）

    Returns:
        生成的 Markdown 文本
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cleaner = PaperDataCleaner()
    text = cleaner.clean_paper(raw)

    if output_path is None:
        output_path = str(json_path.with_suffix(".txt"))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    paper_id = raw.get("paper_id", json_path.stem)
    print(f"[CLEANED] {paper_id}: {len(text):,} chars → {out_path}")

    return text


def batch_convert_papers(
    input_dir: str,
    output_dir: str,
    pattern: str = "*.json",
) -> List[str]:
    """批量将目录下的论文 JSON 转换为 Markdown 文本。

    Args:
        input_dir: JSON 目录路径
        output_dir: 输出目录路径
        pattern: glob 匹配模式（默认 *.json）

    Returns:
        所有生成的 Markdown 文本列表
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    json_files = sorted(input_dir.glob(pattern))
    if not json_files:
        print(f"[WARN] No JSON files matching '{pattern}' in {input_dir}")
        return []

    print(f"[BATCH] {len(json_files)} JSON file(s) found in {input_dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = []
    succeeded = 0
    for i, json_path in enumerate(json_files, 1):
        out_path = output_dir / json_path.with_suffix(".txt").name
        print(f"[{i}/{len(json_files)}] {json_path.name}")
        try:
            text = convert_paper_to_text(str(json_path), str(out_path))
            texts.append(text)
            succeeded += 1
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            texts.append("")

    print(f"\n[BATCH] Complete: {succeeded}/{len(json_files)} succeeded")
    return texts


# ──────────────────────────────────────────────
#  End of public API
# ──────────────────────────────────────────────
