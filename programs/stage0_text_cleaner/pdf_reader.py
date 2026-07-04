"""
PDF Reader - 读取论文PDF并转换为结构化JSON。

采用策略模式设计，解析器可插拔（s2orc-doc2json / MinerU / ...）。
支持单文件和批量处理两种模式。
"""

import json
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

_S2ORC_PATH = Path(__file__).resolve().parent.parent.parent / "s2orc-doc2json"
if str(_S2ORC_PATH) not in sys.path:
    sys.path.insert(0, str(_S2ORC_PATH))


# ──────────────────────────────────────────────
#  Abstract Parser Interface
# ──────────────────────────────────────────────

class PDFParser(ABC):
    """PDF解析器抽象基类。所有解析器须实现 parse() 方法。"""

    @abstractmethod
    def parse(self, pdf_path: str) -> Dict:
        """
        解析PDF文件并返回原始解析结果。

        Args:
            pdf_path: PDF文件路径

        Returns:
            解析器原始输出（格式因实现而异）
        """
        ...


# ──────────────────────────────────────────────
#  S2ORC Parser
# ──────────────────────────────────────────────

class S2OrcPDFParser(PDFParser):
    """基于 s2orc-doc2json + GROBID 的PDF解析器。

    要求 GROBID 服务在 grobid_server:grobid_port 上运行。
    可通过 Docker 启动: docker run -d -p 8070:8070 grobid/grobid:0.9.0-full
    """

    def __init__(
        self,
        grobid_server: str = "localhost",
        grobid_port: str = "8070",
    ):
        from doc2json.grobid2json.grobid.grobid_client import DEFAULT_GROBID_CONFIG

        self._grobid_config = dict(DEFAULT_GROBID_CONFIG)
        self._grobid_config.update({
            "grobid_server": grobid_server,
            "grobid_port": grobid_port,
        })

    def parse(self, pdf_path: str) -> Dict:
        from doc2json.grobid2json.process_pdf import process_pdf_stream

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {pdf_path}")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        return process_pdf_stream(
            input_file=pdf_path.name,
            sha="",
            input_stream=pdf_bytes,
            grobid_config=self._grobid_config,
        )


# ──────────────────────────────────────────────
#  MinerU Parser (placeholder)
# ──────────────────────────────────────────────

class MinerUPDFParser(PDFParser):
    """基于 MinerU 的PDF解析器（预留接口，待后续实现）。

    安装: pip install mineru[all]
    参考: https://github.com/opendatalab/MinerU
    """

    def parse(self, pdf_path: str) -> Dict:
        raise NotImplementedError(
            "MinerU parser is not yet implemented. "
            "To add MinerU support, implement parse() to call the mineru REST API "
            "or CLI and transform content_list.json into the standard format."
        )


# ──────────────────────────────────────────────
#  Format Transformation
# ──────────────────────────────────────────────

def _extract_abstract(s2orc_abstract) -> str:
    """从 s2orc JSON 中提取摘要文本。

    s2orc release_json() 中 abstract 可能是:
      - 字符串（顶层，已合并）
      - 段落对象列表（pdf_parse.abstract 中）
    """
    if isinstance(s2orc_abstract, str):
        return s2orc_abstract.strip()
    if isinstance(s2orc_abstract, list):
        parts = [
            p.get("text", "").strip()
            for p in s2orc_abstract
            if isinstance(p, dict) and p.get("text")
        ]
        return "\n".join(parts)
    return ""


def _sec_num_key(sec_num: str) -> List[int]:
    """将 sec_num 转换为排序用的整数列表。"""
    try:
        return [int(n) for n in sec_num.split(".")]
    except (ValueError, AttributeError):
        return [0]


def _make_section_key(prefix: str, section_name: str) -> str:
    """生成 section 的字典 key：{sec_num}_{sanitized_name}"""
    safe = section_name.strip().replace(" ", "_").replace("/", "_").replace("::", "_") if section_name else ""
    if safe:
        return f"{prefix}_{safe}"
    return f"{prefix}_Section"


def _build_section_tree(body_paragraphs: List[Dict]) -> Dict:
    """将 s2orc 平铺段落列表构建为按 sec_num 嵌套的 dict。

    算法：
      1. 按 sec_num 分组，合并同章节段落文本
      2. 建立父子关系（sec_num B 是 A 的子节点 iff B 以 "A." 开头且比 A 深一层）
      3. 递归构建嵌套 dict

    规则：
      - 无子节点的 section 叶 → 字符串值
      - 有子节点的 section → dict，自身文本放在 "_text" 键

    Args:
        body_paragraphs: s2orc pdf_parse.body_text 列表

    Returns:
        嵌套的 tree dict
    """
    groups = defaultdict(list)
    for para in body_paragraphs:
        sec_num = para.get("sec_num")
        if sec_num is not None:
            groups[sec_num].append(para)

    if not groups:
        combined = "\n".join(
            p.get("text", "") for p in body_paragraphs
            if isinstance(p, dict) and p.get("text")
        ).strip()
        return {"_unnumbered": combined} if combined else {}

    all_sec_nums = set(groups.keys())

    # 找出每个 sec_num 的直接父节点
    parents = {}
    roots = []
    for sn in sorted(all_sec_nums, key=_sec_num_key):
        depth = sn.count(".")
        best_parent = None
        for candidate in all_sec_nums:
            if candidate == sn:
                continue
            if sn.startswith(candidate + "."):
                candidate_depth = candidate.count(".")
                if candidate_depth == depth - 1:
                    best_parent = candidate
                    break
                if best_parent is None or candidate_depth > best_parent.count("."):
                    best_parent = candidate
        if best_parent:
            parents.setdefault(best_parent, []).append(sn)
        else:
            roots.append(sn)

    roots.sort(key=_sec_num_key)

    def build_node(sec_num: str):
        paras = groups[sec_num]
        section_name = paras[0].get("section", "")
        combined = "\n".join(
            p.get("text", "") for p in paras
            if isinstance(p, dict) and p.get("text")
        )
        key = _make_section_key(sec_num, section_name)

        children = parents.get(sec_num, [])
        if children:
            sub = {}
            for child in sorted(children, key=_sec_num_key):
                child_key, child_value = build_node(child)
                sub[child_key] = child_value
            if combined:
                sub["_text"] = combined
            return key, sub
        else:
            return key, combined

    tree = {}
    for root in roots:
        k, v = build_node(root)
        tree[k] = v

    return tree


def _transform_s2orc_to_standard(s2orc_json: Dict, paper_id: str) -> Dict:
    """将 s2orc-doc2json release_json 输出转换为项目标准格式。

    标准格式:
    {
        "paper_id": "...",
        "title": "...",
        "authors": [{"first": "...", "middle": [], "last": "...", "affiliation": {...}}],
        "abstract": "...",
        "body": {
            "1_Introduction": "text",
            "2_Method": {
                "_text": "overview text",
                "2.1_Data": "text",
                "2.2_Model": "text"
            }
        },
        "references": [{"id": "...", "title": "...", "authors": [...], "year": ..., "venue": "..."}]
    }
    """
    output = {"paper_id": paper_id}

    output["title"] = s2orc_json.get("title", "")

    raw_authors = s2orc_json.get("authors", [])
    output["authors"] = [
        {
            "first": a.get("first", ""),
            "middle": a.get("middle", []),
            "last": a.get("last", ""),
            "affiliation": a.get("affiliation", {}),
        }
        for a in raw_authors
        if isinstance(a, dict)
    ]

    output["abstract"] = _extract_abstract(s2orc_json.get("abstract", ""))

    pdf_parse = s2orc_json.get("pdf_parse", {})
    body_text = pdf_parse.get("body_text", [])
    output["body"] = _build_section_tree(body_text)

    bib_entries = pdf_parse.get("bib_entries", {})
    output["references"] = [
        {
            "id": bib_id,
            "title": bib.get("title", ""),
            "authors": bib.get("authors", []),
            "year": bib.get("year"),
            "venue": bib.get("venue", ""),
        }
        for bib_id, bib in bib_entries.items()
        if isinstance(bib, dict)
    ]

    return output


# ──────────────────────────────────────────────
#  PDF Reader
# ──────────────────────────────────────────────

class PDFReader:
    """PDF读取器，管理解析器并将结果转换、输出。"""

    _PARSER_REGISTRY = {
        "s2orc": S2OrcPDFParser,
        "mineru": MinerUPDFParser,
    }

    @staticmethod
    def list_parsers() -> List[str]:
        """列出所有已注册的解析器名称。"""
        return list(PDFReader._PARSER_REGISTRY.keys())

    @staticmethod
    def register_parser(name: str, parser_cls: type):
        """注册新的解析器类。"""
        if not issubclass(parser_cls, PDFParser):
            raise TypeError(f"{parser_cls} must be a subclass of PDFParser")
        PDFReader._PARSER_REGISTRY[name] = parser_cls

    def __init__(
        self,
        parser: Optional[PDFParser] = None,
        parser_name: str = "s2orc",
        **parser_kwargs,
    ):
        """
        Args:
            parser: 直接传入解析器实例（优先级最高）
            parser_name: 按名称从注册表选择解析器
            **parser_kwargs: 传递给解析器构造函数的参数
        """
        if parser is not None:
            self._parser = parser
            self._parser_name = parser.__class__.__name__
        elif parser_name in self._PARSER_REGISTRY:
            self._parser = self._PARSER_REGISTRY[parser_name](**parser_kwargs)
            self._parser_name = parser_name
        else:
            raise ValueError(
                f"Unknown parser: '{parser_name}'. "
                f"Available: {self._PARSER_REGISTRY.keys()}"
            )

    def read(self, pdf_path: str, output_dir: Optional[str] = None) -> Dict:
        """解析单个PDF并返回标准JSON。

        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录（可选），若指定则写入 {output_dir}/{paper_id}.json

        Returns:
            标准格式的 dict
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        paper_id = pdf_path.stem

        print(f"[PARSING] {pdf_path.name}  (parser: {self._parser_name})")
        raw = self._parser.parse(str(pdf_path))
        result = _transform_s2orc_to_standard(raw, paper_id)

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            output_file = out_path / f"{paper_id}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[SAVED] {output_file}")

        return result

    def batch_read(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.pdf",
    ) -> List[Dict]:
        """批量解析目录下的PDF文件。

        Args:
            input_dir: 包含PDF文件的目录
            output_dir: JSON输出目录
            pattern: glob文件匹配模式（默认 *.pdf）

        Returns:
            所有解析结果列表
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {input_dir}")

        pdf_files = sorted(input_dir.glob(pattern))
        if not pdf_files:
            print(f"[WARN] No PDF files matching '{pattern}' in {input_dir}")
            return []

        print(f"[BATCH] {len(pdf_files)} PDF file(s) found in {input_dir}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = []
        succeeded = 0
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] {pdf_path.name}")
            try:
                result = self.read(str(pdf_path), output_dir=output_dir)
                results.append(result)
                succeeded += 1
            except Exception as exc:
                print(f"  [ERROR] {exc}")
                results.append({"paper_id": pdf_path.stem, "error": str(exc)})

        print(f"\n[BATCH] Complete: {succeeded}/{len(pdf_files)} succeeded")
        return results


# ──────────────────────────────────────────────
#  Top-level convenience functions
# ──────────────────────────────────────────────

def convert_pdf_to_json(
    pdf_path: str,
    output_dir: Optional[str] = None,
    parser: Optional[PDFParser] = None,
    parser_name: str = "s2orc",
    **parser_kwargs,
) -> Dict:
    """解析单个PDF并转换为标准JSON格式。

    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录（可选）
        parser: 解析器实例（可选，优先级高于 parser_name）
        parser_name: 解析器名称（"s2orc" 或 "mineru"）
        **parser_kwargs: 解析器参数（如 grobid_server, grobid_port）

    Returns:
        标准格式解析结果 dict
    """
    return PDFReader(parser=parser, parser_name=parser_name, **parser_kwargs).read(
        pdf_path, output_dir=output_dir
    )


def batch_convert_pdfs(
    input_dir: str,
    output_dir: str,
    parser: Optional[PDFParser] = None,
    parser_name: str = "s2orc",
    pattern: str = "*.pdf",
    **parser_kwargs,
) -> List[Dict]:
    """批量解析目录下的PDF文件。

    Args:
        input_dir: PDF目录路径
        output_dir: JSON输出目录
        parser: 解析器实例（可选）
        parser_name: 解析器名称
        pattern: PDF文件匹配模式
        **parser_kwargs: 解析器参数

    Returns:
        解析结果列表
    """
    return PDFReader(parser=parser, parser_name=parser_name, **parser_kwargs).batch_read(
        input_dir, output_dir, pattern=pattern
    )
