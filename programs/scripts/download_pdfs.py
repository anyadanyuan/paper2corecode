#!/usr/bin/env python3
"""
download_pdfs.py — 从 OpenReview / Semantic Scholar / arXiv 爬取论文 PDF。

读取 data/dataset_info.json 中的论文列表，按会议分类下载到：
  data/iclr2024_pdfs/
  data/icml2024_pdfs/
  data/nips2024_pdfs/

文件命名: {repo_name}.pdf

特性:
  - 多源搜索: OpenReview → Semantic Scholar → arXiv
  - 增量运行: 通过 _progress.json 记录状态，可随时中断并重启
  - 速率控制: 请求间隔防止被 ban

用法:
  python download_pdfs.py                           # 下载全部
  python download_pdfs.py --conference iclr2024     # 仅下载 ICLR 2024
  python download_pdfs.py --delay 3 --timeout 30    # 自定义速率/超时
"""

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urljoin

import requests

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_INFO_PATH = BASE_DIR / "data" / "dataset_info.json"
DATA_DIR = BASE_DIR / "data"

CONFERENCE_DIRS = {
    "iclr2024": DATA_DIR / "iclr2024_pdfs",
    "icml2024": DATA_DIR / "icml2024_pdfs",
    "nips2024": DATA_DIR / "nips2024_pdfs",
}

PROGRESS_FILENAME = "_progress.json"

HEADERS = {
    "User-Agent": "paper2XAgent/1.0 (mailto:research@example.com)"
}

# ──────────────────────────────────────────────
#  Title Matching
# ──────────────────────────────────────────────

def _normalize_title(title: str) -> str:
    """规范化标题以用于比较：小写、去多余空白、去尾部句号。"""
    return re.sub(r"\s+", " ", title.lower().strip().rstrip("."))


def _titles_match(title_a: str, title_b: str) -> bool:
    return _normalize_title(title_a) == _normalize_title(title_b)


# ──────────────────────────────────────────────
#  Source 1: OpenReview
# ──────────────────────────────────────────────

OPENREVIEW_SEARCH_URL = "https://api2.openreview.net/notes/search"
OPENREVIEW_PDF_BASE = "https://openreview.net"


def _search_openreview(title: str, timeout: int) -> Optional[str]:
    """在 OpenReview 中搜索论文标题，返回 PDF URL 或 None。"""
    try:
        params = {"term": title, "content": "all", "limit": 5}
        resp = requests.get(
            OPENREVIEW_SEARCH_URL, params=params, headers=HEADERS, timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"    [OpenReview] search error: {exc}")
        return None

    notes = data.get("notes", [])
    if not notes:
        print(f"    [OpenReview] no results for title")
        return None

    for note in notes:
        content = note.get("content", {})
        found_title = (content.get("title", {}) or {}).get("value", "")
        pdf_rel = (content.get("pdf", {}) or {}).get("value", "")

        if _titles_match(found_title, title) and pdf_rel:
            pdf_url = urljoin(OPENREVIEW_PDF_BASE, pdf_rel)
            print(f"    [OpenReview] matched: {found_title[:80]}...")
            return pdf_url

    print(f"    [OpenReview] no exact title match among {len(notes)} results")
    return None


# ──────────────────────────────────────────────
#  Source 2: Semantic Scholar
# ──────────────────────────────────────────────

SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"


def _search_semantic_scholar(title: str, timeout: int) -> Optional[str]:
    """在 Semantic Scholar 中搜索论文，返回 openAccessPdf URL 或 None。"""
    try:
        params = {
            "query": title,
            "limit": 3,
            "fields": "title,openAccessPdf",
        }
        resp = requests.get(
            SEMANTIC_SCHOLAR_SEARCH, params=params, headers=HEADERS, timeout=timeout
        )
        if resp.status_code == 429:
            print("    [SemanticScholar] rate limited (429), skipping")
            return None
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        print(f"    [SemanticScholar] search error: {exc}")
        return None

    papers = data.get("data", [])
    if not papers:
        print(f"    [SemanticScholar] no results")
        return None

    for paper in papers:
        found_title = paper.get("title", "")
        if _titles_match(found_title, title):
            pdf_info = paper.get("openAccessPdf") or {}
            pdf_url = pdf_info.get("url")
            if pdf_url:
                print(f"    [SemanticScholar] matched: {found_title[:80]}...")
                return pdf_url
            else:
                print(f"    [SemanticScholar] title matched but no openAccessPdf")
                continue

    print(f"    [SemanticScholar] no exact title match")
    return None


# ──────────────────────────────────────────────
#  Source 3: arXiv
# ──────────────────────────────────────────────

ARXIV_API_URL = "http://export.arxiv.org/api/query"


def _search_arxiv(title: str, timeout: int) -> Optional[str]:
    """在 arXiv 中按标题搜索，返回 PDF URL 或 None。"""
    try:
        params = {
            "search_query": f'ti:"{title}"',
            "max_results": 3,
        }
        resp = requests.get(ARXIV_API_URL, params=params, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:
        print(f"    [arXiv] search error: {exc}")
        return None

    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)

    if not entries:
        print(f"    [arXiv] no results")
        return None

    for entry in entries:
        title_elem = entry.find("atom:title", ns)
        found_title = title_elem.text.strip() if title_elem is not None else ""

        if _titles_match(found_title, title):
            id_elem = entry.find("atom:id", ns)
            if id_elem is not None:
                arxiv_id = id_elem.text.strip().split("/abs/")[-1]
                # Strip version suffix (v1, v2, etc.)
                arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                print(f"    [arXiv] matched: {found_title[:80]}...")
                return pdf_url

    print(f"    [arXiv] no exact title match")
    return None


# ──────────────────────────────────────────────
#  PDF Downloader
# ──────────────────────────────────────────────

def _download_pdf(pdf_url: str, output_path: Path, timeout: int) -> bool:
    """从 URL 下载 PDF 到指定路径。返回是否成功。"""
    try:
        resp = requests.get(pdf_url, headers=HEADERS, timeout=timeout, stream=True)
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk_data in resp.iter_content(chunk_size=8192):
                if chunk_data:
                    f.write(chunk_data)

        file_size = output_path.stat().st_size
        if file_size < 1000:
            output_path.unlink()
            print(f"    Downloaded file too small ({file_size} bytes), discarding")
            return False

        # Verify PDF magic bytes
        with open(output_path, "rb") as f:
            header = f.read(5)
        if not header.startswith(b"%PDF"):
            output_path.unlink()
            print(f"    File is not a valid PDF, discarding")
            return False

        print(f"    Downloaded: {file_size / 1024:.0f} KB")
        return True

    except Exception as exc:
        print(f"    Download error: {exc}")
        if output_path.exists():
            output_path.unlink()
        return False


# ──────────────────────────────────────────────
#  Progress Tracker (incremental)
# ──────────────────────────────────────────────

def _load_progress(progress_path: Path) -> Dict:
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_progress(progress_path: Path, progress: Dict):
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────
#  Main download logic
# ──────────────────────────────────────────────

def _find_pdf_url(title: str, timeout: int) -> Tuple[Optional[str], str]:
    """依次尝试各数据源，返回 (pdf_url, source_name)。"""
    sources = [
        ("openreview", lambda: _search_openreview(title, timeout)),
        ("semantic_scholar", lambda: _search_semantic_scholar(title, timeout)),
        ("arxiv", lambda: _search_arxiv(title, timeout)),
    ]

    for name, searcher in sources:
        url = searcher()
        if url:
            return url, name

    return None, ""


def download_conference(
    conference: str,
    delay: float = 2.0,
    timeout: int = 30,
) -> Dict[str, int]:
    """下载指定会议的所有论文 PDF。

    Args:
        conference: 会议名 (iclr2024, icml2024, nips2024)
        delay: 请求间隔（秒）
        timeout: 单个请求超时（秒）

    Returns:
        {"total": N, "downloaded": N, "skipped": N, "failed": N}
    """
    with open(DATASET_INFO_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if conference not in dataset:
        raise ValueError(f"Unknown conference: {conference}. "
                         f"Available: {list(dataset.keys())}")

    papers = dataset[conference]
    out_dir = CONFERENCE_DIRS[conference]
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / PROGRESS_FILENAME

    progress = _load_progress(progress_path)

    stats = {"total": len(papers), "downloaded": 0, "skipped": 0, "failed": 0}

    for i, entry in enumerate(papers, 1):
        repo_name = entry["repo_name"]
        paper_title = entry["paper"]
        output_path = out_dir / f"{repo_name}.pdf"

        print(f"\n[{i}/{stats['total']}] {repo_name}")
        print(f"  {paper_title}")

        # 增量跳过
        if (repo_name in progress
                and progress[repo_name].get("status") == "downloaded"
                and output_path.exists()
                and output_path.stat().st_size > 0):
            print(f"  [SKIP] already downloaded")
            stats["skipped"] += 1
            continue

        if repo_name in progress and progress[repo_name].get("status") == "downloaded":
            print(f"  [MISSING] progress says downloaded but file missing, re-downloading")

        pdf_url, source = _find_pdf_url(paper_title, timeout)

        if pdf_url:
            print(f"  [SOURCE] {source}")
            success = _download_pdf(pdf_url, output_path, timeout)
            if success:
                progress[repo_name] = {"status": "downloaded", "source": source}
                _save_progress(progress_path, progress)
                stats["downloaded"] += 1
            else:
                progress[repo_name] = {"status": "failed", "error": "download failed"}
                _save_progress(progress_path, progress)
                stats["failed"] += 1
        else:
            print(f"  [FAIL] not found on any source")
            progress[repo_name] = {"status": "failed", "error": "not found"}
            _save_progress(progress_path, progress)
            stats["failed"] += 1

        time.sleep(delay)

    print(f"\n{'=' * 60}")
    print(f"[{conference}] Done: {stats['downloaded']} downloaded, "
          f"{stats['skipped']} skipped, {stats['failed']} failed (total {stats['total']})")
    return stats


def download_all(delay: float = 2.0, timeout: int = 30) -> Dict[str, Dict[str, int]]:
    """下载所有三个会议的论文 PDF。"""
    results = {}
    for conf in ["iclr2024", "icml2024", "nips2024"]:
        print(f"\n{'=' * 60}")
        print(f"[CONFERENCE] {conf}")
        print(f"{'=' * 60}")
        results[conf] = download_conference(conf, delay=delay, timeout=timeout)
    return results


# ──────────────────────────────────────────────
#  CLI Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Download paper PDFs from OpenReview / Semantic Scholar / arXiv."
    )
    ap.add_argument(
        "--conference", "-c", type=str, default=None,
        choices=["iclr2024", "icml2024", "nips2024"],
        help="Download a specific conference only (default: all three)",
    )
    ap.add_argument(
        "--delay", "-d", type=float, default=2.0,
        help="Delay in seconds between requests (default: 2.0)",
    )
    ap.add_argument(
        "--timeout", "-t", type=int, default=30,
        help="HTTP request timeout in seconds (default: 30)",
    )

    args = ap.parse_args()

    if args.conference:
        download_conference(args.conference, delay=args.delay, timeout=args.timeout)
    else:
        download_all(delay=args.delay, timeout=args.timeout)
