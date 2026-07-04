#!/usr/bin/env python3
"""
run_stage0.py - Stage 0 一键脚本：PDF → JSON → 结构化 Markdown。

同时作为测试和执行脚本，支持三种模式：
  --test                              测试模式（data/test/examples/ → data/test/outputs/）
  -i paper.pdf                        单文件模式
  -i pdfs/ --batch                    批量模式

输出目录控制:
  --json-dir <path>   JSON 输出目录（默认: data/paper_jsons/）
  --txt-dir <path>    TXT 输出目录（默认: data/paper_txts/）

Docker GROBID 管理：
  --start-grobid      启动 GROBID 容器并等待就绪
  --stop-grobid       处理完成后停止 GROBID

用法:
  python run_stage0.py --test
  python run_stage0.py -i paper.pdf
  python run_stage0.py -i pdfs/ --batch --json-dir data/iclr2024_jsons --txt-dir data/training_set_txts
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pdf_reader import convert_pdf_to_json, batch_convert_pdfs
from data_cleaner import convert_paper_to_text, batch_convert_papers


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "data" / "test" / "examples"
TEST_OUTPUT_DIR = PROJECT_ROOT / "data" / "test" / "outputs"
PAPER_JSONS_DIR = PROJECT_ROOT / "data" / "paper_jsons"
PAPER_TXTS_DIR = PROJECT_ROOT / "data" / "paper_txts"

GROBID_CONTAINER = "grobid"
GROBID_IMAGE = "grobid/grobid:0.9.0-full"
GROBID_PORT = 8070
GROBID_HEALTH_URL = f"http://localhost:{GROBID_PORT}/api/isalive"
GROBID_STARTUP_TIMEOUT = 120


# ═══════════════════════════════════════════════
#  Docker GROBID 管理
# ═══════════════════════════════════════════════

def _run(cmd: List[str], capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=capture, text=True, check=check)


def _docker_grobid_status() -> str:
    """返回 grobid 容器状态: 'running' | 'stopped' | 'not_found'"""
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={GROBID_CONTAINER}",
         "--format", "{{.Status}}"],
        capture_output=True, text=True,
    )
    status_line = result.stdout.strip()
    if not status_line:
        return "not_found"
    if status_line.lower().startswith("up"):
        return "running"
    return "stopped"


def _grobid_is_ready() -> bool:
    """检查 GROBID API 是否就绪。"""
    import requests
    for _ in range(3):
        try:
            resp = requests.get(GROBID_HEALTH_URL, timeout=5)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_grobid() -> bool:
    """启动 GROBID 容器并等待就绪。返回是否成功。"""
    print(f"[GROBID] checking Docker ...")
    try:
        ver = subprocess.run(["docker", "--version"], capture_output=True, text=True).stdout.strip()
        print(f"[GROBID] {ver}")
    except Exception:
        print("[GROBID] ERROR: Docker not found or not running")
        return False

    status = _docker_grobid_status()
    print(f"[GROBID] container status: {status}")

    if status == "running":
        print(f"[GROBID] already running")
    elif status == "stopped":
        print(f"[GROBID] starting existing container ...")
        _run(["docker", "start", GROBID_CONTAINER])
    elif status == "not_found":
        print(f"[GROBID] pulling image & creating container ...")
        _run(["docker", "pull", GROBID_IMAGE])
        _run([
            "docker", "run", "-d",
            "-p", f"{GROBID_PORT}:8070",
            "--name", GROBID_CONTAINER,
            GROBID_IMAGE,
        ])

    print(f"[GROBID] waiting for service to be ready ...")
    deadline = time.time() + GROBID_STARTUP_TIMEOUT
    while time.time() < deadline:
        if _grobid_is_ready():
            elapsed = GROBID_STARTUP_TIMEOUT - (deadline - time.time()) + time.time()
            print(f"[GROBID] ready (took ~{elapsed:.0f}s)")
            return True
        print(".", end="", flush=True)
        time.sleep(3)
    print(f"\n[GROBID] ERROR: timed out after {GROBID_STARTUP_TIMEOUT}s")
    return False


def stop_grobid():
    """停止 GROBID 容器。"""
    status = _docker_grobid_status()
    if status == "running":
        print("[GROBID] stopping container ...")
        _run(["docker", "stop", GROBID_CONTAINER], check=False)
        print("[GROBID] stopped")
    else:
        print(f"[GROBID] container not running (status: {status}), nothing to stop")


# ═══════════════════════════════════════════════
#  处理逻辑
# ═══════════════════════════════════════════════

def process_single(pdf_path: Path, json_dir: Path, txt_dir: Path = None) -> Dict:
    """处理单个 PDF → JSON → TXT。

    Args:
        pdf_path: PDF 文件路径
        json_dir: JSON 输出目录
        txt_dir: TXT 输出目录（默认与 json_dir 相同）
    """
    if txt_dir is None:
        txt_dir = json_dir

    paper_id = pdf_path.stem
    json_path = json_dir / f"{paper_id}.json"
    txt_path = txt_dir / f"{paper_id}.txt"

    print(f"  [pdf_reader] {pdf_path.name} ...")
    convert_pdf_to_json(str(pdf_path), output_dir=str(json_dir))

    print(f"  [data_cleaner] → {paper_id}.txt ...")
    convert_paper_to_text(str(json_path), str(txt_path))

    json_size = json_path.stat().st_size if json_path.exists() else 0
    txt_size = txt_path.stat().st_size if txt_path.exists() else 0
    txt_content = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""

    return {
        "paper_id": paper_id,
        "json_path": str(json_path), "json_size": json_size,
        "txt_path": str(txt_path), "txt_size": txt_size,
        "txt_content": txt_content,
    }


def process_batch(input_dir: Path, json_dir: Path, txt_dir: Path) -> List[Dict]:
    """批量处理目录下的 PDF。"""
    from pdf_reader import PDFReader

    print(f"  [pdf_reader] batch {input_dir} → {json_dir} ...")
    PDFReader(parser_name="s2orc").batch_read(
        str(input_dir), str(json_dir), pattern="*.pdf"
    )

    print(f"  [data_cleaner] batch {json_dir} → {txt_dir} ...")
    batch_convert_papers(str(json_dir), str(txt_dir), pattern="*.json")

    results = []
    for json_path in sorted(json_dir.glob("*.json")):
        paper_id = json_path.stem
        txt_path = txt_dir / f"{paper_id}.txt"
        json_size = json_path.stat().st_size if json_path.exists() else 0
        txt_size = txt_path.stat().st_size if txt_path.exists() else 0
        txt_content = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
        results.append({
            "paper_id": paper_id,
            "json_path": str(json_path), "json_size": json_size,
            "txt_path": str(txt_path), "txt_size": txt_size,
            "txt_content": txt_content,
        })

    return results


# ═══════════════════════════════════════════════
#  验证（--test 模式）
# ═══════════════════════════════════════════════

def _validate_json(data: dict) -> List[str]:
    errors = []
    if not data.get("paper_id"):
        errors.append("missing 'paper_id'")
    if not data.get("title"):
        errors.append("missing 'title'")
    if not data.get("body"):
        errors.append("missing 'body'")
    return errors


def _count_sections(txt: str) -> int:
    return sum(1 for line in txt.splitlines() if line.lstrip().startswith("# "))


# ═══════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Stage 0: PDF → JSON → Markdown (test or production)"
    )
    # 模式
    ap.add_argument("--test", action="store_true",
                    help="Test mode: use data/test/examples/ → data/test/outputs/")
    ap.add_argument("--input", "-i", type=str, default=None,
                    help="Input PDF file (single) or directory (with --batch)")
    ap.add_argument("--batch", "-b", action="store_true",
                    help="Batch mode: process all PDFs in --input directory")

    # 输出目录
    ap.add_argument("--json-dir", type=str, default=None,
                    help="JSON output directory (default: data/paper_jsons/)")
    ap.add_argument("--txt-dir", type=str, default=None,
                    help="TXT output directory (default: data/paper_txts/)")

    # GROBID
    ap.add_argument("--start-grobid", action="store_true",
                    help="Start Docker GROBID before processing")
    ap.add_argument("--stop-grobid", action="store_true",
                    help="Stop Docker GROBID after processing")
    ap.add_argument("--skip-grobid-check", action="store_true",
                    help="Skip GROBID readiness check")

    args = ap.parse_args()

    # ── 模式确定 ──
    if args.test:
        input_source = EXAMPLES_DIR
        output_dir = TEST_OUTPUT_DIR
        is_test = True
        single_file = None
        print("=" * 60)
        print("Stage 0 — Test Mode")
        print("=" * 60)
    elif args.input:
        input_path = Path(args.input)
        if args.batch or input_path.is_dir():
            input_source = input_path
            single_file = None
            is_test = False
        else:
            input_source = None
            single_file = input_path
            is_test = False
    else:
        ap.print_help()
        sys.exit(1)

    # ── GROBID 启动 ──
    if args.start_grobid:
        print(f"\n{'─' * 40}")
        if not start_grobid():
            sys.exit(1)

    # GROBID 可用性检查
    if not args.skip_grobid_check:
        if not _grobid_is_ready():
            print("\n[ERROR] GROBID is not running at localhost:8070")
            print("  Start it with:  docker run -d -p 8070:8070 --name grobid grobid/grobid:0.9.0-full")
            print("  Or use:         --start-grobid")
            sys.exit(1)

    # ── 处理 ──
    if is_test:
        pdf_files = sorted(EXAMPLES_DIR.glob("*.pdf"))
        if not pdf_files:
            print("[ERROR] No PDFs in data/test/examples/")
            sys.exit(1)

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nPDFs: {len(pdf_files)}  →  {output_dir}")

        results = []
        for pdf_path in pdf_files:
            print(f"\n--- {pdf_path.name} ---")
            try:
                r = process_single(pdf_path, output_dir)
                # 验证
                with open(r["json_path"], encoding="utf-8") as f:
                    data = json.load(f)
                errs = _validate_json(data)
                if errs:
                    r["status"] = "fail"
                    r["error"] = ", ".join(errs)
                    print(f"  [WARN] {r['error']}")
                elif r["txt_size"] == 0:
                    r["status"] = "fail"
                    r["error"] = "empty output"
                    print(f"  [WARN] {r['error']}")
                else:
                    r["status"] = "pass"
                    r["error"] = ""
                    secs = _count_sections(r["txt_content"])
                    print(f"  [PASS] JSON={r['json_size']:,}B  TXT={r['txt_size']:,} chars  {secs} sections")
                results.append(r)
            except Exception as exc:
                print(f"  [FAIL] {exc}")
                results.append({
                    "paper_id": pdf_path.stem, "status": "fail", "error": str(exc),
                    "json_path": "", "json_size": 0,
                    "txt_path": "", "txt_size": 0, "txt_content": "",
                })

        passed = [r for r in results if r["status"] == "pass"]
        failed = [r for r in results if r["status"] == "fail"]
        print(f"\n{'=' * 60}")
        print(f"Test Results: {len(passed)}/{len(results)} passed")
        for r in failed:
            print(f"  [FAIL] {r['paper_id']}: {r['error']}")
        print(f"{'=' * 60}")

        if failed:
            sys.exit(1)

    elif single_file:
        if not single_file.suffix.lower() == ".pdf":
            print(f"[ERROR] Not a PDF: {single_file}")
            sys.exit(1)
        json_dir = Path(args.json_dir) if args.json_dir else PAPER_JSONS_DIR
        txt_dir = Path(args.txt_dir) if args.txt_dir else PAPER_TXTS_DIR
        json_dir.mkdir(parents=True, exist_ok=True)
        txt_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nSingle: {single_file}  →  JSON: {json_dir}  TXT: {txt_dir}")
        print(f"--- {single_file.name} ---")
        try:
            process_single(single_file, json_dir, txt_dir)
            print(f"  Done")
        except Exception as exc:
            print(f"[FAIL] {exc}")
            sys.exit(1)

    else:
        input_dir = Path(args.input)
        json_dir = Path(args.json_dir) if args.json_dir else PAPER_JSONS_DIR
        txt_dir = Path(args.txt_dir) if args.txt_dir else PAPER_TXTS_DIR
        json_dir.mkdir(parents=True, exist_ok=True)
        txt_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nBatch: {input_dir}  →  JSON: {json_dir}  TXT: {txt_dir}")
        try:
            results = process_batch(input_dir, json_dir, txt_dir)
            count = len(results)
            total_chars = sum(r["txt_size"] for r in results)
            print(f"\nDone: {count} PDFs, {total_chars:,} chars total")
        except Exception as exc:
            print(f"[FAIL] {exc}")
            sys.exit(1)

    # ── GROBID 停止 ──
    if args.stop_grobid:
        print()
        stop_grobid()


if __name__ == "__main__":
    main()
