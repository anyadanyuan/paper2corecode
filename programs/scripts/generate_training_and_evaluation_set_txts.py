#!/usr/bin/env python3
"""
generate_training_and_evaluation_set_txts.py - 批量运行 Stage 0，处理 2024/2025 年全部会议论文。

JSON 按会议输出: data/{conf}_jsons/
TXT 按用途输出: 2024 → data/training_set_txts/  2025 → data/evaluation_set_txts/

用法:
  python generate_training_and_evaluation_set_txts.py
  python generate_training_and_evaluation_set_txts.py --start-grobid --stop-grobid
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CONFERENCES_2024 = ["iclr2024", "icml2024", "nips2024"]
CONFERENCES_2025 = ["iclr2025", "icml2025", "nips2025"]

RUN_SCRIPT = Path(__file__).resolve().parent.parent / "stage0_text_cleaner" / "run_stage0.py"


def run_batch(conf: str, txt_dir_name: str, extra_args: list = None):
    pdf_dir = DATA_DIR / f"{conf}_pdfs"
    json_dir = DATA_DIR / f"{conf}_jsons"
    txt_dir = DATA_DIR / txt_dir_name

    if not pdf_dir.is_dir():
        print(f"[SKIP] {pdf_dir} not found")
        return

    pdf_count = len(list(pdf_dir.glob("*.pdf")))
    if pdf_count == 0:
        print(f"[SKIP] no PDFs in {pdf_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"  {conf}  ({pdf_count} PDFs)")
    print(f"  JSON → {json_dir}")
    print(f"  TXT  → {txt_dir}")
    print(f"{'=' * 60}")

    cmd = [
        sys.executable, str(RUN_SCRIPT),
        "-i", str(pdf_dir), "--batch",
        "--json-dir", str(json_dir),
        "--txt-dir", str(txt_dir),
        "--skip-grobid-check",
    ]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[FAIL] {conf} exited with {result.returncode}")


def main():
    ap = argparse.ArgumentParser(
        description="Batch run Stage 0 for all 2024/2025 conference PDFs."
    )
    ap.add_argument("--start-grobid", action="store_true",
                    help="Start Docker GROBID before first batch")
    ap.add_argument("--stop-grobid", action="store_true",
                    help="Stop Docker GROBID after last batch")
    args = ap.parse_args()

    print("=" * 60)
    print("Stage 0 — Batch Run (2024 + 2025)")
    print("=" * 60)

    all_confs = [(c, "training_set_txts") for c in CONFERENCES_2024] \
               + [(c, "evaluation_set_txts") for c in CONFERENCES_2025]

    for i, (conf, txt_dir_name) in enumerate(all_confs):
        extras = []
        if i == 0 and args.start_grobid:
            extras.append("--start-grobid")
        if i == len(all_confs) - 1 and args.stop_grobid:
            extras.append("--stop-grobid")
        run_batch(conf, txt_dir_name, extras)

    print(f"\n{'=' * 60}")
    print("All done")


if __name__ == "__main__":
    main()
