#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_mini_data.py — 为消融实验准备 mini 训练集和测试集

运行位置：服务器
  cd /root/paper2CoreCode/programs/fine_tuning/ablation
  python prepare_mini_data.py

输出：
  /root/autodl-tmp/ablation_data/train_mini.json   (10条训练样本)
  /root/autodl-tmp/ablation_data/test_mini.json    (10条测试样本)
  /root/autodl-tmp/ablation_data/dataset_info.json (LLaMA-Factory 数据集注册)
"""

import json
import random
import shutil
from pathlib import Path

# ── 路径配置（服务器环境）──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FULL_TRAIN = _PROJECT_ROOT / "data" / "train_dataset.json"
FULL_TEST = _PROJECT_ROOT / "programs" / "fine_tuning" / "evaluation" / "test_dataset.json"
OUT_DIR = Path("/root/autodl-tmp/ablation_data")

# LLaMA-Factory 数据集目录
LLAMAFACTORY_DATA = Path("/root/LLaMA-Factory/data")

SEED = 42
N_TRAIN = 10
N_TEST = 10


def prepare():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    # ── 1. 训练集 ──────────────────────────────────────────────────────────
    print(f"[1/3] Reading full training set: {FULL_TRAIN}")
    if not FULL_TRAIN.exists():
        raise FileNotFoundError(f"Training set not found: {FULL_TRAIN}")

    with open(FULL_TRAIN, encoding="utf-8") as f:
        full_train = json.load(f)

    mini_train = random.sample(full_train, min(N_TRAIN, len(full_train)))
    train_out = OUT_DIR / "train_mini.json"
    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(mini_train, f, indent=2, ensure_ascii=False)
    print(f"  [OK] mini training set: {len(mini_train)} / {len(full_train)} samples -> {train_out}")

    # ── 2. 测试集 ──────────────────────────────────────────────────────────
    print(f"\n[2/3] Reading full test set: {FULL_TEST}")
    if not FULL_TEST.exists():
        raise FileNotFoundError(f"Test set not found: {FULL_TEST}")

    with open(FULL_TEST, encoding="utf-8") as f:
        full_test = json.load(f)

    # 从每个会议各取若干篇（ICLR在前25, ICML 25-50, NeurIPS 50-75）
    iclr = full_test[:25]
    icml = full_test[25:50]
    nips = full_test[50:75] if len(full_test) >= 75 else full_test[50:]

    mini_test = (
        random.sample(iclr, min(4, len(iclr)))
        + random.sample(icml, min(3, len(icml)))
        + random.sample(nips, min(3, len(nips)))
    )
    test_out = OUT_DIR / "test_mini.json"
    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(mini_test, f, indent=2, ensure_ascii=False)
    print(f"  [OK] mini test set: {len(mini_test)} / {len(full_test)} samples -> {test_out}")

    # ── 3. 注册到 LLaMA-Factory ──────────────────────────────────────────
    print(f"\n[3/3] Registering dataset to LLaMA-Factory: {LLAMAFACTORY_DATA}")

    lf_train_path = LLAMAFACTORY_DATA / "train_mini.json"
    shutil.copy(train_out, lf_train_path)
    print(f"  [OK] copied to: {lf_train_path}")

    dataset_info_path = LLAMAFACTORY_DATA / "dataset_info.json"
    if dataset_info_path.exists():
        with open(dataset_info_path, encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    dataset_info["mini_paper_code"] = {
        "file_name": "train_mini.json",
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }

    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f"  [OK] registered 'mini_paper_code' in {dataset_info_path}")

    # 同时保存到 OUT_DIR 用于记录
    with open(OUT_DIR / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump({"mini_paper_code": dataset_info["mini_paper_code"]}, f, indent=2, ensure_ascii=False)

    # ── 统计 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Mini dataset preparation complete")
    print("=" * 60)
    print(f"  Train: {train_out}  ({len(mini_train)} samples)")
    print(f"  Test:  {test_out}  ({len(mini_test)} samples)")
    print(f"  LF data: {lf_train_path}")
    print(f"\n  Avg input length: "
          f"{sum(len(d.get('input', '')) for d in mini_train) // len(mini_train)} chars")
    print(f"  Test with ref_code: "
          f"{sum(1 for d in mini_test if d.get('ref_code'))} / {len(mini_test)}")


if __name__ == "__main__":
    prepare()