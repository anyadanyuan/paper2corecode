#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_configs.py — 批量生成消融实验的 9 组 LLaMA-Factory 训练配置

运行位置：服务器 /root/paper2XAgent/training/ablation/
运行命令：python generate_configs.py

输出：configs/ 目录下 9 个 .yaml 文件
"""

import os
from pathlib import Path

OUT_DIR = Path(__file__).parent / "configs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_OUTPUT = "/root/autodl-tmp/ablation_outputs"

# ── 实验组定义 ────────────────────────────────────────────────────────────
# 组1-5：基座模型对比（固定 4-bit QLoRA）
# 组6-8：量化方案对比（固定 Qwen2.5-Coder-7B-Instruct）
# 注：Qwen-Coder 4-bit 组同时属于两组，只训练一次（configs/group1_qwen_coder_4bit.yaml）

EXPERIMENTS = [
    # ── 组1-5：基座模型对比 ──────────────────────────────────────────────
    {
        "name": "group1_qwen_coder_4bit",
        "group": "backbone",
        "label": "Qwen2.5-Coder-7B-Instruct (4-bit QLoRA)",
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "ft_type": "lora",
        "quant_bit": 4,
    },
    {
        "name": "group2_qwen_7b_4bit",
        "group": "backbone",
        "label": "Qwen2.5-7B-Instruct (4-bit QLoRA)",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "ft_type": "lora",
        "quant_bit": 4,
    },
    {
        "name": "group3_deepseek_coder_4bit",
        "group": "backbone",
        "label": "DeepSeek-Coder-V2-Lite-Instruct (4-bit QLoRA)",
        "model": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        "ft_type": "lora",
        "quant_bit": 4,
    },
    {
        "name": "group4_codellama_4bit",
        "group": "backbone",
        "label": "CodeLlama-7b-Instruct (4-bit QLoRA)",
        "model": "codellama/CodeLlama-7b-Instruct-hf",
        "ft_type": "lora",
        "quant_bit": 4,
    },
    {
        "name": "group5_mistral_4bit",
        "group": "backbone",
        "label": "Mistral-7B-Instruct-v0.3 (4-bit QLoRA)",
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "ft_type": "lora",
        "quant_bit": 4,
    },
    # ── 组6-8：量化方案对比（其余均基于 Qwen2.5-Coder-7B-Instruct）────────
    # 注：4-bit 方案已在 group1 中，此处不重复
    {
        "name": "group6_qwen_coder_bf16_full",
        "group": "quantization",
        "label": "BF16 全参数微调",
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "ft_type": "full",
        "quant_bit": None,  # 无量化
    },
    {
        "name": "group7_qwen_coder_bf16_lora",
        "group": "quantization",
        "label": "BF16 + LoRA（不量化）",
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "ft_type": "lora",
        "quant_bit": None,  # 无量化
    },
    {
        "name": "group8_qwen_coder_8bit",
        "group": "quantization",
        "label": "8-bit QLoRA",
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "ft_type": "lora",
        "quant_bit": 8,
    },
    # group1 复用为量化组的 4-bit 基准（不重新训练，但 README 中会说明）
]


def make_yaml(exp: dict) -> str:
    """生成单组实验的 LLaMA-Factory 训练配置 yaml"""

    quant_line = f"quantization_bit: {exp['quant_bit']}\n" if exp["quant_bit"] else ""

    # BF16 全参数微调不需要 lora_target
    lora_lines = ""
    if exp["ft_type"] == "lora":
        lora_lines = "lora_target: all\nlora_rank: 8\nlora_alpha: 16\n"

    # 全参数微调用 bf16，量化训练用 fp16
    precision = "bf16: true" if exp["quant_bit"] is None else "fp16: true"

    return f"""# ============================================================
# 消融实验配置
# 组名:  {exp["name"]}
# 描述:  {exp["label"]}
# 实验组: {exp["group"]}
# ============================================================

model_name_or_path: {exp["model"]}
stage: sft
do_train: true
finetuning_type: {exp["ft_type"]}
{lora_lines}
dataset: mini_paper_code
template: default
output_dir: {BASE_OUTPUT}/{exp["name"]}
overwrite_output_dir: true

# ── 快速训练参数（消融实验用，样本10条，2 epochs）──────────────
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 2
logging_steps: 1
save_steps: 500
{precision}
{quant_line}gradient_checkpointing: true
cutoff_len: 2048
"""


def main():
    print(f"生成 {len(EXPERIMENTS)} 组训练配置...\n")
    for exp in EXPERIMENTS:
        yaml_path = OUT_DIR / f"{exp['name']}.yaml"
        content = make_yaml(exp)
        yaml_path.write_text(content, encoding="utf-8")
        print(f"  [OK] {yaml_path.name}  ({exp['label']})")

    print(f"\n[OK] All configs written to {OUT_DIR}/")
    print(f"\nTotal {len(EXPERIMENTS)} groups, plus group1 reused as quantization baseline.")
    print("\n下一步：在服务器上运行 bash run_all_training.sh")


if __name__ == "__main__":
    main()
