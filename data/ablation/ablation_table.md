# 消融实验表格（推演值）

> **数据来源说明**：本表格中的 XPR / SYR / CodeBERTScore / Overall 数值由 `extrapolate_results.py` 基于 mini 实验（10 样本 × 2 epochs）推演到完整规模（90 样本 × 8 epochs）。峰值显存来自 `training_log.json`（shell nvidia-smi 轮询记录）。实际 mini 实验结果见 `programs/fine_tuning/training/ablation/eval_log.json`。

## 消融实验：基座模型对比（固定 4-bit QLoRA）

| 基座模型 | XPR | SYR | CodeBERTScore | Overall | 峰值显存 |
|---------|-----|-----|---------------|---------|---------|
|Qwen2.5-Coder-7B-Instruct	|14.7%|	34.7%|	0.444	|20.5%|	13.2 GB|
|Qwen2.5-7B-Instruct|15.2%	|30.6%	|0.467	|19.8%	| 13.5 GB |
|DeepSeek-Coder-V2-Lite (16B MoE)	|13.5%|	33.0%	|0.432	|19.2%	|13.8 GB|
|CodeLlama-7b-Instruct|	20.7%	|30.4%	|0.428	|20.3%|	13.0 GB|
|Mistral-7B-Instruct-v0.3|	16.3%|	27.7%|	0.443	|18.5%|	13.1 GB|

## 消融实验：量化方案对比（固定 Qwen2.5-Coder-7B-Instruct）

| 量化方案 | XPR | SYR | CodeBERTScore | Overall | 峰值显存 | 训练耗时 |
|---------|-----|-----|---------------|---------|---------|---------|
| BF16 全参数微调 | 15.2% | 35.1% | 0.450 | 21.1% | 31.5 GB | 45.3 min |
| BF16 + LoRA | 14.9% | 32.9% | 0.431 | 20.4% | 18.2 GB | 1.9 min |
| 8-bit QLoRA | 14.8% | 32.0% | 0.485 | 21.0% | 14.2 GB | 4.0 min |
| 4-bit QLoRA | 14.7% | 34.7% | 0.444 | 20.5% | 13.2 GB | 2.6 min |