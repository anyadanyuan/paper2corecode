# paper2CoreCode — Fine-Tuned Qwen2.5-Coder-7B for Paper-to-Code Generation

将学术论文文本自动转换为 PyTorch 核心模型代码的微调模型。

---

## 期末交付说明

- 报告：final_report.pdf
- 展示slides：paperagent.pdf

---

## 项目概述

基于 Qwen2.5-Coder-7B-Instruct，使用 90 篇 2024 年顶级 ML 会议论文（ICLR/ICML/NeurIPS）构建 Alpaca 格式 SFT 数据集，通过 LLaMA-Factory QLoRA 微调。在独立的 75 篇 2025 年论文测试集上进行评测，并完成了 5 种基座模型 × 4 种量化方案的消融实验。

```
输入: 论文核心文本 (Abstract + Body)
  ↓  Qwen2.5-Coder-7B-SFT (LoRA, from HuggingFace: Anyadanyuan/Qwen2.5-7B-paper2XCode)
输出: <file name="core_model.py">
      import torch
      import torch.nn as nn
      class ProposedModel(nn.Module): ...
      </file>
```

---

## 目录结构

```
paper2CoreCode/
├── requirements.txt                 # 统一依赖（pip install -r requirements.txt）
├── README.md                        # 本文件
│
├── data/                            # 数据（训练、测试、消融、结果）
│   ├── train_dataset.json           # SFT 训练集（90 条 Alpaca 格式）
│   ├── test_dataset.json            # 评测测试集（75 条，2025 年论文）
│   ├── dataset_2024.json            # 训练集论文元数据
│   ├── dataset_2025.json            # 测试集论文元数据
│   ├── training_set_txts/           # 清洗后的论文文本（86 篇）
│   ├── evaluation_set_txts/         # 评测用论文文本
│   ├── evaluation_results/          # 评测结果（baseline + finetuned + comparison.md）
│   ├── ablation/                    # 消融实验数据与结果
│   │   ├── train_mini.json          # Mini 训练集（10 条）
│   │   ├── test_mini.json           # Mini 测试集（10 条）
│   │   ├── ablation_eval/           # 8 组消融评估结果
│   │   ├── full_results.json        # 推演完整结果（已标注数据来源）
│   │   └── ablation_table.md        # 消融实验表格（已标注推算值）
│   └── *_{pdfs,jsons}/              # 论文 PDF 和结构化 JSON（按会议年份组织）
│
├── programs/
│   ├── fine_tuning/
│   │   ├── training/                # 训练
│   │   │   ├── run_training.py      # 训练编排器（环境检查 + LLaMA-Factory + TaskMonitor）
│   │   │   ├── train_qwen.yaml      # 训练配置模板
│   │   │   ├── merge_qwen.yaml      # LoRA 合并配置
│   │   │   ├── training_set_generation/  # 训练集生成
│   │   │   │   ├── run_training_set_generation.py  # 生成编排器（增量 + 测试模式）
│   │   │   │   ├── training_set_assemblage.py      # Alpaca 格式组装
│   │   │   │   └── distill_code_repository.py      # ACPP 代码提纯（AST 筛选 + LLM 蒸馏 + 沙盒验证）
│   │   │   └── ablation/            # 消融实验
│   │   │       ├── run_all_training.sh      # 批量训练（TaskMonitor 监控）
│   │   │       ├── run_all_evaluation.sh    # 批量评估（TaskMonitor 监控）
│   │   │       ├── prepare_mini_data.py     # Mini 数据集准备
│   │   │       ├── generate_configs.py      # 配置生成
│   │   │       ├── extrapolate_results.py   # 结果推演
│   │   │       ├── configs/                 # 8 组训练 YAML 配置
│   │   │       ├── training_log.json        # 训练显存/耗时记录
│   │   │       └── eval_log.json            # 评估指标记录
│   │   │
│   │   ├── evaluation/              # 评测系统
│   │   │   ├── evaluate.py          # 主评测脚本（含 TaskMonitor）
│   │   │   ├── compare.py           # 基座 vs 微调对比报告
│   │   │   ├── eval_from_hf.sh      # 从 HuggingFace 一键评测
│   │   │   ├── build_test_set_correct.py    # 测试集构建工具
│   │   │   ├── EVALUATION_REPORT.md         # 完整评测分析报告
│   │   │   ├── evaluators/          # 指标实现
│   │   │   │   ├── execution_eval.py        # XPR / SAR / SYR
│   │   │   │   ├── fidelity_eval.py         # PCR / AAR / CodeBERTScore
│   │   │   │   └── paper_extractor.py       # LLM 论文组件提取
│   │   │   └── utils/               # 工具
│   │   │       ├── model_loader.py          # 模型加载（支持本地/HF/API）
│   │   │       ├── code_parser.py           # 代码提取
│   │   │       └── sandbox.py               # 沙盒执行
│   │   │
│   │   └── inference/               # 推理
│   │       └── qwen_infer.py        # 批量推理（本地 transformers / vLLM API）
│   │
│   ├── stage0_text_cleaner/         # 论文 PDF 文本提取
│   │   ├── pdf_reader.py            # s2orc-doc2json + GROBID 解析
│   │   ├── data_cleaner.py          # JSON 渲染为结构化纯文本
│   │   ├── run_stage0.py            # 统一执行器
│   │   └── download_pdfs.py         # 多源 PDF 下载
│   │
│   └── scripts/
│       ├── download_pdfs.py         # PDF 批量下载
│       └── generate_training_and_evaluation_set_txts.py
│
├── utils/
│   └── nvidia_utils.py              # GPU 显存监控工具（TaskMonitor / @monitor）
│
└── s2orc-doc2json/                  # PDF 解析依赖库（精简版）
    └── doc2json/grobid2json/        # GROBID client + TEI → JSON
```

---

## 模型信息

| 项目 | 内容 |
|---|---|
| **基础模型** | Qwen2.5-Coder-7B-Instruct |
| **微调模型** | [Anyadanyuan/Qwen2.5-7B-paper2XCode](https://huggingface.co/Anyadanyuan/Qwen2.5-7B-paper2XCode) |
| **微调方式** | QLoRA (4-bit quantization, LoRA target=all) |
| **训练框架** | LLaMA-Factory |
| **训练数据** | 90 条 Alpaca 格式（ICLR/ICML/NeurIPS 2024） |
| **训练配置** | 8 epochs, lr=2e-4, batch_size=2, grad_accum=8, cutoff_len=2048 |
| **训练 Loss** | 0.5489 |
| **训练耗时** | ~10 分钟 |
| **LoRA 适配器** | [Anyadanyuan/Qwen2.5-7B-paper2XCode-lora](https://huggingface.co/Anyadanyuan/Qwen2.5-7B-paper2XCode-lora) |

---

## 一键安装与评测

```bash
git clone https://github.com/<username>/paper2CoreCode
cd paper2CoreCode

# 安装依赖
pip install -r requirements.txt

# 评测微调模型（自动从 HuggingFace 下载模型 + TaskMonitor 显存监控）
cd programs/fine_tuning/evaluation
bash eval_from_hf.sh
```

输出：
- `outputs/eval/hf_finetuned/summary.json` — 评测汇总
- `outputs/eval/hf_finetuned/results.jsonl` — 逐样本结果
- `outputs/eval/hf_finetuned/monitor/` — TaskMonitor 显存记录

如需 PCR 指标，设置环境变量：`export OPENROUTER_API_KEY="sk-..."`

---

## 评测系统

基于"可运行性 × 论文忠实度"双维度，共 6 项指标：

| 维度 | 指标 | 缩写 | 说明 |
|---|---|---|---|
| 可运行性 | 可执行率 | XPR | 沙盒执行成功比例 |
| | 语法正确率 | SYR | AST 解析通过比例 |
| 论文忠实度 | 论文组件覆盖率 | PCR | LLM 提取论文组件 → 规则匹配代码 |
| | API 对齐率 | AAR | PyTorch API 与参考代码 Jaccard 相似度 |
| | 代码语义相似度 | CodeBERTScore | CodeBERT embedding F1 |

综合评分：`0.5 × execution_score + 0.5 × fidelity_score`

### 评测结果

| 指标 | 基座模型 | 微调模型 | 提升 |
|---|---|---|---|
| XPR | 9.4% | 14.7% | +5.3pp |
| SYR | 39.1% | 34.7% | -4.4pp |
| PCR | 9.8% | 16.2% | +6.4pp |
| CodeBERTScore | 0.380 | 0.444 | +0.064 |
| **Overall** | **0.161** | **0.205** | **+0.044** |

> 完整报告：`programs/fine_tuning/evaluation/EVALUATION_REPORT.md`

---

## 消融实验

5 种基座模型 × 4 种量化方案的消融实验，验证 Qwen2.5-Coder-7B + 4-bit QLoRA 为最优配置。

| 类别 | 对比项 |
|---|---|
| 基座模型 | Qwen2.5-Coder-7B, Qwen2.5-7B, DeepSeek-Coder-V2-Lite, CodeLlama-7B, Mistral-7B |
| 量化方案 | BF16 全参数, BF16+LoRA, 8-bit QLoRA, 4-bit QLoRA |

> 数据与结果：`data/ablation/`，脚本与配置：`programs/fine_tuning/training/ablation/`

---

## 其他功能

### 训练

```bash
python programs/fine_tuning/training/run_training.py --check-only   # 环境检查
python programs/fine_tuning/training/run_training.py                # 完整训练
```

### 推理

```bash
# 单篇论文推理
python programs/fine_tuning/inference/qwen_infer.py --single data/training_set_txts/ACT.txt

# 批量推理
python programs/fine_tuning/inference/qwen_infer.py --input_dir data/training_set_txts --output_dir outputs/qwen
```

### 评测

```bash
cd programs/fine_tuning/evaluation

# 本地模型评测
python evaluate.py --model_path /path/to/model --dataset ../../../data/test_dataset.json --output_dir outputs/eval/

# HuggingFace 模型评测
bash eval_from_hf.sh

# 对比报告
python compare.py --finetuned outputs/eval/finetuned/summary.json --baseline outputs/eval/baseline/summary.json
```

### PDF 文本提取

```bash
# 需要 GROBID Docker 服务
docker run -d -p 8070:8070 grobid/grobid:0.9.0-full

python programs/stage0_text_cleaner/run_stage0.py --start-grobid
```

---

## 服务器部署

| 路径 | 用途 |
|---|---|
| `/root/LLaMA-Factory/` | 训练框架 |
| `/root/autodl-tmp/huggingface_cache/` | HuggingFace 模型缓存 |
| `/root/autodl-tmp/lora_adapter_output/` | LoRA 适配器输出 |
| `/root/autodl-tmp/models/Qwen2.5-7B-paper2Xcode/` | 合并后的完整模型 |

```bash
# 环境变量
export HF_HOME="/root/autodl-tmp/huggingface_cache"
```

---

## 依赖

```
torch>=2.0.0          # 核心
transformers>=4.40.0  # 核心
accelerate>=0.29.0    # 核心
peft>=0.10.0          # LoRA 微调
bitsandbytes>=0.43.0  # 量化
code-bert-score>=0.3.0 # 评测
openai>=1.30.0        # LLM API (PCR)
python-dotenv         # 环境变量
beautifulsoup4>=4.7.1 # PDF 解析
lxml                  # PDF 解析
requests              # HTTP
python-magic>=0.4.18  # 文件类型检测
numpy>=1.24.0         # 数值计算
tiktoken>=0.5.0       # Token 计数
```

---

## 引用与致谢

- 基础模型：[Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- 训练框架：[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- 评测指标：[CodeBERTScore](https://github.com/neulab/code-bert-score)
- PDF 解析：[s2orc-doc2json](https://github.com/allenai/s2orc-doc2json)
- 代码蒸馏灵感来自 [OriGen](https://arxiv.org/abs/2305.16264) 的 Code-to-Code Augmentation

---

## 许可证

本项目仅供学术研究使用。