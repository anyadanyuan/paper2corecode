# 测试集构建指南 — LLM蒸馏版本（与训练集一致）

> **重要变更（2026-05-24）**：测试集构建方法已从"简单文件提取"升级为"与训练集完全一致的LLM蒸馏"。

---

## 📢 为什么要改？

### 旧方案的问题

| 维度 | 训练集（旧） | 测试集（旧）| 问题 |
|------|------------|------------|------|
| 代码来源 | GitHub 全仓库所有 .py | GitHub **单个文件** | ❌ 来源不一致 |
| 代码处理 | LLM蒸馏（干净核心代码） | **直接读取原始代码** | ❌ 抽象层次不同 |
| 输出格式 | `<file>` 标签格式 | 原始 Python 代码 | ❌ 格式不匹配 |
| 评测公平性 | — | — | ❌ **不公平** |

**根本矛盾**：模型在训练时学习的是 LLM 蒸馏后的核心代码，但评测时对比的是含大量 boilerplate 的原始代码。导致 AAR / CodeBERTScore 被系统性低估。

### 新方案

测试集的 `ref_code` 现在使用**完全相同的蒸馏流程**，确保评测公平：

| 维度 | 训练集 | 测试集（新）|
|------|--------|------------|
| 代码提取 | `GitExtractor.extract_python_code()` | ✅ **相同方法** |
| 蒸馏引擎 | `DistillationEngine (gpt-4o-mini)` | ✅ **相同引擎** |
| System Prompt | `distill_code_repository_system_prompt.md` | ✅ **相同文件** |
| 输出格式 | `<file name="core_model.py">...</file>` | ✅ **相同格式** |

---

## 🔄 完整蒸馏流程

```
GitHub 仓库 URL
  │
  ▼ GitExtractor.extract_python_code()
  克隆仓库（--depth 1），遍历所有 .py 文件并合并
  （每个文件加 # FILE: xxx 分隔符，便于 LLM 识别）
  │
  ▼ tiktoken 计数
  token_count > 120,000？→ 跳过（ref_code = ""）
  │
  ▼ DistillationEngine.distill_code()
  调用 gpt-4o-mini（responses.create API）
  System Prompt：提取核心 nn.Module，剔除 boilerplate
  │
  ▼ 正则提取 <file> 标签
  返回蒸馏后代码（<file name="core_model.py">...</file>）
  存入 test_dataset.json 的 ref_code 字段
```

---

## 🚀 使用方法

### 前置准备

```powershell
cd D:\Projects\paper2XAgent\evaluation

# 1. 安装依赖（新增 tiktoken）
pip install -r requirements.txt

# 2. 设置 API 密钥
# 支持 OpenRouter（sk-or-v1-...）或 OpenAI 官方（sk-...）
$env:OPENAI_API_KEY = "sk-or-v1-你的密钥"

# 3. 验证环境
python -c "
import tiktoken
from openai import OpenAI
import sys
sys.path.insert(0, '../Paper2Code/data/paper2code')
from distill_code_repository import GitExtractor, DistillationEngine, CostEstimator
print('✓ 所有依赖就绪')
"
```

### 运行构建

```powershell
python build_test_set_correct.py --config test_dataset_info.json --output test_dataset.json
```

**支持随时 Ctrl+C 中断**，下次运行自动从断点继续（断点续传）。

### 典型输出

```
✓ 已加载 LLM 蒸馏引擎（与训练集一致）
✓ 蒸馏引擎初始化成功 (gpt-4o-mini)

======================================================================
开始构建测试集...
======================================================================

处理 ICLR2025 (25 篇论文)
----------------------------------------------------------------------

[1/25] EmbodiedSAM: Online Segment Any 3D Thing in Real Time
  提取文本: EmbodiedSAM：Online Segment Any 3D Thing in Real Time.pdf
  提取代码: https://github.com/xuxw98/ESAM
    [1/3] 克隆仓库并提取Python文件...
    [2/3] Token数: 45,231, 预估成本: $0.0234
    [3/3] LLM蒸馏中 (gpt-4o-mini)...
    ✓ 蒸馏完成 (输出: 3,421 字符)
  ✓ 成功添加 (文本: 4523 字符, 代码: 3421 字符)

[2/25] TopoLM: brain-like spatio-functional organization...
  提取代码: https://github.com/epflneuroailab/topolm
    ⚠️ 跳过已知有问题的仓库: epflneuroailab/topolm
  ✓ 成功添加 (文本: 4226 字符, 代码: 0 字符)

[3/25] UQDM
  提取代码: https://github.com/mandt-lab/uqdm
    [1/3] 克隆仓库并提取Python文件...
    [2/3] Token数: 128,432, 预估成本: $0.0664
    ⚠️ Token超限 (128,432 > 120,000)，跳过蒸馏
  ✓ 成功添加 (文本: 3890 字符, 代码: 0 字符)

...

======================================================================
构建完成! 共 75 个样本
======================================================================

统计:
  总论文数: 75
  找到 PDF: 72 (96.0%)
  提取文本: 71 (94.7%)
  提取代码: 52 (69.3%)

💰 LLM蒸馏统计:
  成功蒸馏: 52
  失败/跳过: 23 (ref_code留空)
  累计Token: 2,341,234
  累计成本: $1.87
  平均每样本: $0.0360

文本文件: test_cleaned_output
测试集文件: test_dataset.json
```

---

## ⚠️ 失败处理策略

所有失败情况均保留样本，`ref_code = ""`（评测时 AAR / CodeBERTScore 自动为 0.0）：

| 失败原因 | 处理方式 |
|---------|---------|
| 黑名单仓库（topolm、AIRS 等） | 跳过蒸馏，`ref_code = ""` |
| Token 超限（> 120,000） | 跳过蒸馏，`ref_code = ""` |
| Git 克隆失败 / 无 Python 文件 | 跳过蒸馏，`ref_code = ""` |
| API 调用失败（超时、限流等） | 捕获异常，`ref_code = ""` |
| LLM 输出无 `<file>` 标签 | 保留原始文本（`DistillationEngine` 内部处理） |

---

## 💰 成本估算

| 项目 | 数值 |
|------|------|
| 模型 | gpt-4o-mini |
| 输入价格 | $0.15 / 1M tokens |
| 输出价格 | $0.60 / 1M tokens |
| 平均输入 | ~50,000 tokens / 仓库 |
| 平均输出 | ~1,500 tokens / 仓库 |
| 单仓库成本 | ~$0.015 - $0.04 |
| **75篇总成本** | **~$1.50 - $3.00** |
| 预计耗时 | 2 - 3 小时 |

---

## 📊 验证结果

```powershell
python -c "
import json
data = json.load(open('test_dataset.json', encoding='utf-8'))

total = len(data)
with_code = sum(1 for d in data if d['ref_code'])
with_file_tag = sum(1 for d in data if '<file' in d.get('ref_code', ''))

print(f'总样本数:        {total}')
print(f'包含ref_code:    {with_code} / {total}')
print(f'含<file>标签:    {with_file_tag} / {total}')

# 格式验证
required = ['paper_id', 'paper_title', 'paper_text', 'ref_code', 'github_url']
missing = [k for k in required if any(k not in d for d in data)]
print(f'格式验证: {\"通过\" if not missing else \"缺少字段: \" + str(missing)}')

# 展示第一个成功蒸馏的样本
for d in data:
    if d['ref_code'] and '<file' in d['ref_code']:
        print(f'\n示例样本: {d[\"paper_title\"]}')
        print(f'  ref_code长度: {len(d[\"ref_code\"])} 字符')
        print(f'  ref_code开头: {d[\"ref_code\"][:120]}...')
        break
"
```

---

## 🎯 预期效果

改进后，由于 `ref_code` 与模型输出处于同一抽象层次（均为蒸馏后的核心代码），
以下指标预期显著提升：

| 指标 | 改进前 | 改进后（预期）| 提升原因 |
|------|--------|--------------|---------|
| AAR | 5.08% | 8 - 12% | ref_code 也是核心 API，集合匹配度提升 |
| CodeBERTScore | 0.444 | 0.50 - 0.55 | 格式一致（都是 `<file>` 标签格式）|
| PCR | 16.19% | 不变 | 不依赖 ref_code |
| XPR | 14.67% | 不变 | 执行指标不受影响 |

---

## 📁 相关文件

| 文件 | 说明 |
|------|------|
| `evaluation/build_test_set_correct.py` | 测试集构建脚本（本文件对应）|
| `evaluation/test_dataset_info.json` | 输入配置（论文标题 + GitHub URL）|
| `evaluation/test_dataset.json` | 输出测试集（运行后生成）|
| `Paper2Code/data/paper2code/distill_code_repository.py` | 蒸馏引擎（直接复用，不修改）|
| `Paper2Code/data/paper2code/distill_code_repository_system_prompt.md` | 蒸馏 System Prompt（直接复用）|

---

## ❓ 常见问题

**Q: 为什么不直接提取 GitHub 单个文件？**

A: 训练集使用全仓库蒸馏，测试集必须采用相同方法才能确保评测公平。如果训练集是蒸馏代码而测试集是原始代码，AAR/CodeBERTScore 会被系统性低估。

**Q: `responses.create` 是什么 API？**

A: 训练集构建时已验证可用的 OpenAI SDK 调用方式，直接复用，不做修改。

**Q: Token 超限的样本怎么办？**

A: 自动跳过蒸馏（与训练集处理方式一致），该样本 `ref_code = ""`，评测时 AAR / CodeBERTScore 自动为 0.0，不影响其他指标（XPR / SYR / PCR）。

**Q: 可以中断后继续吗？**

A: 可以。脚本检测 `test_dataset.json` 中已处理的 `paper_title`，自动跳过，从断点继续。

**Q: LSP 提示 `distill_code_repository` 无法解析怎么办？**

A: 这是静态分析误报。脚本在运行时动态将 `Paper2Code/data/paper2code/` 添加到 `sys.path`，实际运行不受影响。

---

*文档更新时间：2026-05-24*
*对应脚本：`evaluation/build_test_set_correct.py`*
