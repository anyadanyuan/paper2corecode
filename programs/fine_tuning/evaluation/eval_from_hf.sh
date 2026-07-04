#!/bin/bash
# ============================================================
# eval_from_hf.sh — 从 HuggingFace 下载微调模型并运行评测（含 TaskMonitor 监控）
#
# 使用前提：
#   1. pip install -r ../../../requirements.txt
#   2. export OPENROUTER_API_KEY="sk-..."  （可选，PCR 指标需要）
#
# 用法：bash eval_from_hf.sh
#
# 输出：
#   ../../../outputs/eval/hf_finetuned/summary.json    评测汇总
#   ../../../outputs/eval/hf_finetuned/results.jsonl   逐样本结果
#   ../../../outputs/eval/hf_finetuned/monitor/        TaskMonitor 显存记录
# ============================================================

set -euo pipefail

# ── 配置 ──────────────────────────────────────────────────────────────────
MODEL_NAME="Anyadanyuan/Qwen2.5-7B-paper2XCode"
TEST_DATASET="../../../data/test_dataset.json"
OUTPUT_DIR="../../../outputs/eval/hf_finetuned"
MAX_TOKENS=512

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " paper2CoreCode — Evaluation from HuggingFace"
echo "============================================================"
echo " Model:     $MODEL_NAME"
echo " Test Set:  $TEST_DATASET"
echo " Output:    $OUTPUT_DIR"
echo ""

# ── Step 1: 依赖检查 ─────────────────────────────────────────────────────
echo "[1/4] Checking dependencies..."

MISSING=()
for pkg in torch transformers accelerate peft; do
    python3 -c "import $pkg" 2>/dev/null || MISSING+=("$pkg")
done

python3 -c "import code_bert_score" 2>/dev/null || MISSING+=("code-bert-score")

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "  [FAIL] Missing packages: ${MISSING[*]}"
    echo "  Run: pip install -r requirements.txt"
    exit 1
fi
echo "  [OK] all dependencies found"

# ── Step 2: 检查测试集 ────────────────────────────────────────────────────
echo "[2/4] Checking test dataset..."
if [ ! -f "$TEST_DATASET" ]; then
    echo "  [FAIL] Test dataset not found: $TEST_DATASET"
    exit 1
fi
SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('$TEST_DATASET'))))")
echo "  [OK] $SAMPLE_COUNT samples found"

# ── Step 3: 运行评测 ─────────────────────────────────────────────────────
echo "[3/4] Starting evaluation (this may take 30+ minutes)..."
mkdir -p "$OUTPUT_DIR"

python3 evaluate.py \
    --model_path "$MODEL_NAME" \
    --dataset    "$TEST_DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --max_tokens $MAX_TOKENS \
    --resume

echo ""
echo "  [OK] evaluation complete"

# ── Step 4: 打印结果 ─────────────────────────────────────────────────────
echo "[4/4] Results:"
python3 - <<PYEOF
import json

with open("$OUTPUT_DIR/summary.json") as f:
    s = json.load(f)

print(f"  Valid samples: {s.get('valid_samples', 0)} / {s.get('total_samples', 0)}")
print(f"  XPR:           {s.get('xpr', 0):.2%}")
print(f"  SYR:           {s.get('syr', 0):.2%}")
print(f"  PCR:           {s.get('pcr_avg', 0):.2%}")
print(f"  CodeBERTScore: {s.get('codebertscore_avg', 0):.3f}")
print(f"  Overall:       {s.get('overall_score', 0):.2%}")
print(f"  Summary:       $OUTPUT_DIR/summary.json")
PYEOF

# TaskMonitor 汇总
MONITOR_DIR=$(ls -dt "$OUTPUT_DIR/monitor"/Eval_paper2CoreCode_* 2>/dev/null | head -1)
if [ -n "$MONITOR_DIR" ] && [ -f "$MONITOR_DIR/summary.json" ]; then
    echo ""
    echo "  --- TaskMonitor Summary ---"
    python3 -c "
import json
s = json.load(open('$MONITOR_DIR/summary.json'))
m = s['memory']
print(f\"  Elapsed:  {s['elapsed_min']:.1f} min\")
print(f\"  Peak GPU: {m['peak_reserved_mb']/1024:.1f} GB\")
print(f\"  Samples:  {s['sample_count']}\")
print(f\"  Monitor:  $MONITOR_DIR\")
"
fi

echo ""
echo "============================================================"
echo "Done."