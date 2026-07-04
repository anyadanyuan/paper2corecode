#!/bin/bash
# ============================================================
# run_all_evaluation.sh — 消融实验批量评估脚本（TaskMonitor 版）
#
# 功能：
#   - 对 8 组训练结果依次评估
#   - 跳过 PCR（需要 LLM API）
#   - 使用 utils.nvidia_utils.TaskMonitor 记录显存与耗时
#   - 结果汇总到 eval_log.json
#
# 运行位置：服务器
#   cd /root/paper2XAgent/programs/fine_tuning/ablation
#   bash run_all_evaluation.sh
#
# 前提：
#   1. 训练脚本已完成 (run_all_training.sh)
#   2. conda activate llama
# ============================================================

export DISABLE_VERSION_CHECK=1
export OMP_NUM_THREADS=4

export HF_HOME="/root/autodl-tmp/huggingface_cache"
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

EVAL_DIR="$REPO_ROOT/programs/fine_tuning/evaluation"
OUTPUT_BASE="/root/autodl-tmp/ablation_outputs"
EVAL_RESULTS_BASE="/root/autodl-tmp/ablation_eval"
MONITOR_LOG_DIR="/root/autodl-tmp/ablation_eval_monitor"
MINI_TEST="/root/autodl-tmp/ablation_data/test_mini.json"
EVAL_LOG="$SCRIPT_DIR/eval_log.json"
LOG_DIR="/root/autodl-tmp/ablation_logs"

mkdir -p "$EVAL_RESULTS_BASE" "$MONITOR_LOG_DIR"

# ── 实验组列表 ────────────────────────────────────────────────────────────
EXPERIMENTS=(
    "group1_qwen_coder_4bit|Qwen2.5-Coder-7B-Instruct (4-bit QLoRA)|backbone"
    "group2_qwen_7b_4bit|Qwen2.5-7B-Instruct (4-bit QLoRA)|backbone"
    "group3_deepseek_coder_4bit|DeepSeek-Coder-V2-Lite-Instruct (4-bit QLoRA)|backbone"
    "group4_codellama_4bit|CodeLlama-7b-Instruct (4-bit QLoRA)|backbone"
    "group5_mistral_4bit|Mistral-7B-Instruct-v0.3 (4-bit QLoRA)|backbone"
    "group6_qwen_coder_bf16_full|BF16 full-parameter finetuning|quantization"
    "group7_qwen_coder_bf16_lora|BF16 + LoRA (no quantization)|quantization"
    "group8_qwen_coder_8bit|8-bit QLoRA|quantization"
)

TOTAL=${#EXPERIMENTS[@]}
echo "============================================================"
echo " Ablation Experiment Batch Evaluation (TaskMonitor)"
echo " Total $TOTAL groups | Test Set: $MINI_TEST"
echo "============================================================"

echo '{"experiments": []}' > "$EVAL_LOG"

for i in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME LABEL GROUP <<< "${EXPERIMENTS[$i]}"
    IDX=$((i + 1))

    MODEL_PATH="$OUTPUT_BASE/$NAME"
    EVAL_OUT="$EVAL_RESULTS_BASE/$NAME"
    EVAL_LOG_FILE="$LOG_DIR/eval_${NAME}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo " [$IDX/$TOTAL] $LABEL"
    echo "  Model: $MODEL_PATH"
    echo "  Output: $EVAL_OUT"
    echo "------------------------------------------------------------"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "  [SKIP] model dir not found: $MODEL_PATH"
        continue
    fi

    mkdir -p "$EVAL_OUT"

    # 记录评估前显存
    MEM_BEFORE=$(python3 -c "from utils.nvidia_utils import get_memory_snapshot; s=get_memory_snapshot(); print(int(s['reserved_mb']) if s else 0)")
    echo "  GPU before: ${MEM_BEFORE} MB"

    START_TS=$(date '+%Y-%m-%d %H:%M:%S')
    START_SEC=$(date +%s)
    echo "  Start: $START_TS"

    # 使用 TaskMonitor 包裹评估命令
    cd "$REPO_ROOT"
    EXIT_CODE=0
    python3 -c "
import sys, json
from utils.nvidia_utils import run_monitored_command

exit_code = run_monitored_command(
    name='eval_$NAME',
    cmd=[
        'python', '$EVAL_DIR/evaluate.py',
        '--model_path', '$MODEL_PATH',
        '--dataset', '$MINI_TEST',
        '--output_dir', '$EVAL_OUT',
        '--max_tokens', '512',
    ],
    log_dir='$MONITOR_LOG_DIR',
    interval=15,
    log_file='$EVAL_LOG_FILE',
)
sys.exit(exit_code)
" || EXIT_CODE=$?
    cd "$SCRIPT_DIR"

    END_SEC=$(date +%s)
    END_TS=$(date '+%Y-%m-%d %H:%M:%S')
    DURATION=$((END_SEC - START_SEC))

    # 读取 TaskMonitor 峰值显存
    MONITOR_DIR=$(ls -dt "$MONITOR_LOG_DIR"/eval_"$NAME"_* 2>/dev/null | head -1)
    MEM_PEAK=0
    if [ -f "$MONITOR_DIR/summary.json" ]; then
        MEM_PEAK=$(python3 -c "import json; s=json.load(open('$MONITOR_DIR/summary.json')); print(int(s['memory']['peak_reserved_mb']))")
    fi
    [ "$MEM_PEAK" = "0" ] && MEM_PEAK="$MEM_BEFORE"

    MEM_AFTER=$(python3 -c "from utils.nvidia_utils import get_memory_snapshot; s=get_memory_snapshot(); print(int(s['reserved_mb']) if s else 0)")
    DURATION_MIN=$(python3 -c "print(f'{$DURATION/60:.1f}')")
    MEM_PEAK_GB=$(python3 -c "print(f'{$MEM_PEAK/1024:.2f}')")

    echo "  Duration: ${DURATION}s (${DURATION_MIN} min)"
    echo "  Peak GPU: ${MEM_PEAK} MB (${MEM_PEAK_GB} GB)"

    # 读取评估结果并写入 eval_log.json
    SUMMARY_FILE="$EVAL_OUT/summary.json"
    if [ $EXIT_CODE -eq 0 ] && [ -f "$SUMMARY_FILE" ]; then
        echo "  [OK] evaluation successful, reading results..."
        python3 - <<PYEOF
import json

with open("$SUMMARY_FILE", encoding="utf-8") as f:
    s = json.load(f)

print(f"    XPR:           {s.get('xpr', 0):.2%}")
print(f"    SYR:           {s.get('syr', 0):.2%}")
print(f"    AAR:           {s.get('aar_avg', 0):.2%}")
print(f"    CodeBERTScore: {s.get('codebertscore_avg', 0):.3f}")
print(f"    Overall:       {s.get('overall_score', 0):.2%}")

with open("$EVAL_LOG", encoding="utf-8") as f:
    log = json.load(f)

log["experiments"].append({
    "name":            "$NAME",
    "label":           "$LABEL",
    "group":           "$GROUP",
    "start_time":      "$START_TS",
    "duration_sec":    $DURATION,
    "duration_min":    round($DURATION / 60, 2),
    "mem_before_mb":   $MEM_BEFORE,
    "mem_peak_mb":     $MEM_PEAK,
    "mem_peak_gb":     round($MEM_PEAK / 1024, 2),
    "mem_after_mb":    $MEM_AFTER,
    "exit_code":       $EXIT_CODE,
    "monitor_dir":     "$MONITOR_DIR",
    "metrics": {
        "xpr":            s.get("xpr", 0),
        "syr":            s.get("syr", 0),
        "aar":            s.get("aar_avg", 0),
        "codebertscore":  s.get("codebertscore_avg", 0),
        "overall":        s.get("overall_score", 0),
    },
    "status": "success"
})

with open("$EVAL_LOG", "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2, ensure_ascii=False)
print("  -> written to eval_log.json")
PYEOF
    else
        echo "  [FAILED] evaluation failed (exit code: $EXIT_CODE), log: $EVAL_LOG_FILE"
        python3 - <<PYEOF
import json
with open("$EVAL_LOG", encoding="utf-8") as f:
    log = json.load(f)
log["experiments"].append({
    "name": "$NAME", "label": "$LABEL", "group": "$GROUP",
    "duration_sec": $DURATION, "mem_peak_gb": round($MEM_PEAK/1024, 2),
    "exit_code": $EXIT_CODE, "monitor_dir": "$MONITOR_DIR",
    "metrics": {}, "status": "failed"
})
with open("$EVAL_LOG", "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2, ensure_ascii=False)
PYEOF
    fi
done

# ── 汇总展示 ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Evaluation Results Summary"
echo "============================================================"

python3 - <<'PYEOF'
import json

with open("$EVAL_LOG", encoding="utf-8") as f:
    data = json.load(f)

exps = data["experiments"]
print(f"\n{'Name':<42} {'XPR':>6} {'SYR':>6} {'CBS':>7} {'Overall':>8} {'Dur(min)':>9} {'Peak':>8}")
print("-" * 90)
for e in exps:
    m = e.get("metrics", {})
    status = "" if e["status"] == "success" else " [FAILED]"
    print(
        f"{e['label']:<42} "
        f"{m.get('xpr',0):>6.1%} "
        f"{m.get('syr',0):>6.1%} "
        f"{m.get('codebertscore',0):>7.3f} "
        f"{m.get('overall',0):>8.1%} "
        f"{e.get('duration_min',0):>9.1f} "
        f"{e.get('mem_peak_gb',0):>6.2f}GB"
        f"{status}"
    )

print(f"\nDetails: $EVAL_LOG")
print("Next: python extrapolate_results.py")
PYEOF