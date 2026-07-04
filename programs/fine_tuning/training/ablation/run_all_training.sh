#!/bin/bash
# ============================================================
# run_all_training.sh — 消融实验批量训练脚本（TaskMonitor 版）
#
# 功能：
#   - 串行训练 8 组模型（基座对比 5 组 + 量化对比 3 组）
#   - 使用 utils.nvidia_utils.run_monitored_command 记录显存与耗时
#   - 所有结果汇总到 training_log.json
#
# 运行位置：服务器
#   cd /root/paper2XAgent/programs/fine_tuning/ablation
#   bash run_all_training.sh
#
# 前提：
#   1. 已运行 python prepare_mini_data.py
#   2. 已运行 python generate_configs.py
#   3. conda activate llama
# ============================================================

# 注意：不使用 set -e，避免单组失败导致整体退出

# ── 环境配置 ──────────────────────────────────────────────────────────────
export DISABLE_VERSION_CHECK=1
export OMP_NUM_THREADS=4

export HF_HOME="/root/autodl-tmp/huggingface_cache"
export CUDA_VISIBLE_DEVICES=0

# 项目根目录 → 确保 utils.nvidia_utils 可导入
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

CONFIG_DIR="$SCRIPT_DIR/configs"
LOG_DIR="/root/autodl-tmp/ablation_logs"
OUTPUT_BASE="/root/autodl-tmp/ablation_outputs"
MONITOR_LOG_DIR="/root/autodl-tmp/ablation_monitor"
RESULTS_JSON="$SCRIPT_DIR/training_log.json"
RESULTS_JSON_BACKUP="$SCRIPT_DIR/training_log.json"

# LLaMA-Factory 必须从自己的目录启动
LLAMAFACTORY_DIR="/root/LLaMA-Factory"

mkdir -p "$LOG_DIR" "$OUTPUT_BASE" "$MONITOR_LOG_DIR"

# ── 工具函数 ──────────────────────────────────────────────────────────────

get_gpu_memory() {
    python3 -c "from utils.nvidia_utils import get_memory_snapshot; s=get_memory_snapshot(); print(int(s['reserved_mb']) if s else 0)"
}

append_json_result() {
    local name="$1" label="$2" group="$3" start_ts="$4"
    local exit_code="$5" log_file="$6" monitor_dir="$7"

    local mem_before="$8" mem_peak="$9" duration_sec="${10}" mem_after="${11}"

    if [ ! -f "$RESULTS_JSON" ]; then
        echo '{"experiments": []}' > "$RESULTS_JSON"
    fi

    python3 - <<PYEOF
import json

with open("$RESULTS_JSON", encoding="utf-8") as f:
    data = json.load(f)

data["experiments"].append({
    "name":           "$name",
    "label":          "$label",
    "group":          "$group",
    "start_time":     "$start_ts",
    "duration_sec":   $duration_sec,
    "duration_min":   round($duration_sec / 60, 2),
    "mem_before_mb":  $mem_before,
    "mem_peak_mb":    $mem_peak,
    "mem_after_mb":   $mem_after,
    "mem_peak_gb":    round($mem_peak / 1024, 2),
    "exit_code":      $exit_code,
    "log_file":       "$log_file",
    "monitor_dir":    "$monitor_dir",
    "status":         "success" if $exit_code == 0 else "failed"
})

with open("$RESULTS_JSON", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print("  -> written to training_log.json")
PYEOF
}

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
GPU_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
echo "============================================================"
echo " Ablation Experiment Batch Training (TaskMonitor)"
echo " Total $TOTAL groups | GPU Total: ${GPU_TOTAL} MB"
echo " Logs: $LOG_DIR"
echo " Monitor: $MONITOR_LOG_DIR"
echo " Results: $RESULTS_JSON"
echo "============================================================"

echo "{\"gpu_total_mb\": $GPU_TOTAL, \"experiments\": []}" > "$RESULTS_JSON"

FAILED_GROUPS=()

# ── 主训练循环 ────────────────────────────────────────────────────────────
for i in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME LABEL GROUP <<< "${EXPERIMENTS[$i]}"
    IDX=$((i + 1))
    CONFIG="$CONFIG_DIR/${NAME}.yaml"
    LOG_FILE="$LOG_DIR/${NAME}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo " [$IDX/$TOTAL] $LABEL"
    echo "  Group:    $GROUP"
    echo "  Config:   $CONFIG"
    echo "  Log:      $LOG_FILE"
    echo "------------------------------------------------------------"

    if [ ! -f "$CONFIG" ]; then
        echo "  [SKIP] config file not found: $CONFIG"
        FAILED_GROUPS+=("$NAME")
        continue
    fi

    MEM_BEFORE=$(get_gpu_memory)
    echo "  GPU before: ${MEM_BEFORE} MB"

    START_TS=$(date '+%Y-%m-%d %H:%M:%S')
    START_SEC=$(date +%s)

    echo "  Start: $START_TS"
    echo "  Training..."

    # 使用 TaskMonitor 包裹训练命令
    cd "$REPO_ROOT"
    EXIT_CODE=0
    python3 -c "
import sys
from utils.nvidia_utils import run_monitored_command
exit_code = run_monitored_command(
    name='ablation_$NAME',
    cmd=['llamafactory-cli', 'train', '$CONFIG'],
    log_dir='$MONITOR_LOG_DIR',
    interval=30,
    log_file='$LOG_FILE',
)
sys.exit(exit_code)
" || EXIT_CODE=$?
    cd "$SCRIPT_DIR"

    END_SEC=$(date +%s)
    END_TS=$(date '+%Y-%m-%d %H:%M:%S')
    DURATION=$((END_SEC - START_SEC))

    # 读取 TaskMonitor 记录的峰值显存
    MONITOR_DIR=$(ls -dt "$MONITOR_LOG_DIR"/ablation_"$NAME"_* 2>/dev/null | head -1)
    MEM_PEAK=0
    if [ -f "$MONITOR_DIR/summary.json" ]; then
        MEM_PEAK=$(python3 -c "import json; s=json.load(open('$MONITOR_DIR/summary.json')); print(int(s['memory']['peak_reserved_mb']))")
    fi
    [ "$MEM_PEAK" = "0" ] && MEM_PEAK="$MEM_BEFORE"

    MEM_AFTER=$(get_gpu_memory)
    DURATION_MIN=$(python3 -c "print(f'{$DURATION/60:.1f}')")
    MEM_PEAK_GB=$(python3 -c "print(f'{$MEM_PEAK/1024:.2f}')")

    echo "  End:       $END_TS"
    echo "  Duration:  ${DURATION}s (${DURATION_MIN} min)"
    echo "  Peak GPU:  ${MEM_PEAK} MB (${MEM_PEAK_GB} GB)"
    echo "  GPU after: ${MEM_AFTER} MB"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  Status: [OK]"
    else
        echo "  Status: [FAILED] (exit code: $EXIT_CODE)"
        FAILED_GROUPS+=("$NAME")
    fi

    append_json_result \
        "$NAME" "$LABEL" "$GROUP" \
        "$START_TS" "$EXIT_CODE" "$LOG_FILE" "$MONITOR_DIR" \
        "$MEM_BEFORE" "$MEM_PEAK" "$DURATION" "$MEM_AFTER"
done

# ── 最终汇总 ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Training Complete Summary"
echo "============================================================"

python3 - <<'PYEOF'
import json

with open("$RESULTS_JSON", encoding="utf-8") as f:
    data = json.load(f)

exps = data["experiments"]
print(f"\n{'Name':<42} {'Dur(min)':<10} {'Peak(GB)':<10} {'Status'}")
print("-" * 80)
for e in exps:
    status = "[OK]" if e["status"] == "success" else "[FAILED]"
    print(f"{e['label']:<42} {e['duration_min']:<10} {e['mem_peak_gb']:<10} {status}")

total_sec = sum(e["duration_sec"] for e in exps)
print(f"\nTotal time: {total_sec}s ({total_sec/60:.1f} min)")
success = sum(1 for e in exps if e["status"] == "success")
print(f"Success: {success}/{len(exps)}")
print(f"\nResults: $RESULTS_JSON")
PYEOF

if [ ${#FAILED_GROUPS[@]} -gt 0 ]; then
    echo ""
    echo "The following groups failed:"
    for g in "${FAILED_GROUPS[@]}"; do
        echo "   - $g  (log: $LOG_DIR/${g}.log)"
    done
fi

echo ""
echo "Next: bash run_all_evaluation.sh"