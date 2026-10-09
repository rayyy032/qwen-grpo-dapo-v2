#!/usr/bin/env bash
# 七组单变量消融：除算法开关外超参完全一致。
# 用法: bash scripts/run_ablations.sh
# 环境变量可覆盖: BASE_MODEL / DATASET / MAX_STEPS / GROUP_SIZE / DEVICE
set -e

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
DATASET=${DATASET:-gsm8k_math}
MAX_STEPS=${MAX_STEPS:-150}
GROUP_SIZE=${GROUP_SIZE:-8}
DEVICE=${DEVICE:-auto}

for algo in grpo clip_higher dynamic token_level overlong dapo entropy_reg; do
    echo "=== training: ${algo} ==="
    python scripts/train.py --algorithm "${algo}" \
        --base-model "${BASE_MODEL}" \
        --dataset "${DATASET}" \
        --device "${DEVICE}" \
        --max-steps "${MAX_STEPS}" \
        --group-size "${GROUP_SIZE}" \
        --run-name "${algo}"
done

echo "All ablations complete."
