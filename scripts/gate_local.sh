#!/bin/bash
# Local MPS gate with auto-resume: Metal transient crashes (Apple M4 sustained
# load) kill the python process; each arm retries from its latest ckpt.
cd /Users/rickena/shixi01/qwen-grpo-dapo-v2
LOG=outputs/gate.log

latest_ckpt() {
  ls -d outputs/runs/gate_$1/ckpt_step* 2>/dev/null \
    | awk -F'step' '{print $2" "$0}' | sort -n | tail -1 | cut -d' ' -f2-
}

for arm in grpo clip_higher token_level dapo; do
  echo "=== GATE $arm start $(date +%m-%d\ %H:%M:%S) ===" >> $LOG
  for attempt in 1 2 3 4 5 6; do
    ckpt=$(latest_ckpt $arm)
    resume_args=""
    if [ -n "$ckpt" ] && [ -f "$ckpt/meta.json" ]; then
      resume_args="--resume-from $ckpt"
    fi
    echo "--- GATE $arm attempt $attempt ${resume_args:+resume=$ckpt} $(date +%H:%M:%S) ---" >> $LOG
    .venv/bin/python scripts/train.py --algorithm $arm --dataset gsm8k_math \
      --train-samples 400 --eval-samples 60 --group-size 8 --groups-per-step 2 \
      --max-new-tokens 384 --max-steps 24 --learning-rate 2e-5 --ppo-epochs 4 \
      --eval-every 12 --save-every 12 --device mps --use-swanlab false \
      --run-name gate_$arm $resume_args >> $LOG 2>&1
    rc=$?
    echo "--- GATE $arm attempt $attempt rc=$rc $(date +%H:%M:%S) ---" >> $LOG
    if [ $rc -eq 0 ]; then
      echo "=== GATE $arm DONE $(date +%H:%M:%S) ===" >> $LOG
      break
    fi
    sleep 10
  done
done
echo "ALL_DONE $(date +%m-%d\ %H:%M:%S)" >> $LOG
