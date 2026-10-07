#!/bin/bash
# MMLU in the standard Hendrycks / lm-eval-harness format (subject header + 5 dev shots).
cd /mnt/yuang/LinearQuery
PY=/mnt/yuang/gdn2-in-place/.venv/bin/python
run(){ CUDA_VISIBLE_DEVICES=$1 $PY scripts/exp21_generalize.py --model $2 --tag $3 --variant $4 \
  2>&1 | grep --line-buffered -v "Loading weights\|Generating .* split" > logs/exp21/$3_$4.log
  echo "$(date +%H:%M) DONE $3 $4 (exit ${PIPESTATUS[0]})"; }
run 0 /mnt/yuang/gdn2-in-place/models/Qwen3.5-0.8B 0.8B-5shot mmlu &
run 1 /mnt/yuang/gdn2-in-place/models/Qwen3.5-0.8B 0.8B-5shot mmlu_hint &
sleep 30
run 2 /mnt/yuang/models/Qwen3.5-9B 9B-5shot mmlu &
sleep 45
run 3 /mnt/yuang/models/Qwen3.5-9B 9B-5shot mmlu_hint &
wait
echo "$(date +%H:%M) ALL DONE"
