#!/bin/bash
# Part XVI battery on Qwen3.5-9B: 9 variants (long2048 excluded) over GPUs 0-7.
# Launches are staggered so the 36 GB fp32 loads do not all hit the disk at once.
cd /user/yac/LinearAblation
PY=/user/yac/LinearSwap/.venv/bin/python
M=/public/jyh/models/Qwen3.5-9B
mkdir -p logs/exp21 results/exp21
run() {  # gpu variants...
  local gpu=$1; shift
  for v in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/exp21_generalize.py --model $M --tag 9B --variant $v \
      2>&1 | grep --line-buffered -v "Loading weights\|Generating .* split" > logs/exp21/9B_${v}.log
    echo "$(date +%H:%M) DONE 9B $v (exit ${PIPESTATUS[0]})"
  done
}
run 0 long512 & sleep 45
run 1 perm8 & sleep 45
run 2 mmlu & sleep 45
run 3 mmlu_hint & sleep 45
run 4 chat16 & sleep 45
run 5 rev8 & sleep 45
run 6 list8 & sleep 45
run 7 chat4 chat8 &
wait
echo "$(date +%H:%M) ALL DONE"
