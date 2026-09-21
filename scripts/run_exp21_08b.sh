#!/bin/bash
# Part XVI generalisation battery on Qwen3.5-0.8B: 9 variants (long2048 excluded) over GPUs 0-7.
cd /user/yac/LinearAblation
PY=/user/yac/LinearSwap/.venv/bin/python
M=/user/yac/LinearSwap/models/Qwen3.5-0.8B
mkdir -p logs/exp21 results/exp21
run() {  # gpu variants...
  local gpu=$1; shift
  for v in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/exp21_generalize.py --model $M --tag 0.8B --variant $v \
      2>&1 | grep --line-buffered -v "Loading weights\|Generating .* split" > logs/exp21/0.8B_${v}.log
    echo "DONE 0.8B $v (exit ${PIPESTATUS[0]})"
  done
}
run 0 long512 &
run 1 mmlu &
run 2 mmlu_hint &
run 3 perm8 &
run 4 chat16 &
run 5 rev8 &
run 6 list8 &
run 7 chat4 chat8 &
wait
echo "ALL DONE"
