#!/bin/bash
# Generalisation battery (Part XVI): 10 task variants x 2 model sizes, spread over GPUs 0-5.
cd /user/yac/LinearAblation
PY=/user/yac/LinearSwap/.venv/bin/python
M9=/public/jyh/models/Qwen3.5-9B
mkdir -p logs/exp21 results/exp21
run() {  # gpu tag model variants...
  local gpu=$1 tag=$2 model=$3; shift 3
  for v in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/exp21_generalize.py --model $model --tag $tag --variant $v \
      2>&1 | grep --line-buffered -v "Loading weights\|Warning\|\[transformers\]\|Generating" \
      > logs/exp21/${tag}_${v}.log
    echo "DONE $tag $v"
  done
}
run 1 9B $M9 chat8 long2048 &
run 2 9B $M9 chat4 mmlu &
run 3 9B $M9 chat16 mmlu_hint &
run 4 9B $M9 list8 rev8 &
run 5 9B $M9 perm8 long512 &
wait
echo "ALL DONE"
