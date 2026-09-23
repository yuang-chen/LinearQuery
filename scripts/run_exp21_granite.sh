#!/bin/bash
# Part XVIII: the same battery on IBM Granite 4.0 H (Mamba-2 + attention hybrid).
# --group_size 0: a "group" is every linear layer since the previous softmax layer (5 or 9 here).
cd /user/yac/LinearAblation
PY=/user/yac/LinearSwap/.venv/bin/python
mkdir -p logs/exp21 results/exp21
run(){ local gpu=$1 tag=$2 model=$3; shift 3
  for v in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/exp21_generalize.py --model $model --tag $tag \
      --variant $v --group_size 0 2>&1 \
      | grep --line-buffered -v "Loading weights\|Generating .* split" > logs/exp21/${tag}_${v}.log
    echo "$(date +%H:%M) DONE $tag $v (exit ${PIPESTATUS[0]})"
  done; }
M1=models/granite-4.0-h-1b
MT=models/granite-4.0-h-tiny
run 0 G1b   $M1 chat8 chat4 chat16 &
run 1 G1b   $M1 list8 rev8 perm8 &
run 2 G1b   $M1 long512 &
run 3 G1b   $M1 mmlu mmlu_hint &
sleep 30
run 4 Gtiny $MT chat8 chat4 chat16 &
run 5 Gtiny $MT list8 rev8 perm8 &
run 6 Gtiny $MT long512 &
run 7 Gtiny $MT mmlu mmlu_hint &
wait
echo "$(date +%H:%M) ALL DONE"
