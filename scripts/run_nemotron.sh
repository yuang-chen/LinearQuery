#!/bin/bash
# Nemotron-H-8B (Reasoning-128K, thinking off): the Part XVIII battery (exp21) and the query
# transplant (exp22), with the Granite settings (--group_size 0: a group is every Mamba layer since
# the previous softmax layer). Two queues per GPU.
cd /mnt/yuang/LinearQuery
PY=/mnt/yuang/gdn2-in-place/.venv/bin/python
M=models/nemotron-h-8b-reasoning-128k
T=N8B
mkdir -p logs/exp21 results/exp21 logs/nemo
b21(){ CUDA_VISIBLE_DEVICES=$1 $PY scripts/exp21_generalize.py --model $M --tag $T --variant $2 --group_size 0 2>&1 \
         | grep --line-buffered -v "Loading weights\|Generating .* split" > logs/exp21/${T}_$2.log
       echo "$(date +%H:%M) DONE exp21 $2 (exit ${PIPESTATUS[0]})"; }
t22(){ CUDA_VISIBLE_DEVICES=$1 $PY scripts/exp22_query_transplant.py --model $M --tag $T --variant $2 --group_size 0 2>&1 \
         | grep --line-buffered -v "Loading weights" > logs/nemo/exp22_${T}_$2.log
       echo "$(date +%H:%M) DONE exp22 $2 (exit ${PIPESTATUS[0]})"; }
( b21 0 chat8; t22 0 chat8 ) &
( b21 0 list8; t22 0 list8 ) &
( b21 1 rev8;  t22 1 rev8 ) &
( for v in perm8 chat4 chat16 long512 mmlu mmlu_hint; do b21 1 $v; done ) &
wait
echo "$(date +%H:%M) ALL DONE"
