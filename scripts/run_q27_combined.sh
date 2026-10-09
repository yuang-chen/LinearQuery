#!/bin/bash
# 27B combined final-token transplants: unions of the blocks feeding the reader banks (softmax layers
# 39-59), with heads from every bank watched. Groups G_i = layers 4i..4i+2 (softmax at 4i+3).
cd /mnt/yuang/LinearQuery
PY=/mnt/yuang/gdn2-in-place/.venv/bin/python
C="9+10+11+12+13+14,12+13+14,9+10+11,10+11+12,6+7+8+9+10+11+12+13+14,1+2+3+4+5+6+7+8+9+10+11+12+13+14+15"
x(){ CUDA_VISIBLE_DEVICES=$1 $PY scripts/exp22_query_transplant.py --model $2 --tag $3 --variant $4 \
       --reader $5 --watch "$6" --combine $C --positions final,question \
       --out results/exp22c_combined_$3_$4.json 2>&1 | grep -v "Loading weights" > logs/q27/exp22c_$3_$4.log
     echo "$(date +%H:%M) DONE $3 $4 (exit ${PIPESTATUS[0]})"; }
Q35=/mnt/yuang/models/Qwen3.5-27B; Q38=/mnt/yuang/models/Qwen3.8-27B
( for v in chat8 list8 rev8; do x 0 $Q35 27B $v 59,3 "39,15;43,22;51,20;59,21"; done ) &
( x 1 $Q38 Q38-27B chat8 43,14 "39,15;51,23;59,3;59,21"
  for v in list8 rev8; do x 1 $Q38 Q38-27B $v 59,3 "43,14;47,20;51,23;59,21"; done ) &
wait; echo "$(date +%H:%M) ALL DONE"
