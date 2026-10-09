#!/bin/bash
# Precision check: re-run the headline measurements in bfloat16 (the models' release precision) and
# compare with the float32 results. chat8 only:
#   exp21 B1+B2  baseline and block ablations             -> results/exp21/<tag>-bf16_chat8.json
#   exp22        single-block query transplant            -> results/bf16/exp22_<tag>_chat8.json
#   exp22c       combined transplants (multi-bank models) -> results/bf16/exp22c_<tag>_chat8.json
cd /mnt/yuang/LinearQuery
PY=/mnt/yuang/gdn2-in-place/.venv/bin/python
mkdir -p results/bf16 logs/bf16
Q08=/mnt/yuang/gdn2-in-place/models/Qwen3.5-0.8B; Q9=/mnt/yuang/models/Qwen3.5-9B
Q27=/mnt/yuang/models/Qwen3.5-27B; Q38=/mnt/yuang/models/Qwen3.8-27B
G1=models/granite-4.0-h-1b; GT=models/granite-4.0-h-tiny; N8=models/nemotron-h-8b-reasoning-128k
b21(){ CUDA_VISIBLE_DEVICES=$1 $PY scripts/exp21_generalize.py --model $2 --tag $3-bf16 --variant chat8 $4 \
         --dtype bf16 --stop_after B2 > logs/bf16/exp21_$3.log 2>&1; echo "$(date +%H:%M) exp21 $3 (exit $?)"; }
t22(){ CUDA_VISIBLE_DEVICES=$1 $PY scripts/exp22_query_transplant.py --model $2 --tag $3 --variant chat8 $4 \
         --reader $5 --dtype bf16 --out results/bf16/exp22_$3_chat8.json > logs/bf16/exp22_$3.log 2>&1
       echo "$(date +%H:%M) exp22 $3 (exit $?)"; }
c22(){ CUDA_VISIBLE_DEVICES=$1 $PY scripts/exp22_query_transplant.py --model $2 --tag $3 --variant chat8 $4 \
         --reader $5 --watch "$6" --combine $7 --positions final,question --dtype bf16 \
         --out results/bf16/exp22c_$3_chat8.json > logs/bf16/exp22c_$3.log 2>&1
       echo "$(date +%H:%M) exp22c $3 (exit $?)"; }
C27="9+10+11+12+13+14,12+13+14,9+10+11,10+11+12,6+7+8+9+10+11+12+13+14,1+2+3+4+5+6+7+8+9+10+11+12+13+14+15"
( b21 0 $Q08 0.8B ""; b21 0 $Q9 9B ""; b21 0 $G1 G1b "--group_size 0"
  t22 0 $Q08 0.8B "" 15,5; t22 0 $Q9 9B "" 19,11; t22 0 $G1 G1b "--group_size 0" 25,1
  c22 0 $Q27 27B "" 59,3 "39,15;43,22;51,20;59,21" $C27 ) &
( b21 1 $GT Gtiny "--group_size 0"; b21 1 $N8 N8B "--group_size 0"; b21 1 $Q27 27B ""
  t22 1 $GT Gtiny "--group_size 0" 25,2; t22 1 $N8 N8B "--group_size 0" 29,8
  c22 1 $GT Gtiny "--group_size 0" 25,2 "25,8;35,1;35,11;35,7" 2,3,2+3,1+2+3
  c22 1 $N8 N8B   "--group_size 0" 29,8 "29,15;40,11;40,8;40,25" 2,3,2+3,1+2+3
  c22 1 $Q38 Q38-27B "" 43,14 "39,15;51,23;59,3;59,21" $C27 ) &
wait; echo "$(date +%H:%M) ALL DONE"
