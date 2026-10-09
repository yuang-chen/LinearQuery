#!/bin/bash
# Whole-layer interventions (--unit layer: mixer + MLP, GnA's unit of removal) on Qwen3.5-0.8B/9B/27B:
# block ablations (exp21 B1/B2) and query transplants (exp22; for 27B also the combined blocks),
# chat8 / list8 / rev8, bf16. Compare with the mixer-only results of Parts XVI-XVII and exp22c.
cd /mnt/yuang/LinearQuery
PY=/mnt/yuang/gdn2-in-place/.venv/bin/python
mkdir -p results/unit_layer logs/unit_layer results/exp21
Q08=/mnt/yuang/gdn2-in-place/models/Qwen3.5-0.8B; Q9=/mnt/yuang/models/Qwen3.5-9B; Q27=/mnt/yuang/models/Qwen3.5-27B
b21(){ CUDA_VISIBLE_DEVICES=$1 $PY scripts/exp21_generalize.py --model $2 --tag $3-layer --variant $4 --unit layer \
         --stop_after B2 > logs/unit_layer/exp21_$3_$4.log 2>&1; echo "$(date +%H:%M) exp21 $3 $4 (exit $?)"; }
t22(){ CUDA_VISIBLE_DEVICES=$1 $PY scripts/exp22_query_transplant.py --model $2 --tag $3 --variant $4 --unit layer \
         --reader $5 --out results/unit_layer/exp22_$3_$4.json > logs/unit_layer/exp22_$3_$4.log 2>&1
       echo "$(date +%H:%M) exp22 $3 $4 (exit $?)"; }
c22(){ CUDA_VISIBLE_DEVICES=$1 $PY scripts/exp22_query_transplant.py --model $2 --tag $3 --variant $4 --unit layer \
         --reader $5 --watch "$6" --combine $7 --positions final,question \
         --out results/unit_layer/exp22c_$3_$4.json > logs/unit_layer/exp22c_$3_$4.log 2>&1
       echo "$(date +%H:%M) exp22c $3 $4 (exit $?)"; }
C27="9+10+11+12+13+14,12+13+14,9+10+11,10+11+12,6+7+8+9+10+11+12+13+14,1+2+3+4+5+6+7+8+9+10+11+12+13+14+15"
W27="39,15;43,22;51,20;59,21"
( for v in chat8 list8 rev8; do b21 0 $Q08 0.8B $v; done
  t22 0 $Q08 0.8B chat8 15,5; t22 0 $Q08 0.8B list8 15,5; t22 0 $Q08 0.8B rev8 19,6
  for v in chat8 list8 rev8; do b21 0 $Q9 9B $v; done
  t22 0 $Q9 9B chat8 19,11; t22 0 $Q9 9B list8 19,15; t22 0 $Q9 9B rev8 27,1 ) &
( for v in chat8 list8 rev8; do b21 1 $Q27 27B $v; done
  for v in chat8 list8 rev8; do t22 1 $Q27 27B $v 59,3; c22 1 $Q27 27B $v 59,3 "$W27" $C27; done ) &
wait; echo "$(date +%H:%M) ALL DONE"
