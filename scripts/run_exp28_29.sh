#!/usr/bin/env bash
# Step 28-29: Bind-block and whole-circuit transplant across families; token-level Bind swaps
# (in-family sufficiency on all four hosts + cross-family). One queue per GPU.
cd "$(dirname "$0")/.."
PY=/mnt/yuang/gdn2-in-place/.venv/bin/python
mkdir -p logs/exp28 logs/exp29
c28() { CUDA_VISIBLE_DEVICES=$1 $PY scripts/exp28_xfamily_circuit.py --donor $2 --recv $3 --seed $4 \
          --out results/exp28_xcircuit_$2_into_$3_chat8${5:-}.json > logs/exp28/$2_into_$3_chat8${5:-}.log 2>&1; }
c29() { CUDA_VISIBLE_DEVICES=$1 $PY scripts/exp29_bind_transplant.py --host $2 --donor $3 \
          > logs/exp29/$3_into_$2_chat8.log 2>&1; }
( c28 0 0.8B G1b 0;          c29 0 G1b 0.8B ) &
( c28 1 G1b 0.8B 0;          c29 1 0.8B G1b ) &
( c28 2 0.8B G1b 1 _seed1;   c29 2 Gtiny 0.8B ) &
( c28 3 G1b 0.8B 1 _seed1;   c29 3 G1b 9B ) &
( c28 4 0.8B G1b 2 _seed2 ) &
( c28 5 G1b 0.8B 2 _seed2 ) &
( c28 6 9B G1b 0 ) &
( c28 7 G1b 9B 0;            c29 7 9B G1b ) &
wait
