#!/usr/bin/env bash
# Step 27 (exp 3B): stitched query-block transplant across families. One queue per GPU.
cd "$(dirname "$0")/.."
PY=/mnt/yuang/gdn2-in-place/.venv/bin/python
mkdir -p logs/exp27
q() {
  local g=$1; shift
  for t in "$@"; do set -- $t
    CUDA_VISIBLE_DEVICES=$g $PY scripts/exp27_xfamily_block.py --donor $1 --recv $2 --variant $3 \
      > logs/exp27/$1_into_$2_$3.log 2>&1
  done
}
q 4 "0.8B G1b chat8"  "0.8B G1b list8" "0.8B G1b rev8" &
q 5 "G1b 0.8B chat8"  "G1b 0.8B list8" "G1b 0.8B rev8" &
q 6 "9B G1b chat8"    "G1b 9B chat8"   "0.8B Gtiny chat8" &
q 7 "Gtiny 0.8B chat8" "0.8B 9B chat8" "9B 0.8B chat8" &
wait
