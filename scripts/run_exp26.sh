#!/usr/bin/env bash
# Step 26 (exp 3A): cross-family query transfer, Qwen3.5 <-> Granite-4.0-H. One queue per GPU.
cd "$(dirname "$0")/.."
PY=/mnt/yuang/gdn2-in-place/.venv/bin/python
mkdir -p logs/exp26
q() {  # gpu, then "donor recv variant" triples
  local g=$1; shift
  for t in "$@"; do set -- $t
    CUDA_VISIBLE_DEVICES=$g $PY scripts/exp26_xfamily_query.py --donor $1 --recv $2 --variant $3 \
      > logs/exp26/$1_to_$2_$3.log 2>&1
  done
}
q 4 "0.8B G1b chat8" "G1b 0.8B chat8" "0.8B G1b list8" "G1b 0.8B list8" &
q 5 "0.8B G1b rev8"  "G1b 0.8B rev8"  "Gtiny 0.8B chat8" "0.8B 9B chat8" &
q 6 "9B G1b chat8"   "9B 0.8B chat8"  "Gtiny G1b chat8" &
q 7 "G1b 9B chat8"   "Gtiny 9B chat8" "9B G1b list8" "G1b 9B list8" &
wait
