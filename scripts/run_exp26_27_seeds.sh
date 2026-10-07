#!/usr/bin/env bash
# Seed replicates (new key split + new dictionaries) of the headline exp 3 results, 0.8B <-> G1b chat8.
cd "$(dirname "$0")/.."
PY=/mnt/yuang/gdn2-in-place/.venv/bin/python
G=${GPU:-3}
for s in 1 2; do
  for p in "0.8B G1b" "G1b 0.8B"; do set -- $p
    CUDA_VISIBLE_DEVICES=$G $PY scripts/exp26_xfamily_query.py --donor $1 --recv $2 --seed $s \
      --out results/exp26_xquery_$1_to_$2_chat8_seed$s.json > logs/exp26/$1_to_$2_chat8_seed$s.log 2>&1
    CUDA_VISIBLE_DEVICES=$G $PY scripts/exp27_xfamily_block.py --donor $1 --recv $2 --seed $s \
      --out results/exp27_xblock_$1_into_$2_chat8_seed$s.json > logs/exp27/$1_into_$2_chat8_seed$s.log 2>&1
  done
done
