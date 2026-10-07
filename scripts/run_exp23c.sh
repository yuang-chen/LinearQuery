#!/bin/bash
# Part XIX: their prompt variant (2) -- trailing space -- same grid, for comparison with variant (1).
cd /mnt/yuang/LinearQuery
PY=/mnt/yuang/gdn2-in-place/.venv/bin/python
GRID=10,15,20,25,30,35,40,45,50
CUDA_VISIBLE_DEVICES=0 $PY scripts/exp23_kv_retrieval.py --model /mnt/yuang/gdn2-in-place/models/Qwen3.5-0.8B --tag 0.8B-sp  --pairs $GRID --n 1000 --trailing_space --out results/exp23_kv_0.8B_space.json  > logs/exp23/grid_0.8B_space.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/exp23_kv_retrieval.py --model models/granite-4.0-h-1b            --tag G1b-sp   --pairs $GRID --n 1000 --trailing_space --out results/exp23_kv_G1b_space.json   > logs/exp23/grid_G1b_space.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY scripts/exp23_kv_retrieval.py --model /mnt/yuang/models/Qwen3.5-9B      --tag 9B-sp    --pairs $GRID --n 250  --trailing_space --out results/exp23_kv_9B_space.json    > logs/exp23/grid_9B_space.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY scripts/exp23_kv_retrieval.py --model models/granite-4.0-h-tiny          --tag Gtiny-sp --pairs $GRID --n 250  --trailing_space --out results/exp23_kv_Gtiny_space.json > logs/exp23/grid_Gtiny_space.log 2>&1 &
wait
echo "$(date +%H:%M) SPACE GRID DONE"
