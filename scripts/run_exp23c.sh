#!/bin/bash
# Part XIX: their prompt variant (2) -- trailing space -- same grid, for comparison with variant (1).
cd /user/yac/LinearAblation
export PYTHONPATH=/tmp/claude-0/-user-yac-LinearAblation/f39766c0-3fc9-412b-b0c2-b5aa9c83d9fc/scratchpad/pylibs
PY=/user/yac/LinearSwap/.venv/bin/python
GRID=10,15,20,25,30,35,40,45,50
CUDA_VISIBLE_DEVICES=0 $PY scripts/exp23_kv_retrieval.py --model /user/yac/LinearSwap/models/Qwen3.5-0.8B --tag 0.8B-sp  --pairs $GRID --n 1000 --trailing_space --out results/exp23_kv_0.8B_space.json  > logs/exp23/grid_0.8B_space.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/exp23_kv_retrieval.py --model models/granite-4.0-h-1b            --tag G1b-sp   --pairs $GRID --n 1000 --trailing_space --out results/exp23_kv_G1b_space.json   > logs/exp23/grid_G1b_space.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY scripts/exp23_kv_retrieval.py --model /public/jyh/models/Qwen3.5-9B      --tag 9B-sp    --pairs $GRID --n 250  --trailing_space --out results/exp23_kv_9B_space.json    > logs/exp23/grid_9B_space.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY scripts/exp23_kv_retrieval.py --model models/granite-4.0-h-tiny          --tag Gtiny-sp --pairs $GRID --n 250  --trailing_space --out results/exp23_kv_Gtiny_space.json > logs/exp23/grid_Gtiny_space.log 2>&1 &
wait
echo "$(date +%H:%M) SPACE GRID DONE"
