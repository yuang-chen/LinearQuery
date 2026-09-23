#!/bin/bash
# Part XIX: the G&A paper's KV-Retrieval recipe (goombalab/Gather-and-Aggregate) on our models.
cd /user/yac/LinearAblation
export PYTHONPATH=/tmp/claude-0/-user-yac-LinearAblation/f39766c0-3fc9-412b-b0c2-b5aa9c83d9fc/scratchpad/pylibs
PY=/user/yac/LinearSwap/.venv/bin/python
Q08=/user/yac/LinearSwap/models/Qwen3.5-0.8B
Q9=/public/jyh/models/Qwen3.5-9B
G1=models/granite-4.0-h-1b
GT=models/granite-4.0-h-tiny
GRID=10,15,20,25,30,35,40,45,50
mkdir -p logs/exp23 results
# A. their dictionary-size grid, their 1000 samples where affordable
CUDA_VISIBLE_DEVICES=0 $PY scripts/exp23_kv_retrieval.py --model $Q08 --tag 0.8B  --pairs $GRID --n 1000 --gen > logs/exp23/grid_0.8B.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/exp23_kv_retrieval.py --model $G1  --tag G1b   --pairs $GRID --n 1000 --gen > logs/exp23/grid_G1b.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY scripts/exp23_kv_retrieval.py --model $Q9  --tag 9B    --pairs $GRID --n 250  --gen > logs/exp23/grid_9B.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY scripts/exp23_kv_retrieval.py --model $GT  --tag Gtiny --pairs $GRID --n 250  --gen > logs/exp23/grid_Gtiny.log 2>&1 &
# B. their head-ablation protocol at pairs=20, sharded over the remaining GPUs
for s in 1 2; do
  CUDA_VISIBLE_DEVICES=$((3+s)) $PY scripts/exp23_kv_retrieval.py --model $Q08 --tag 0.8B --pairs 20 --n 200 \
    --heads attn --head_shard $s/2 --out results/exp23_heads_0.8B_$s.json > logs/exp23/heads_0.8B_$s.log 2>&1 &
done
for s in 1 2; do
  CUDA_VISIBLE_DEVICES=$((5+s)) $PY scripts/exp23_kv_retrieval.py --model $G1 --tag G1b --pairs 20 --n 200 \
    --heads attn --head_shard $s/2 --out results/exp23_heads_G1b_$s.json > logs/exp23/heads_G1b_$s.log 2>&1 &
done
wait
echo "$(date +%H:%M) ALL DONE"
