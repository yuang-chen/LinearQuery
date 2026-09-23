#!/bin/bash
# Part XIX, head sweeps for the two larger models (protocol: pairs=20, ablate each softmax head).
cd /user/yac/LinearAblation
export PYTHONPATH=/tmp/claude-0/-user-yac-LinearAblation/f39766c0-3fc9-412b-b0c2-b5aa9c83d9fc/scratchpad/pylibs
PY=/user/yac/LinearSwap/.venv/bin/python
Q9=/public/jyh/models/Qwen3.5-9B
GT=models/granite-4.0-h-tiny
g=(0 3 4 5)
for s in 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=${g[$((s-1))]} $PY scripts/exp23_kv_retrieval.py --model $Q9 --tag 9B --pairs 20 --n 200 \
    --heads attn --head_shard $s/4 --out results/exp23_heads_9B_$s.json > logs/exp23/heads_9B_$s.log 2>&1 &
done
for s in 1 2; do
  CUDA_VISIBLE_DEVICES=$((5+s)) $PY scripts/exp23_kv_retrieval.py --model $GT --tag Gtiny --pairs 20 --n 200 \
    --heads attn --head_shard $s/2 --out results/exp23_heads_Gtiny_$s.json > logs/exp23/heads_Gtiny_$s.log 2>&1 &
done
wait
echo "$(date +%H:%M) SWEEPS DONE"
