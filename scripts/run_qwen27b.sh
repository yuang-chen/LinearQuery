#!/bin/bash
# Qwen3.5-27B and Qwen3.8-27B (64 layers, 48 GDN + 16 softmax, dense), float32: the exp21 battery
# and the exp22 query transplant on chat8 / list8 / rev8, Qwen settings (groups of 3 GDN layers).
# One model per GPU (~108 GB fp32); waits for the Nemotron-H queue and for both downloads.
cd /mnt/yuang/LinearQuery
PY=/mnt/yuang/gdn2-in-place/.venv/bin/python
mkdir -p logs/exp21 results/exp21 logs/q27
until grep -q "ALL DONE" logs/nemo/run.log; do sleep 60; done
echo "$(date +%H:%M) Nemotron queue finished"
q(){ local gpu=$1 tag=$2 model=$3
  until ! pgrep -f "hf download .*$(basename $model)" > /dev/null; do sleep 60; done
  for v in chat8 list8 rev8; do
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/exp21_generalize.py --model $model --tag $tag --variant $v 2>&1 \
      | grep --line-buffered -v "Loading weights\|Generating .* split" > logs/exp21/${tag}_$v.log
    echo "$(date +%H:%M) DONE exp21 $tag $v (exit ${PIPESTATUS[0]})"
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/exp22_query_transplant.py --model $model --tag $tag --variant $v 2>&1 \
      | grep --line-buffered -v "Loading weights" > logs/q27/exp22_${tag}_$v.log
    echo "$(date +%H:%M) DONE exp22 $tag $v (exit ${PIPESTATUS[0]})"
  done; }
q 0 Q38-27B /mnt/yuang/models/Qwen3.8-27B &
q 1 27B     /mnt/yuang/models/Qwen3.5-27B &
wait
echo "$(date +%H:%M) ALL DONE"
