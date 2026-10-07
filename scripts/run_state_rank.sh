#!/usr/bin/env bash
# Memory-state rank by circuit role (README Part XXII; adapted from LinearSwap tools/state_rank*.py).
# Three sources x four models, one queue per GPU, then the four reports and figures.
cd "$(dirname "$0")/.."
PY=/mnt/yuang/gdn2-in-place/.venv/bin/python
PLOT=/mnt/yuang/LinearQuery/.venv-plot/bin/python
mkdir -p logs/state_rank results/state_rank figures
r() { CUDA_VISIBLE_DEVICES=$1 $PY scripts/state_rank.py --tag $2 --model $3 ${4:+--source $4} ${5:+--variant $5} \
        > logs/state_rank/$2${4:+_$4}.log 2>&1; }
g=0
for t in "0.8B /mnt/yuang/gdn2-in-place/models/Qwen3.5-0.8B" "9B /mnt/yuang/models/Qwen3.5-9B" \
         "G1b models/granite-4.0-h-1b" "Gtiny models/granite-4.0-h-tiny"; do
  set -- $t
  ( r $g $1 $2; r $g $1 $2 longbench ) &
  r $((g+4)) $1 $2 dict chat16 &
  g=$((g+1))
done
wait
$PLOT scripts/state_rank_report.py 0.8B 9B G1b Gtiny --fig figures/state_rank.png \
      --csv per_head_rank_dclm.csv > results/state_rank/report_dclm.md
$PLOT scripts/state_rank_report.py 0.8B_longbench 9B_longbench G1b_longbench Gtiny_longbench \
      --fig figures/state_rank_longbench.png --csv per_head_rank_longbench.csv > results/state_rank/report_longbench.md
for m in final dict_end; do
  $PLOT scripts/state_rank_report.py 0.8B_dict 9B_dict G1b_dict Gtiny_dict --mark $m \
        --fig figures/state_rank_${m/final/dict}.png --csv per_head_rank_${m/final/dict}.csv \
        > results/state_rank/report_${m/final/dict}.md
done
