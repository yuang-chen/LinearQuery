#!/usr/bin/env bash
# Reproduce every result in README.md.  Weights are frozen throughout; inputs are text only.
set -euo pipefail
cd "$(dirname "$0")"

PY=/user/yac/LinearSwap/.venv/bin/python        # torch 2.9.1, transformers 5.17.0, fla 0.6.0
PLOT=/user/miniconda3/envs/nha/bin/python       # matplotlib only (the venv above has none)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p results figures logs

# 1. task + baseline accuracy (4 / 8 / 16 pairs, 100 examples each)
$PY scripts/exp1_baseline.py --template chat --out results/exp1_baseline.json | tee logs/exp1.log

# 2. validation of the intervention machinery
$PY scripts/exp2_validate.py --dtype fp32 --out results/exp2_validate_fp32.json | tee logs/exp2.log

# 3. writer scan over GDN layers (100 discovery pairs) + controls
$PY scripts/exp3_writers.py   --out results/exp3_writers.json  2>&1 | tee logs/exp3.log
$PY scripts/exp3b_controls.py --out results/exp3b_controls.json 2>&1 | tee logs/exp3b.log

# 4. downstream readers: module screen, path open/block, head resolution, Q/K/V localisation
$PY scripts/exp4_readers.py  --out results/exp4_readers.json  2>&1 | tee logs/exp4.log
$PY scripts/exp4b_headpath.py --out results/exp4b_headpath.json 2>&1 | tee logs/exp4b.log

# 5. stability: 200 fresh pairs, target-query distance, distractor count, second template
$PY scripts/exp5_stability.py --out results/exp5_stability.json 2>&1 | tee logs/exp5.log

# figures
$PLOT scripts/plot_writers.py
$PLOT scripts/plot_readers.py
$PLOT scripts/plot_stability.py

# 6-8. what the GDN recurrent state is for
$PY scripts/exp6_state_role.py   --out results/exp6_state_role.json 2>&1 | tee logs/exp6.log
$PY scripts/exp7_state_probe.py  --n 200 --cut dict_end --out results/exp7_state_probe_dictend.json 2>&1 | tee logs/exp7.log
$PY scripts/exp7_state_probe.py  --n 200 --cut final    --out results/exp7_state_probe_final.json  2>&1 | tee logs/exp7_final.log
$PY scripts/exp8_horizon_ppl.py  --n_seq 16 --seq_len 512 --n_random 3 2>&1 | tee logs/exp8.log
$PLOT scripts/plot_state_role.py

# 9. why is GDN layer 0 so important?
$PY scripts/exp9_layer0.py 2>&1 | tee logs/exp9.log
$PLOT scripts/plot_layer0.py

# 10. affine re-embedding fit
$PY scripts/exp10_affine.py      --n_stat 2000 --n_eval 16 2>&1 | tee logs/exp10.log
$PY scripts/exp10b_affine_ood.py --n_stat 2000 --n_eval 16 2>&1 | tee logs/exp10b.log
$PLOT scripts/plot_affine.py

# 11. per-token NLL attribution
$PY scripts/exp11_per_token.py --n_stat 2000 --n_eval 32 2>&1 | tee logs/exp11.log
$PLOT scripts/plot_per_token.py

# 12. layer 0 vs layer 1
$PY scripts/exp12_layer01.py 2>&1 | tee logs/exp12.log
$PY scripts/exp12b_matched_noise.py 2>&1 | tee logs/exp12b.log
$PLOT scripts/plot_layer01.py

# 13. GDN layer groups and keep-only conditions
$PY scripts/exp13_groups.py 2>&1 | tee logs/exp13.log
$PLOT scripts/plot_groups.py

# 14. super-group roles
$PY scripts/exp14_supergroups.py 2>&1 | tee logs/exp14.log
$PLOT scripts/plot_supergroups.py

# 15. G3 resolved to layers; GDN heads in layers 0 and 14; where head 8 is needed
$PY scripts/exp15_heads.py 2>&1 | tee logs/exp15.log
$PY scripts/exp15b_head8.py 2>&1 | tee logs/exp15b.log

# 16. Q/K path patch at the reader (does G3 build the query or the keys?)
$PY scripts/exp16_qk_patch.py 2>&1 | tee logs/exp16.log
$PY scripts/exp16_qk_patch.py --abl "L8=8;L9=9;L10=10;G2=8,9,10;G1=4,5,6;G1+G2=4,5,6,8,9,10" \
    --out results/exp16b_qk_patch_g2.json 2>&1 | tee logs/exp16b.log

# 17. key-binding probe at the value tokens, and where the reader's attention goes
$PY scripts/exp17_binding_probe.py 2>&1 | tee logs/exp17.log
$PY scripts/exp17b_attn_targets.py 2>&1 | tee logs/exp17b.log

# 18. role of G1
$PY scripts/exp18_g1.py 2>&1 | tee logs/exp18.log
$PY scripts/exp18b_g1_cancel.py 2>&1 | tee logs/exp18b.log

# 19. which G0 layer binds each value token to its key
$PY scripts/exp19_binding_origin.py 2>&1 | tee logs/exp19.log

# 20. q.k selectivity of the reader with G2 / L10 / G3 removed from Q and/or K
$PY scripts/exp20_qk_selectivity.py 2>&1 | tee logs/exp20.log

# 21. generalisation battery on 0.8B (Part XVI): 9 variants over GPUs 0-7, then rev8 at L19 readers
bash scripts/run_exp21_08b.sh
for h in 6 7 5; do
  $PY scripts/exp21_generalize.py --tag 0.8B-L19H$h --variant rev8 --reader 19,$h --qk_groups 4,3,2,1 \
    2>&1 | tee logs/exp21/0.8B-L19H${h}_rev8.log
done

# 21b. same battery on Qwen3.5-9B, then list8/rev8 q.k split at their top readers
bash scripts/run_exp21_9b.sh
$PY scripts/exp21_generalize.py --model /public/jyh/models/Qwen3.5-9B --tag 9B-L19H15 --variant list8 --reader 19,15 --qk_groups 1,2,3,4       2>&1 | tee logs/exp21/9B-L19H15_list8.log
$PY scripts/exp21_generalize.py --model /public/jyh/models/Qwen3.5-9B --tag 9B-L27H1  --variant list8 --reader 27,1  --qk_groups 1,2,3,4,5,6   2>&1 | tee logs/exp21/9B-L27H1_list8.log
$PY scripts/exp21_generalize.py --model /public/jyh/models/Qwen3.5-9B --tag 9B-L27H1  --variant rev8  --reader 27,1  --qk_groups 1,2,3,4,5,6   2>&1 | tee logs/exp21/9B-L27H1_rev8.log
$PY scripts/exp21_generalize.py --model /public/jyh/models/Qwen3.5-9B --tag 9B-L23H12 --variant rev8  --reader 23,12 --qk_groups 1,2,3,4,5     2>&1 | tee logs/exp21/9B-L23H12_rev8.log
$PY scripts/summarize_exp21.py

# 22. query transplant: is the query group's write sufficient to redirect retrieval (Part XVII)
$PY scripts/exp22_query_transplant.py --tag 0.8B --variant chat8 2>&1 | tee logs/exp22_0.8B_chat8.log
$PY scripts/exp22_query_transplant.py --tag 0.8B --variant list8 2>&1 | tee logs/exp22_0.8B_list8.log
$PY scripts/exp22_query_transplant.py --tag 0.8B --variant rev8 --reader 19,6 2>&1 | tee logs/exp22_0.8B_rev8.log
$PY scripts/exp22_query_transplant.py --model /public/jyh/models/Qwen3.5-9B --tag 9B --variant chat8 2>&1 | tee logs/exp22_9B_chat8.log
$PY scripts/exp22_query_transplant.py --model /public/jyh/models/Qwen3.5-9B --tag 9B --variant list8 --reader 19,15 2>&1 | tee logs/exp22_9B_list8.log
$PY scripts/exp22_query_transplant.py --model /public/jyh/models/Qwen3.5-9B --tag 9B --variant rev8  --reader 27,1  2>&1 | tee logs/exp22_9B_rev8.log

# 23. second family: IBM Granite 4.0 H battery + transplant (Part XVIII)
hf download ibm-granite/granite-4.0-h-1b   --local-dir models/granite-4.0-h-1b
hf download ibm-granite/granite-4.0-h-tiny --local-dir models/granite-4.0-h-tiny
bash scripts/run_exp21_granite.sh
for t in "G1b models/granite-4.0-h-1b" "Gtiny models/granite-4.0-h-tiny"; do
  set -- $t
  $PY scripts/exp22_query_transplant.py --model $2 --tag $1 --variant chat8 --group_size 0 2>&1 | tee logs/exp22_$1_chat8.log
  $PY scripts/exp22_query_transplant.py --model $2 --tag $1 --variant list8 --group_size 0 2>&1 | tee logs/exp22_$1_list8.log
done
$PY scripts/exp22_query_transplant.py --model models/granite-4.0-h-1b   --tag G1b   --variant rev8 --group_size 0 --reader 35,6 2>&1 | tee logs/exp22_G1b_rev8.log
$PY scripts/exp22_query_transplant.py --model models/granite-4.0-h-tiny --tag Gtiny --variant rev8 --group_size 0 --reader 35,1 2>&1 | tee logs/exp22_Gtiny_rev8.log
