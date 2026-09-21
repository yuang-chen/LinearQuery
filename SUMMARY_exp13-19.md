# Summary for review: GDN roles in key–value retrieval (exp13–exp20)

> **Update (exp20, README Part XV):** the q·k selectivity test revises the G2 claims below.
> G2 contributes to **both** query and keys about equally (selectivity among values 5.57 →
> 3.28 with G2-less keys, → 2.95 with a G2-less query, → 1.85 both). G2-less *keys* lower the
> target's match strength (target falls below the sink; still top-1 among values); a G2-less
> *query* raises all values' scores — this, not the keys, is the "attention leaks to other
> values" failure. Part XI's "keys, not query" was an artefact of measuring attention mass.
> G3 is confirmed purely query-side (G3-less query: top-1 among values 0.11, chance 0.125).
> Script `scripts/exp20_qk_selectivity.py`, log `logs/exp20.log`.

Purpose of this note: a self-contained description of the setting, experiments, results and
conclusions of README Parts VII–XIV, written for an independent reviewer. It states the
methodological choices and the weaknesses I know of, so they can be checked rather than found.
Full tables are in `README.md`; every number here comes from `logs/exp*.log` / `results/exp*.json`.

---

## 1. Setting

**Model.** Qwen3.5-0.8B (`/user/yac/LinearSwap/models/Qwen3.5-0.8B`), frozen, **float32**,
eager attention where attention weights are read. 24 layers, hidden 1024. Gated DeltaNet (GDN)
"linear attention" in all layers except **3, 7, 11, 15, 19, 23** (softmax, 8 query heads, 2 KV
heads, head_dim 256, output-gated). Each GDN layer: 16 value heads × 128, a 4-tap causal
depthwise conv, a delta-rule recurrent state, gated RMSNorm, `out_proj`.

GDN layers are grouped by the softmax layer that follows them:
```
G0 [0,1,2] | attn 3 | G1 [4,5,6] | attn 7 | G2 [8,9,10] | attn 11 |
G3 [12,13,14] | attn 15 | G4 [16,17,18] | attn 19 | G5 [20,21,22] | attn 23
```

**Retrieval task** (`src/task.py`). Chat template, assistant turn prefilled so the answer is
the next token:
```
<|im_start|>user
Dictionary: garlic=fire, tomato=ice, almond=black, potato=gold, pepper=steel, grape=gray, fig=green, berry=blue.
Question: What is the value of grape?<|im_end|>
<|im_start|>assistant
<think>

</think>

The value of grape is **
```
- 8 pairs; keys from 22 words, values from 20 words, sampled independently without replacement;
  every key/value is a single token in context (verified).
- **Corrupted prompt:** the queried value is swapped with a distractor entry's value. Clean and
  corrupted prompts have identical length and differ at exactly 2 positions (asserted).
- **Eval set:** 100 examples = 4 (target, distractor) index configs {(2,5),(5,2),(1,6),(6,1)} × 25
  (seeds 11–14), each in clean and corrupted form = 200 prompts. Positions are shared within a config, so batches are position-aligned.
- Baseline: acc 1.000 on clean and corrupted prompts; `D_clean − D_corrupt` ≈ +20.8.

**Metrics.**
| name | definition |
|---|---|
| `acc` | argmax at final position = correct value, over 200 forwards (100 clean + 100 corrupted) |
| `gap` | mean over examples of `D_clean − D_corrupt`, `D = logit(clean ans) − logit(corrupt ans)` at final position |
| `lens@L15` | same gap, read from the residual stream right after layer 15 through final norm + lm_head (logit lens) |
| `attn` | softmax prob of reader head **L15H5** from the final token to the target value token. **exp15/15b/16: corrupted prompts; exp17b/18: clean prompts** (see §5) |
| PPL / ΔNLL | WikiText-2 test, 512-token chunks (12 chunks in exp15, 40 in exp18) |
| `rec` | `(gap_patched − gap_ablated) / (gap_intact − gap_ablated)` |

**Interventions.**
- *Remove a layer* = zero the mixer (GDN or softmax) output via forward hook; MLPs untouched.
- *Remove a GDN head* = zero its 128-dim slice of the `out_proj` input (after gated RMSNorm).
- *Position-restricted removal* = zero the mixer output only at listed positions (the layer's
  internal state is unaffected; only its write to the residual at those positions).
- *Q/K/V path patch* = overwrite layer-15 (or 19) `q_proj`/`k_proj`/`v_proj` output at given
  positions with the same prompt's activations from a donor run (before q/k-norm, RoPE).
  Q patch covers the query half only unless "+gate" (q_proj emits query and output gate).
- *conv 1 tap* = zero all but the current-token tap of the GDN conv weight.
- *no recurrence* = run token by token with the layer's recurrent state zeroed after every step
  (conv cache kept).
- *Linear probes* = 22-way logistic regression (LBFGS, L2 1e-3, standardised features),
  4,800 train / 1,600 test value tokens from fresh random dictionaries (seeds 1001 / 2002).

Prior context (Parts I–VI, earlier work): L15H5 is the primary reader (attends from the final
token to the source value); patching GDN-0 output at the source value token recovers ~1.04 of
the gap; layer 0 behaves as a learned re-embedding (token-type-mean surrogate keeps retrieval
at 1.000).

---

## 2. Experiments, results, conclusions

### exp13 / exp14 — group ablations (Parts VII–VIII, pre-existing)
Remove each group / super-group; logit lens; reader attention; state-reset locality.
- PPL: only G0 is catastrophic; G2, G3 cost less than a random triple (+0.13, +0.14 nats).
- Retrieval: G0 → 0.000, **G3 → 0.150**, G2 → 0.600, G1 → 0.840, G4 → 0.965, G5 → 0.940.
- G1+G2 and G3 removal destroy the reader's aim (attn 0.19 / 0.06) and the answer never forms at
  L15; G4+G5 removal leaves retrieval intact through L19 and loses the answer at L23.
- Conclusion then: pipeline G0 encode → G1+G2, G3 build the address → 15/19 retrieve → G4+G5 carry.

### exp15 — G3 layers; heads of GDN layers 0 and 14 (Part IX)
Setting: all subsets of {12,13,14}; single-head zeroing and keep-one-head in layers 0 and 14;
per-head |o_h|, forget-gate half-life (`ln0.5 / mean log-decay`, capped 1e6), token-identity R²
(between-token-type / total variance of the head output over 200×512 WikiText tokens).
| result | value |
|---|---|
| remove L14 / L13 / L12 | acc 0.870 / 0.995 / 1.000; attn 0.35 / 0.68 / 0.79 |
| keep only L14 of G3 | acc 1.000 |
| remove L13+L14 | acc 0.220 |
| zero L0 head 8 | **acc 0.000**, PPL 22.4 → 66.0 |
| zero L0 head 3 / 4 | acc 0.995 / 1.000, PPL 34.4 / 29.4 |
| any single L14 head | acc ≥ 0.965 |
| keep one head, L0 | acc 0.000 for all; PPL ≥ 4,119 |
| keep one head, L14 | acc 0.84–0.96 for all ≈ removing L14 (0.870) |
| L0 head 8 profile | largest |o_h| (2.80), half-life at cap, R² 0.996 |
| L14 heads | R² 0.10–0.49 |
Conclusion: L14 is the critical G3 layer (L13 a partial backup); L14's function is distributed
across heads. L0 head 8 is a single point of failure.

### exp15b — is L0 head 8 the Part I "writer"? (Part IX)
Setting: head-resolved clean→corrupt patch of GDN-0 output at the source value token; zero
head 8 at selected positions.
| result | value |
|---|---|
| writer-patch recovery by head | head 3 **+0.45**, head 8 +0.19, others ≤ 0.03 (sum ≈ 0.80 of whole-layer 1.04) |
| head 8 zeroed at source value / all values / query key / final | acc 1.000 each |
| head 8 zeroed everywhere except value tokens | acc 0.050 |
Conclusion: head 8 is not the value writer (head 3 is); it is needed at non-value positions.

### exp16 — Q vs K at the reader for G3 / L14 (Part X)
Setting: ablate L14, L13+14 or G3; rescue by restoring L15 Q (final token), K (all / value /
key-word tokens), V, from the intact run; necessity by inserting ablated Q/K into the intact run.
| G3 removed | attn | lens@L15 | acc |
|---|---|---|---|
| no patch | 0.056 | +0.6 | 0.150 |
| rescue Q@final | **0.425** | +11.0 | 0.680 |
| rescue K@all | 0.084 | +0.9 | 0.380 |
| rescue Q+K | 0.782 | +20.8 | 0.900 |
| rescue Q+K+V at L15 and L19 | 0.782 | +19.2 | 0.995 (gap +13.0 vs +20.8) |
| necessity: ablated Q into intact | 0.084 | +0.8 | — |
| necessity: ablated K into intact | 0.425 | +5.9 | — |
Internal check: "rescue Q" and "insert ablated K" are the same configuration and match (0.425).
Conclusion: **G3 builds mainly the reader's query**; keys contribute via interaction; V nothing;
G3 also affects L19 and later.

### exp16b — same for G2, its layers, G1 (Part XI)
| attn L15H5 | G1 | G2 | L10 |
|---|---|---|---|
| no patch | 0.640 | 0.284 | 0.533 |
| rescue Q@final | **0.725** | 0.291 | 0.492 |
| rescue K@value tokens | 0.616 | **0.562** | **0.753** |
| rescue K@key-word tokens | 0.641 | 0.283 | 0.535 |
| insert ablated K@value tokens into intact | 0.473 | **0.244** | 0.436 |
Layer subsets: remove L8 / L9 / L10 → acc 1.000 / 1.000 / 0.935.
Conclusion: **G2 (mainly L10) acts on the keys at the value tokens**; G1 mainly on the query.

### exp17 / 17b — does G2 create the key→value binding? (Part XII)
Setting: probe own key at each value token (intact / G1 / G2 / L10 / G3 / G1+G2 removed),
"fixed" (intact probe) and "retrain"; attention split of L15H5 by target category.
| result | value |
|---|---|
| own key decodable, intact, after layer 3 | 0.996 |
| G2 removed, retrained probe, any layer | ≥ 0.99 |
| G2 removed, intact probe on L15 K vector | 0.990 |
| attn to *other values*: intact / G2 removed / L10 removed | 0.06 / **0.40** / 0.31 |
| attn to `<|im_start|>` sink: intact / L14 removed | 0.06 / **0.38** |
Conclusion: binding hypothesis for G2 **rejected**. Without G2 the reader confuses values
(keys not discriminable); without L14 the query matches nothing (sink). Proposed: G2 makes the
key identity *selective along the query's read direction* — **not directly measured**.

### exp18 / 18b — role of G1 (Part XIII)
| test | result |
|---|---|
| ΔNLL on induction tokens (bigram seen earlier; 28.7 % of targets) vs other tokens | G1 **+0.07 vs +0.98** (ratio 0.07, lowest of all groups); G2 1.17, G3 1.40 |
| G1 layers | no single critical layer; [4,5,6] → acc 0.840 |
| G1 zeroed only on whole dictionary | acc 0.930, attn to other values 0.217 (≈ full removal) |
| G1 zeroed at question / final / after dictionary | acc 1.000 |
| remove G1, restore softmax 7+11 outputs | **acc 0.990**, lens +14.0 (intact +14.6) |
| remove G1, restore G2 mixers / MLPs 4–14 | acc 0.515 / **0.000** |
| keep G1 write, downstream from G1-less run | acc 0.73–0.81 (≈ removal) |
| G1-dependent part of MLPs 4–6 vs G1 write | cos **−0.51**, size 0.79×; MLPs 4–14 response 1.58× |
| removing L6 (or G1 at non-dictionary positions) | L15H5 sink attention → 0.000 |
Conclusion: G1 is a general language-modelling block, not a retrieval component; retrieval
uses it indirectly (best-supported route: softmax 7/11), over the dictionary as a whole. Its
write is tightly coupled with anti-aligned MLP responses.

### exp19 — which G0 layer makes the binding (Part XIV)
| test | result |
|---|---|
| own key from each mixer's own output at value token | L0 0.996 (CE 0.38), L1 1.000 (0.03), **L2 1.000 (0.01)**, attn3 0.988 |
| embedding | 0.043 (chance) |
| after L3, L0 token-local | 1.000 |
| after L3, L1 / L2 zeroed at value tokens | 0.981 / 0.943 |
| after L3, L1+L2 zeroed at value tokens | 0.726 |
| after L3, L0–2 conv 1 tap vs no recurrence | CE 0.91 vs 0.19 |
| after L3, L0–2 token-local | 0.576 |
| retrieval: L0 token-local | acc **0.170** (binding intact) |
| retrieval: L0–2 conv 1 tap | acc **0.000** (binding 0.888) |
Conclusion: the binding is a **redundant, conv-driven G0 product** (L0 first, L1–L2 re-write,
L2 strongest; softmax 3 backup); no single layer necessary. Retrieval collapses under G0 conv /
token-local interventions **without** losing the binding — unexplained.

---

## 3. Current overall model

```
G0 (L0 head 8; L0→L1→L2 conv; softmax 3 backup)   token identity; bind each value to its key
G1 (+ softmax 7)                                   general context features (LM), over the dictionary
G2 (mainly L10)                                    both sides: key match strength + query specificity (exp20)
G1 / G3 (mainly L14)                               the query's content (which entry)
softmax 15 (L15H5) / 19                            match query to key, copy value
G4 + G5                                            carry the answer to the output
```

---

## 4. Claims ranked by strength

| strength | claim |
|---|---|
| strong (direct, internally consistent) | L0 head 8 is necessary; L14 is the critical G3 layer; G3 acts mainly via L15's query and G2 mainly via L15's keys at value tokens (rescue ⇄ necessity agree); value-token key identity exists from layer 0 on and survives G2 removal; G1's PPL cost is on non-induction tokens |
| moderate | L14 function is distributed over heads; binding is conv-driven and redundant across L0–L2; different failure modes (value confusion vs sink) |
| revised by exp20 | G2 acts on both query specificity and key match strength (was: keys only, "discriminable along the query direction") |
| weak / inferred | G1 acts "through softmax 7/11"; head 8 "builds key-side context" |

---

## 5. Known methodological issues (please scrutinise)

1. **One template, one model, n = 100** examples (200 prompts), 8 pairs, 4 fixed position configs,
   ~62 tokens. No confidence intervals on retrieval metrics (a few scripts report SEM only for
   recovery). Accuracy near 1.000 compresses effects; `gap` is the more sensitive metric.
2. **Zero-ablation** (not mean/resample ablation) throughout. Zeroing a mixer output with a large
   typical norm pushes downstream layers off-distribution; some "necessity" effects may be
   partly off-distribution damage. Same concern for **conv 1 tap** (weights zeroed, no
   renormalisation) — exp19's retrieval collapses may be this artefact (Part III's
   token-type-mean surrogate kept retrieval at 1.000).
3. **Attention metric inconsistency:** `attn` is measured on corrupted prompts in exp15/15b/16/16b
   and on clean prompts in exp17b/18 (values differ slightly: intact 0.782 vs 0.794).
4. **Only one reader head is tracked** (L15H5). L19 heads matter too (rescuing L15 alone never
   restores the final gap). The Q/K story is established for L15H5 only.
5. **Logit lens** at L15 uses the final norm/unembedding on an intermediate residual — standard
   but uncalibrated at mid-depth.
6. **Mediation in exp18 is non-additive** (restoring more components can be worse than fewer);
   the "through softmax 7/11" claim rests on one condition.
7. **Probes:** linear decodability ≠ use. 22 classes with 4.8k samples in 1024-d; ceiling effects
   (hence CE also reported). "Fixed" probes on ablated models conflate information loss with
   representation shift. `resid3` probes include softmax 3's contribution.
8. **Position bookkeeping:** `key_pos` uses the first occurrence of each key token; the query key
   appears again in the question. "K@key tokens" patches only the dictionary occurrences.
9. **Induction definition** (exp18 D) is exact bigram repetition — crude.
10. **Half-life** for L0 head 8 hits the 1e6 cap; it is "no measurable decay", not a value.
11. **exp15 "keep only head 15 of L14 → 0.960"** was initially over-read by me as a rescue; it is
    within the band of every single-head condition (0.84–0.96) ≈ removing L14.
12. **Q patch excludes the output gate** by default; "+gate" variant sometimes does worse (not
    explained).
13. Donor activations for patches come from the **same prompt** in the intact run (not a
    different prompt), so rescue tests restore the exact intact values — appropriate for
    "which side is damaged", not for "what information is carried".

## 6. Experiments I think are still missing

1. ~~Direct q·k selectivity~~ — done in exp20 (Part XV); still open: the same at L19 heads.
2. **Resample/mean ablation** versions of the key ablations (G2, G3, L14, L0 head 8) to rule out
   zero-ablation artefacts.
3. **Generalisation:** other templates (`chat_alt` exists in `src/task.py`), more pairs / longer
   contexts, and an **MMLU-style letter-retrieval task** (Gather-and-Aggregate paper) with
   letter-swap counterfactuals.
4. **L19 readers:** repeat the Q/K rescue at L19 heads; identify which L19 heads read.
5. **What softmax 7/11 do** with G1's output (head-level patching).
6. **Resolve exp19 C:** retrieval under G0 conv interventions with a norm-matched or surrogate
   replacement, and localise which positions need G0's mixing (keys, question, template).
7. **Head 8 localisation:** which non-value positions (keys, `=`, `,`, question, template) need it.
8. **Statistics:** bootstrap CIs over prompts for acc/gap/attn; repeat probes with multiple seeds.

---

## 7. Files

| exp | script | log | results | README |
|---|---|---|---|---|
| 13 | `scripts/exp13_groups.py` | `logs/exp13.log` | `results/exp13_groups.json` | Part VII |
| 14 | `scripts/exp14_supergroups.py` | `logs/exp14.log` | `results/exp14_supergroups.json` | Part VIII |
| 15 | `scripts/exp15_heads.py` | `logs/exp15.log` | `results/exp15_heads.json` | Part IX |
| 15b | `scripts/exp15b_head8.py` | (none; regenerated by `run_all.sh`) | `results/exp15b_head8.json` | Part IX |
| 16 | `scripts/exp16_qk_patch.py` | `logs/exp16.log` | `results/exp16_qk_patch.json` | Part X |
| 16b | `scripts/exp16_qk_patch.py --abl "L8=8;L9=9;L10=10;G2=8,9,10;G1=4,5,6;G1+G2=4,5,6,8,9,10" --out results/exp16b_qk_patch_g2.json` | `logs/exp16b.log` | `results/exp16b_qk_patch_g2.json` | Part XI |
| 17 | `scripts/exp17_binding_probe.py` | `logs/exp17.log` | `results/exp17_binding_probe.json` | Part XII |
| 17b | `scripts/exp17b_attn_targets.py` | `logs/exp17b.log` | `results/exp17b_attn_targets.json` | Part XII |
| 18 | `scripts/exp18_g1.py` | `logs/exp18.log` | `results/exp18_g1.json` | Part XIII |
| 18b | `scripts/exp18b_g1_cancel.py` | `logs/exp18b.log` | `results/exp18b_g1_cancel.json` | Part XIII |
| 19 | `scripts/exp19_binding_origin.py` | `logs/exp19.log` | `results/exp19_binding_origin.json` | Part XIV |
| 20 | `scripts/exp20_qk_selectivity.py` | `logs/exp20.log` | `results/exp20_qk_selectivity.json` | Part XV |

Shared code: `src/task.py` (prompts), `src/positions.py` (token positions), `src/runner.py`
(model, hooks, patches), `src/qkv.py` (Q/K/V patch/capture, attention weights), `src/harness.py`
(position-aligned batches). Environment: `/user/yac/LinearSwap/.venv/bin/python`, one GPU
(`CUDA_VISIBLE_DEVICES=0` was used). Full pipeline: `bash run_all.sh`.
