# GDN → softmax pathways in Qwen3.5-0.8B

An exploratory activation-patching study asking whether a Gated DeltaNet (GDN) component
supplies information that a later softmax attention head uses for key–value retrieval.
Weights are frozen; all inputs are text.

Model: `/mnt/yuang/gdn2-in-place/models/Qwen3.5-0.8B` — 24 layers, `hidden=1024`,
GDN ("linear_attention") everywhere except layers **3, 7, 11, 15, 19, 23**, which are
softmax attention (8 query heads, 2 KV heads, `head_dim=256`, GQA group 4).

## Reproduce

```bash
bash /mnt/yuang/LinearQuery/run_all.sh
```

Environment (reused, nothing installed into it): `/mnt/yuang/gdn2-in-place/.venv`
(torch 2.9.1+cu128, transformers 5.17.0, flash-linear-attention 0.6.0). Figures are drawn
with `/mnt/yuang/LinearQuery/.venv-plot/bin/python` because that venv has no matplotlib.
All model runs are **float32** (see "Validation"). The full pipeline takes ≈45 min on one L20X.

Individual steps:

```bash
PY=/mnt/yuang/gdn2-in-place/.venv/bin/python
$PY scripts/exp1_baseline.py --template chat          # task + baseline accuracy
$PY scripts/exp2_validate.py --dtype fp32             # intervention sanity checks
$PY scripts/exp3_writers.py                           # GDN writer scan
$PY scripts/exp3b_controls.py                         # writer controls
$PY scripts/exp4_readers.py                           # reader screen + path open/block + heads
$PY scripts/exp4b_headpath.py                         # head-level Q/K/V localisation
$PY scripts/exp5_stability.py                         # stability across conditions
```

Raw results: `results/*.json` (per-example recovery values). Logs: `logs/*.log`.
Figures: `figures/fig1_writers.png`, `fig2_readers.png`, `fig3_stability.png`, `fig4_state_role.png`, `fig5_layer0.png`, `fig6_affine.png`, `fig7_per_token.png`, `fig8_layer01.png`, `fig9_groups.png`, `fig10_supergroups.png`.

## 1. Task

Prompt (chat template, assistant turn prefilled so the answer is the very next token):

```
<|im_start|>user
Dictionary: elder=iron, carrot=red, cherry=steel, mango=wood, ...
Question: What is the value of mango?<|im_end|>
<|im_start|>assistant
<think>

</think>

The value of mango is **
```

Keys and values were filtered so that **every value is a single token in the rendered
context** and the answer token is that same token id. A corrupted prompt swaps the queried
entry's value with another entry's value, so clean and corrupted prompts have identical
token counts and differ at exactly two positions (asserted for every example).

Metric: `D = logit(clean answer) − logit(corrupted answer)` at the final position, and
`recovery = (D_patched − D_corrupted) / (D_clean − D_corrupted)`.

Baseline (100 examples per size, `results/exp1_baseline.json`):

| pairs | tokens | acc. clean | acc. corrupted | D_clean | D_corrupt |
|---|---|---|---|---|---|
| 4  | 46 | 1.00 | 1.00 | +8.08  | −8.01  |
| 8  | 62 | 1.00 | 1.00 | +10.07 | −10.07 |
| 16 | 94 | 1.00 | 1.00 | +11.56 | −11.03 |

All three settings are solved perfectly in both versions. **8 pairs** is used for discovery.

> An earlier plain-text template ("Answer: ") reached only D≈+6 with 0 % argmax accuracy —
> the model's digit prior beat the colour words. The chat template with the `**` prefill
> fixes the format without changing the retrieval structure.

## 2. Validation of the intervention code (`results/exp2_validate_fp32.json`, n=10)

| check | result |
|---|---|
| split execution (prefix+cache → suffix) vs ordinary forward | max abs logit error **0.011**, argmax match 10/10 |
| identity recurrent-state patch (clean → clean) | max |ΔD| = **0.007** |
| identity GDN-output patch (clean → clean) | max |ΔD| = **0.000** |
| cache isolation (plain run after a patched run) | max |ΔD| = **0.000** |

In bfloat16 the same checks give max |ΔD| = 0.125, ~0.6 % of the 20-point clean–corrupt gap.
Everything reported below is float32.

Recurrent state and convolution cache are kept separate: a state patch replaces only
`cache.layers[L].recurrent_states[0]` and the suffix `[t+1:]` is then recomputed. Consistent
with the caveat in the plan, patching a GDN *output* at position *t* does **not** repair the
state that *t* writes — both are measured separately throughout.

## 3. Writer scan (figure 1, 100 discovery pairs, 4 position configurations)

| intervention | best layer | recovery |
|---|---|---|
| clean **recurrent state** after the target value, any single GDN layer | — | ≤ **+0.01** |
| clean **recurrent state**, *all 18 GDN layers at once* | — | **+0.021 ± 0.002** |
| GDN **output @ the source value token** | **0** | **+1.043 ± 0.008** |
| GDN output @ the source value token | 14 | +0.057 ± 0.005 |
| GDN output @ the final token | 22 | +0.101 ± 0.003 |
| GDN output @ the following delimiter | any | ≈ 0 |

The headline negative result: **the GDN recurrent state carries essentially none of the
answer information.** Transplanting the clean state of *every* GDN layer, taken right after
the target value token, into the corrupted run recovers 2 % of the gap. Whatever moves the
value to the answer is not the recurrence.

The headline positive result is position-local: **GDN layer 0's output at the source value
token** is both sufficient (+1.04 clean→corrupted) and necessary (+1.04 in the reversed
direction, corrupted donor into the clean run). Controls (figure 1B) put this in context:
patching the raw token embedding at that position gives +1.066 and the whole layer-0 residual
gives +1.067, so GDN-0's own write accounts for ~98 % of a "re-insert the token" patch. This
is a genuine localised write (the residual stream at that position still holds the *corrupted*
embedding), but it is a shallow one — layer 0 is largely re-encoding token identity, not
computing anything the embedding did not already contain.

Candidates kept: **GDN-0 @ value** (primary), GDN-14 @ value and GDN-22 @ final (secondary).

## 4. Readers and the path (figure 2)

Writer patch = GDN-0 output at the source value token, recovery **+1.043**.

**Module screen** (figure 2A). Blocking each downstream module — freezing its output to the
plain-corrupted values — inside the writer-patched run. Only softmax layers **15** and **19**
matter (recovery 1.04 → 0.64 and → 0.36); every GDN layer and every MLP leaves it ≥ 0.9.
But the control matters: blocking the *same* modules in the clean run with no writer patch
gives 0.54 and 0.45. **Module blocking alone cannot separate mediation from general
importance here** — layers 15 and 19 are simply where retrieval happens.

**Path opening / blocking** (figure 2B–C) is decisive, because it requires no blocking at all:

| intervention (layer) | 15 | 19 | 3 / 7 / 11 |
|---|---|---|---|
| open **V @ source** (transplant writer-run V into the plain corrupted run) | **+0.423** | **+0.599** | ≤ 0.02 |
| open **K @ source** | +0.001 | +0.035 | ≤ 0.02 |
| open **Q @ final** | +0.007 | +0.057 | ≈ 0 |
| block **V @ source** (restore V to corrupted in the writer-patched run) | 1.04 → **0.647** | 1.04 → **0.387** | no change |
| block **K @ source** | 1.04 → 1.022 | 1.04 → 1.031 | no change |

Transplanting a *single* projection output at a *single* token position recovers 42 % / 60 %
of the gap. Blocking that same edge removes almost exactly the whole contribution of the
layer (layer 19: block-V 0.387 vs. block-the-entire-mixer 0.358).

**Heads** (figure 2D, `exp4b`). Resolving layer 15 and 19 into heads and repeating the
decomposition per head (donor run carries one patched input; then that head's slice of the
`o_proj` input is transplanted):

| head | V @ source | K @ source | Q @ final | Q+K+V all | attn. prob. final→source |
|---|---|---|---|---|---|
| **L15H5** | **+0.388** | −0.001 | +0.002 | +0.392 | 0.78 → 0.82 |
| **L19H1** | **+0.370** | +0.021 | +0.036 | +0.375 | 0.40 → 0.38 |
| L19H5 | +0.242 | +0.034 | +0.071 | +0.253 | 0.52 → 0.48 |
| L15H3 | −0.213 | −0.012 | −0.017 | −0.225 | 0.53 → 0.52 |
| L23H3 | −0.338 | −0.131 | −0.319 | −0.368 | 0.67 → 0.50 |

For L15H5 and L19H1 the V-only recovery equals the all-inputs recovery to within noise:
**the effect enters entirely through the value vector at the source position.** The attention
probability from the final token to the source is high and barely moves when the writer is
patched — the heads already point at the right token; the writer changes *what is stored
there*, not *whether it is attended to*. (Attention weights are reported as supplementary
context only; the patching results are the evidence.)

L15H3 and L23H3 carry the effect with the opposite sign — they promote the competing value.
L23H3 is the only head where Q@final matters (−0.319), so the last layer behaves differently
and was not pursued further.

## 5. Stability (figure 3)

Candidates were frozen after step 4 and re-tested on unseen examples only.

| condition | writer GDN-0 | state ref. | L15H5 V | L19H1 V | block V@15 | block V@19 |
|---|---|---|---|---|---|---|
| fresh, 8 pairs, **200 examples** | +1.047 | +0.018 | +0.395 | +0.372 | 0.662 | 0.385 |
| target_pos = 0 (farthest) | +1.048 | +0.035 | +0.314 | +0.347 | 0.732 | 0.401 |
| target_pos = 3 | +1.018 | +0.021 | +0.359 | +0.364 | 0.631 | 0.327 |
| target_pos = 7 (nearest) | +1.069 | +0.005 | +0.408 | +0.373 | 0.643 | 0.425 |
| 4 pairs | +1.100 | +0.039 | +0.372 | +0.365 | 0.724 | 0.400 |
| 8 pairs | +1.034 | +0.037 | +0.392 | +0.372 | 0.654 | 0.438 |
| 16 pairs | +0.987 | +0.025 | +0.356 | +0.358 | 0.600 | 0.404 |
| second template (`chat_alt`) | +1.046 | +0.017 | +0.407 | +0.417 | 0.674 | 0.377 |

Accuracy stays at 1.00 everywhere except 4 pairs (0.96). Every quantity moves by less than
±0.1 across all eight conditions: the same writer, the same two reader heads, and the same
V-at-source edge. The state reference stays at ≈ 0.02 throughout.

## Conclusion

**What the evidence supports.** There is a reproducible, narrowly localised pathway for
dictionary retrieval in this model, and it is a *positional* pathway, not a recurrent one.
GDN layer 0 writes the value token's content into the residual stream at the source position
(sufficient and necessary, +1.04 in both directions). Softmax heads **L15H5** and **L19H1**
(and more weakly L19H5) read it, and they read it **as the V payload of KV retrieval**:
transplanting the value vector at that one position recovers 37–42 % of the logit gap per
head, while K and Q transplants recover ~0. This holds on 200 fresh pairs and is unchanged by
target–query distance, distractor count (4/8/16) and a second template.

**What remains uncertain.** (i) The writer is shallow. GDN-0's write at the value position is
worth ~98 % of a raw token-embedding patch, so what it supplies is close to token identity;
this study does not show GDN-0 performing retrieval-specific computation. (ii) The *recurrent*
state is not used at all here (2 % for all 18 layers combined), so this is a GDN→softmax
pathway only in the sense of "GDN block output → attention V", not "GDN memory → attention".
Whether GDN state ever carries retrievable content may need a task the softmax layers cannot
solve positionally. (iii) The two reader heads together account for ~0.8 of the gap under
path opening, so part of the effect flows through routes not isolated here (L15H3/L23H3 carry
it negatively, and layer 23 behaves differently). (iv) Module-level blocking could not
separate mediation from general importance at layers 15/19; only edge-level patching could.
(v) Everything is one model, one task family, and 8–16-entry dictionaries in a 94-token window.

**Single most useful next experiment.** Force the value out of the positional route and see
whether the GDN state takes over: repeat the writer scan on a variant where the source value
token is *destroyed* after it is read — e.g. the dictionary is stated, then over-written or
paraphrased so the answer token no longer appears verbatim at any position, or the value must
be composed from two tokens. If the recurrent-state patch stays at 2 %, the GDN state really
is not a retrieval memory in this model; if it jumps, the positional shortcut was simply
cheaper and the GDN→softmax pathway is conditional on it being unavailable.

## Layout

```
src/task.py       synthetic dictionaries, clean/corrupt pairs, token-count verification
src/positions.py  token-position bookkeeping (asserts the two-position diff)
src/runner.py     model loading, split execution, cache handling, output/input patch hooks
src/qkv.py        Q/K/V projection patches and captures, attention-weight recording
src/harness.py    batching of examples that share token positions
src/metrics.py    logit difference and recovery
scripts/exp*.py   the five steps
scripts/plot_*.py the three figures
```

---

# Part II: what is the GDN recurrent state actually for?

The step-3 null result (all-layer state patch recovers 2 %) is ambiguous between three
worlds: (a) the binding is never written into the state, (b) it is written but decays, and
(c) it is present and readable but the softmax route answers first. Steps 6–8 separate them,
and connect the result to Borobia et al., *Component Ablation for Efficient Hybrid Language
Model Architectures* ([arXiv:2603.22473](https://arxiv.org/abs/2603.22473)), which ablates
whole components in this same model.

Figure: `figures/fig4_state_role.png`.

## 6. Route isolation, memory horizon, state divergence (`exp6_state_role.py`)

**Route isolation.** Cut one route and keep the other. `state route only` = run the *clean*
prompt but replace K and V over the entire dictionary span, in every softmax layer, with the
*corrupted* run's — the softmax layers can no longer see the true bindings, so only the GDN
state can supply them. `kv route only` is the mirror image.

| condition | recovery |
|---|---|
| neither route (corrupted prompt) | +0.000 (identity control, exact) |
| **state route only** | **+0.022 ± 0.002** |
| **KV route only** | **+0.977 ± 0.002** |
| both routes (clean prompt) | +1.000 (identity control, exact) |

The softmax KV over the dictionary span is essentially the entire mechanism. Both identity
controls are exact no-ops, so the isolation is valid.

**World (b) is ruled out.** Recomputing the forget gate `g_t = -exp(A_log)·softplus(a_t+dt_bias)`
on real prompts, the retention of a write made at the value token and read at the final token
is 0.15–0.45 on average and up to 1.00 for the best head in each layer. Median half-lives
range from ~1 token (layer 1) to ~45 tokens (layer 5), with p90 half-lives in the hundreds.
Nothing has decayed away over this 62-token window — forgetting is not the explanation.

**State divergence** `||S_clean − S_corr||/||S_clean||` is non-zero but small: 0.09–0.15 at
the write position for early layers, falling to 0.01–0.03 by the end of the dictionary. A
trace exists; whether it is a *binding* is the probe's job.

## 7. Is the binding decodable from the state at all? (`exp7_state_probe.py`)

A 20-way cross-validated logistic probe predicts **which value word the queried key is bound
to**, from the frozen recurrent state. Clean and corrupted prompts are separate samples with
different labels, so the probe cannot win by memorising examples. Floors: 0.05 uninformed,
**0.125 "presence"** (a probe that knows which 8 of 20 values appear but not the bindings
cannot beat 1/8). Ceiling check: the final-token residual stream decodes at **1.000**.

| cut point | GDN layers 0–14 | GDN layers 16–22 |
|---|---|---|
| **before the question is read** (end of dictionary) | 0.05–0.15 | 0.07–0.09 |
| at the final token | 0.03–0.45 | **1.000** |

Read left to right: **before the question is read, no GDN layer's state holds the binding** —
every layer sits at or below the presence floor. At the final token, decodability jumps to
1.000, and it jumps *exactly at layer 16*, i.e. immediately after softmax layer 15. The state
contains the answer only *after* attention has retrieved it and written it into the residual
stream. So this is world **(a)**, not (c): the state is not a silent-but-readable key–value
store. What looks like "the state knows the answer" in late layers is the state carrying
attention's output forward.

## 8. Functional memory horizon on general text (`exp8_horizon_ppl.py`)

If the state does not store bindings, what is it worth? Reset every GDN recurrent state to
zero every `w` tokens, leaving the convolution cache and the full softmax KV cache intact,
and measure WikiText-2 perplexity (16 × 512 tokens, float32).

| `w` (tokens of recurrent memory) | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | ∞ |
|---|---|---|---|---|---|---|---|---|---|---|
| perplexity | 324 | 217 | 129 | 75.7 | 49.3 | 36.0 | 29.0 | 25.2 | 23.0 | **21.9** |

Removing the recurrence entirely costs **15× perplexity**, and the curve is still improving at
w=256. The state is a genuinely valuable, broad, multi-scale contextual memory for next-token
prediction — it is just not an addressable store that a later head can query for a specific
binding.

**Calibration against Borobia et al.** (their intervention, reimplemented here):

| condition | this study (base 21.9) | Borobia et al. (base 7.6) |
|---|---|---|
| all GDN layers removed | 699,474 — **32,000×** | 268,337 — 35,200× |
| all softmax layers removed | 1,797 — **82×** | 625 — **82×** |
| 18 random layers removed | 1.66M (3 trials) | 1.75M (5 trials) |

The multipliers replicate closely (82× exactly for attention), so the two setups are measuring
the same thing on different corpus slices. Two further decompositions that their whole-layer
protocol cannot make:

| condition | perplexity |
|---|---|
| all GDN mixers zeroed, layer-local MLPs kept | 174,236 |
| all softmax mixers zeroed, MLPs kept | 93 |
| GDN layer 0 whole layer skipped | 35,426 |
| GDN layer 0 mixer only zeroed | 38,012 |

Layer 0's importance is its **token mixer**, not its MLP — which is the same component my
writer analysis localises, reached independently.

## What the GDN state does, and what it does not

- It is **not** a key→value memory that later softmax heads query. The binding is not in it
  (probe at the presence floor), and cutting the softmax route leaves only 2 % of the answer.
- It **is** a high-value distributed context summary for next-token likelihood, worth ~15×
  perplexity, accumulating over hundreds of tokens across heads with very different time
  constants (half-lives from ~1 to ~45 tokens).
- The apparent contradiction with Borobia et al. — "the linear pathway dominates likelihood"
  vs "the state is irrelevant to retrieval" — is not a contradiction. **Whole-layer ablation
  and recurrence ablation measure different things**, and the gap between them (699,474 vs 324)
  is the share of the GDN layer's value that is local rather than recurrent. Most of what a
  GDN layer contributes is a strong *positional* write; the recurrence adds a large but
  diffuse likelihood benefit and, in this model, no retrievable bindings.

## Methodological imports from Borobia et al.

1. **Matched random controls.** They separate "component identity" from "amount of computation
   disrupted" by removing the same number of random layers. My step-4 module screen had exactly
   this problem (blocking layers 15/19 hurt the clean control almost as much), which is why the
   edge-level path patching in figure 2B–D, not the module screen, carries the argument.
   Random controls are now included in `exp8`.
2. **Report task-level and likelihood-level metrics together.** Their clearest finding is that
   benchmark accuracy and perplexity dissociate. Part I used a single task metric; adding
   perplexity (step 8) is what turned a null result into a positive characterisation.
3. **Their logit-lens KL did not predict ablation impact** (r = −0.07, p = 0.74) — a warning
   that correlational internal diagnostics mislead, and the reason every claim here rests on an
   intervention with an identity control.
4. **Independent corroboration.** Their single-layer sweep finds **attention layer 15 the most
   important layer, followed by linear layer 0**, and their hidden-state diagnostics flag linear
   layer 0 with a norm change of 1.98, far above every other layer. Those are exactly the reader
   and the writer this study localises causally, found by a completely different method.

## Revised next experiment

Part I proposed destroying the positional route to see whether the state takes over. Step 6
already does the cheap version of that (state route only → +0.022) and step 7 explains why it
fails: nothing is written. So the better next experiment is upstream — **does the state ever
store bindings, under any training-time pressure?** Run the step-7 probe on (i) dictionaries
longer than the softmax layers' effective attention span, (ii) a streaming variant where each
pair is consumed and discarded, and (iii) the same probe on a *pure* linear-attention model
with no softmax layers to fall back on. If the probe stays at the presence floor even in (iii),
the delta-rule state in models of this scale is a context summariser rather than an associative
memory, and hybrid recall benefits come entirely from the softmax layers — a directly testable,
architecturally consequential claim.

---

# Part III: why is GDN layer 0 so important?

Three lines of evidence converge on layer 0: Borobia et al. report the largest hidden-state
norm change in the model (1.98, far above every other layer) and rank it second only to
attention layer 15 in single-layer importance; Part I finds its output at the source value
token both sufficient and necessary for retrieval (+1.04 in both directions); and removing it
alone costs ~1,600× perplexity. What is it computing?

Figure: `figures/fig5_layer0.png`. Script: `scripts/exp9_layer0.py`.

## Design: an information ladder

Ablation says "this matters"; it never says *what information* matters. So instead of removing
the layer, replace its mixer output `o_L(t)` with surrogates carrying progressively more
information, and see at which rung the damage disappears. Each rung corresponds to a hypothesis.

| rung | surrogate | information content | hypothesis it tests |
|---|---|---|---|
| 0 | zeros | none, no scale | — |
| 1 | global mean vector `μ` | scale + direction, **zero** information | H3 "scale-setter": a large constant write that sets the residual-stream scale downstream RMSNorms expect |
| 2 | norm-matched random direction | scale only | control for rung 1 |
| 3 | token-type mean `μ[x_t]` | current token identity **only**, no context | H1 "second embedding" |
| 3b | the layer's real output for that same token **in a different context** | token identity, no context, not an average | H1 without the averaging confound |
| 4 | intact | everything | — |

Reading the ladder: if rung 1 ≈ rung 4 the layer is a bias; if rung 3 ≈ rung 4 it is a
context-free lookup table; if only rung 4 works it genuinely computes something contextual.

Three supporting analyses: a **mechanism split** inside the mixer (conv window forced 4→1 tap;
recurrent state reset every token) to separate sequence mixing from the pointwise transform; a
**channel concentration** sweep (ablate only the top-k channels by mean |o|) to test the
"massive activations" explanation; and **matched controls** — the whole ladder repeated on GDN
layers 1, 2 and 4, so "layer 0 is special" is a contrast rather than an isolated number.

Statistics come from 200 WikiText-2 sequences; perplexity is measured on 16 **held-out**
sequences (token-type table covers 93.9 % of evaluation tokens, so rung 3 is a *lower* bound).

## Results (base PPL 17.46)

| surrogate | layer 0 | layer 1 | layer 2 | layer 4 |
|---|---|---|---|---|
| rung 0 zeros | **38,919** | 21.8 | 22.0 | 18.4 |
| rung 1 global mean | **14,729** | 22.2 | 18.6 | 18.1 |
| rung 2 norm-matched random | **24,268** | 31.0 | 27.8 | 19.2 |
| rung 3 token-type mean | **111.7** | 19.9 | 18.2 | 17.2 |
| rung 3b same token, other context | **113.1** | 22.0 | 19.4 | 17.3 |
| rung 4 intact | 17.46 | | | |

**Layer 0 is special by three orders of magnitude.** Zeroing any other early GDN mixer costs
under 30 % perplexity; zeroing layer 0 costs 2,200×.

**H3 (scale-setter) is rejected.** The global mean — the right scale, the right average
direction, zero information — gives 14,729, essentially as bad as a norm-matched random
direction (24,268) and the same order as zeros (38,919). Supplying the scale buys almost nothing.

**H1 (second embedding) is strongly supported.** Token identity alone takes perplexity from
38,919 to **111.7**, recovering **76 % of the nats** between zeros and intact
(10.57 → 4.72 → 2.86). Rung 3b (113.1) matches rung 3, so this is not an artefact of averaging
over contexts. The residual gap (111.7 vs 17.5) is genuine context sensitivity, but it is the
minority of the layer's value.

**Sequence mixing is nearly irrelevant.**

| intervention on layer 0 | perplexity |
|---|---|
| whole mixer output removed | 38,919 |
| conv window 4 → 1 tap | 28.5 |
| recurrence removed (state reset every token) | 21.4 |
| recurrence limited to 8 tokens | 18.7 |

Both of layer 0's sequence-mixing mechanisms can be destroyed for under 65 % perplexity cost,
against 2,200× for removing the output. Layer 0's value is overwhelmingly its **pointwise
channel transform of the current token**, not the conv and not the recurrence.

**Not a massive-activation phenomenon.** The top 8 channels carry 5.6 % of mean |o₀| and the
top 32 carry 10.4 % — a diffuse, not outlier-dominated, profile. Replacing the top-32 channels
with a constant costs only 21.7. The information lives across ~128+ channels: replacing the
top-128 with a constant gives 116.9, but replacing the same 128 with their *token-dependent*
means restores 19.1 — i.e. those channels carry token identity, not a bias.

**In the retrieval circuit, layer 0 is exactly a lookup table.** Replacing its output
everywhere with the context-free token-type mean (table built from held-out task prompts,
100 % coverage):

| layer 0 replaced by | retrieval accuracy | D_clean − D_corrupt |
|---|---|---|
| zeros | 0.000 | +0.16 |
| global mean (no token info) | 0.000 | +0.05 |
| **token-type mean (identity only)** | **1.000** | **+21.15** |
| **same token, other context** | **1.000** | **+20.35** |
| intact | 1.000 | +20.81 |

The Part I circuit is fully intact when layer 0 is reduced to a context-free token lookup.

> A methodological note: the first version of this measurement gave accuracy 0.000 for the
> token-type surrogate. That was an artefact — the table was built from WikiText and did not
> cover the chat-template tokens, so those positions silently fell back to the global mean,
> which is catastrophic. Coverage is now asserted and reported (1.000).

## Answer: what layer 0 is for

GDN layer 0 is a **learned second embedding** — a token-identity re-encoder that writes a large,
information-dense, diffuse vector into the residual stream at every position, which the rest of
the network treats as the real input representation. Its importance is not computation over
context: its conv and recurrence can both be destroyed almost for free. It is not a scale or
bias: a constant of the right magnitude does not substitute for it. It is the *content* — the
token-specific direction — that the network cannot do without.

This also dissolves the one loose end in Part I. GDN-0's write at the source value token was
sufficient and necessary for retrieval, and worth ~98 % of a token-embedding patch, which
looked "shallow but hard to interpret". It now has a precise reading: **layer 0 is where the
token's usable identity is created, so the softmax heads at layers 15/19 are retrieving
layer 0's output, not the embedding.** The GDN→softmax pathway is real, and its GDN half is a
per-position encoder rather than a memory — which is the same conclusion Part II reached from
the opposite direction about the recurrent state.

## Caveats and the next experiment

The 76 % figure is a lower bound (93.9 % table coverage) and is measured in nats between the
zeros and intact endpoints — a different normalisation would move it. "Second embedding" is a
functional description, not a claim that `o₀ = W·embed(x)`: rung 3 leaves a real 111.7 vs 17.5
gap, so ~24 % of the value is contextual. And this is one layer in one 0.8B model.

The sharpest next test is whether layer 0 is *literally* an affine re-embedding: fit
`o₀(t) ≈ A·embed(x_t) + b` by least squares on the token-type means and substitute the fitted
map at run time. If perplexity lands near rung 3, layer 0 is a rank-limited linear recode of
the embedding table and could be folded into it — a concrete compression claim, and a direct
answer to Borobia et al.'s question of which components must be preserved. If the fit is poor
while rung 3 is good, layer 0 implements a genuinely non-linear token-wise map, and the
interesting question becomes what that non-linearity buys.

---

# Part IV: is layer 0 literally an affine re-embedding?

Part III established that layer 0's mixer output is mostly a function of the current token.
That is a claim about *information*, not about the *form* of the map. Here the map is fitted
explicitly and substituted at run time:

```
o_0(t)  ≈  A · embed(x_t) + b
```

by weighted least squares over per-token-type means (equivalent to OLS over all occurrences,
since the regressor is constant within a type), fitted on WikiText-2 **train** (2,000 × 512
tokens, 23,947 token types) and evaluated on the **test** split. Ridge is chosen by R² on
token types held out of the fit — with too little data this fit is badly underdetermined
(1,936 types gave held-out R² 0.10; the reported fit uses 23,947).

Figure: `figures/fig6_affine.png`. Scripts: `scripts/exp10_affine.py`, `exp10b_affine_ood.py`.

## The map is affine

| quantity | value |
|---|---|
| R² of token identity (between-type / total variance) | **0.921** |
| R² of the affine map, of total variance | 0.863 |
| R² of the affine map, of the token-wise part | **0.937** |
| R² on token types **held out of the fit** | **0.844** |

And substituting it reaches the context-free ceiling on perplexity:

| layer 0 mixer replaced by | WikiText-2 PPL |
|---|---|
| intact | 21.86 |
| **affine `A·embed(x)+b`** | **90.80** |
| token-type lookup (the ceiling for *any* context-free map) | 92.82 |
| token-type lookup + affine fallback (coverage-matched) | 90.56 |
| global mean (no token information) | 14,942 |
| zeros | 38,919 |

The affine map **matches the token-type lookup ceiling** (90.8 vs 92.8, and 90.6 for the
coverage-matched version). Within the context-free part of layer 0, there is essentially
nothing that a linear function of the embedding misses. Layer 0's token-wise computation *is*
an affine recode of the embedding.

## …but it is full-rank, so the cheap compression does not follow

| rank of `A` | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| PPL | 12,754 | 8,579 | 3,809 | 1,753 | 636 | 242 | 119 | **90.8** |

The spectrum is flat (σ₆₄/σ₁ = 0.35, σ₂₅₆/σ₁ = 0.20) and truncation degrades smoothly all the
way down. So the hoped-for "fold layer 0 into the embedding table at rank r" does not work at
small r. The *full-rank* fold does work as an operation — `embed'(x) = embed(x) + A·embed(x) + b`
is precomputable per token, deleting layer 0's mixer entirely at zero inference cost — but it
costs 4.2× perplexity (21.9 → 90.8), because the ~8 % of variance that is genuinely contextual
turns out to be worth a lot. In nats: the affine substitution recovers 81 % of the distance
between zeros and intact, exactly like the token-type lookup, and the last 19 % is context.

## The retrieval collapse was extrapolation, not non-linearity

Substituting the WikiText-fitted map destroyed the dictionary task (accuracy 1.000 → **0.000**,
gap +20.81 → +2.48) even though an in-distribution token-type lookup was perfect (Part III).
Two candidate explanations — the task tokens are off the WikiText embedding manifold (OOD), or
layer 0 does something non-affine that retrieval needs (NONLIN). The residuals decide it:

| affine residual `‖o − ô‖/‖o‖` | all tokens | ordinary | chat-template tokens |
|---|---|---|---|
| WikiText eval, WikiText-fit `A` | 0.253 | 0.253 | — |
| task prompts, WikiText-fit `A` | 0.529 | 0.469 | **0.754** |
| task prompts, **combined-fit** `A` | 0.317 | 0.296 | 0.392 |

| substitution | WikiText PPL | retrieval accuracy | gap |
|---|---|---|---|
| intact | 21.86 | 1.000 | +20.81 |
| affine fitted on WikiText only | 90.80 | **0.000** | +2.48 |
| affine fitted on WikiText **+ task prompts** | 90.76 | **1.000** | **+21.90** |

Adding task text to the fit restores retrieval completely, at no cost on WikiText (90.76 vs
90.80). **NONLIN is rejected.** The map is affine; the WikiText-only fit simply extrapolated
badly to chat-template tokens, whose embeddings sit far off the natural-text manifold (residual
0.754 vs 0.253). This is a property of the *fit's support*, not of layer 0.

> This is the second time in this study that an apparent null came from an unrepresentative
> surrogate table rather than from the model (the first was in Part III). Both were caught by
> asserting coverage and reporting residuals per token group — worth doing by default.

## Conclusion

GDN layer 0 is an **affine re-embedding**: `o_0 ≈ A·embed(x) + b` with held-out-type R² 0.84,
reaching the context-free ceiling on perplexity and, once the fit covers the token distribution,
leaving the Part I retrieval circuit perfectly intact. Its conv and its recurrence are nearly
irrelevant (Part III); its value is a full-rank linear recode of the token embedding that the
rest of the network treats as the real input representation.

Two consequences. For interpretability: the "writer" in the Part I circuit is now fully
characterised — layers 15/19 retrieve `A·embed(value) + b`, a linear image of the embedding,
which is why patching it was worth 98 % of a token-embedding patch. For compression, the answer
to Borobia et al.'s question is a qualified no: layer 0's mixer *is* mathematically foldable
into the embedding table, but not cheaply (full rank required) and not losslessly (4.2×
perplexity), so it should be preserved — consistent with their finding that early components
tolerate compression worst.

**Remaining uncertainty and next step.** The 19 % contextual residual is now the whole story and
it is unexamined: it is small in variance but expensive in nats, which usually means it is
concentrated on a minority of positions. The next experiment is to find them — compute the
per-token NLL increase under the affine substitution and characterise the worst positions
(sentence-initial? rare tokens? multi-token words, i.e. the detokenisation hypothesis Part III's
conv ablation only tested in aggregate?). If the damage concentrates on word-continuation
tokens, layer 0's small contextual part is doing detokenisation after all, and the clean
statement becomes "an affine re-embedding plus a narrow detokeniser".

---

# Part V: per-token attribution — and a correction to Part III

**NLL** (negative log-likelihood) is `−log p(x_{t+1} | x_{≤t})` in nats: the per-position
quantity whose average, exponentiated, is perplexity. Because it is defined per position,
`Δ(t) = NLL_ablated(t) − NLL_intact(t)` says exactly which tokens an intervention hurts.
Part IV ended by predicting that the affine substitution's residual would be *concentrated* on
a minority of positions — probably word continuations, i.e. detokenisation. It is not, and
chasing that prediction turned up an error in Part III.

Figure: `figures/fig7_per_token.png`. Script: `scripts/exp11_per_token.py`
(32 held-out WikiText-2 sequences, 16,352 predicted positions, intact NLL 2.894 / PPL 18.07).

## The key new condition: layer 0 made *token-local*

Part III ablated layer 0's two sequence-mixing mechanisms **one at a time** — conv window
4→1 tap (PPL 33.6) and recurrent state reset every token (22.1) — and concluded that "sequence
mixing is nearly irrelevant". Doing **both at once** gives a layer 0 that recomputes its real
function with no access to any other token:

| layer 0 condition | PPL | ΔNLL (nats) |
|---|---|---|
| intact | 18.07 | — |
| no recurrence only | 22.12 | +0.202 |
| conv 1 tap only | 33.63 | +0.621 |
| **token-local (conv 1 tap *and* no recurrence)** | **57.25** | **+1.153** |
| token-type mean lookup | 78.03 | +1.463 |
| affine `A·embed(x)+b` | 79.59 | +1.483 |

**Correction to Part III.** The conv and the recurrence are *redundant* sources of context:
removing either alone is cheap, removing both costs 3.2× perplexity. "Sequence mixing is nearly
irrelevant" was an artefact of testing them separately. Layer 0 does use context, worth
1.15 nats.

**Correction to Parts III–IV.** I called the token-type mean "the ceiling for any context-free
map of the token". That is wrong. It is the ceiling for a *lookup table estimated by averaging
observed outputs*; a context-free **recomputation** does much better (57.3 vs 78.0). The real
decomposition of the affine substitution's 1.483 nats is therefore:

| component | nats | share |
|---|---|---|
| genuine loss of context (intact → token-local) | 1.153 | **78 %** |
| surrogate approximation error (token-local → affine) | 0.330 | 22 % |

So the affine map's shortfall is mostly *not* approximation error — it is that any context-free
stand-in loses something real. Against the full span from zeroed output (ΔNLL ≈ +7.5) to intact,
a token-local recomputation still recovers ~85 % of the nats and the affine map ~80 %, so
Part IV's headline — layer 0 is dominantly a token-wise affine recode — stands; what changes is
that its remaining context dependence is larger and more diffuse than I claimed.

## Detokenisation: rejected

| target category | share of tokens | intact NLL | Δ token-local | Δ affine | share of affine damage |
|---|---|---|---|---|---|
| word-initial | 56.7 % | 3.818 | +1.604 | +1.908 | 72.9 % |
| **word-continuation** | **7.1 %** | 1.325 | +1.130 | +1.738 | **8.3 %** |
| punctuation | 23.6 % | 2.032 | +0.481 | +0.820 | 13.0 % |
| digit | 3.7 % | 1.642 | +0.363 | +0.682 | 1.7 % |
| whitespace/newline | 4.5 % | 1.619 | +0.591 | +0.915 | 2.8 % |

Word continuations take 8.3 % of the damage while being 7.1 % of positions — proportional, not
concentrated — and per token they are hurt *less* than word-initial tokens. The direct probe is
even clearer: `conv 1 tap`, the intervention that should break detokenisation if anything does,
hurts continuations (+0.674) **less** than word-initials (+0.838). Layer 0's convolution is not
a detokeniser.

## The damage is broad, not focal

| condition | top 1 % of positions | top 5 % | top 10 % | positions hurt at all |
|---|---|---|---|---|
| affine | 7.0 % | 25.9 % | 43.2 % | 79.1 % |
| token-local | 7.9 % | 29.3 % | 47.9 % | 78.6 % |
| no recurrence (a genuinely focal ablation, for contrast) | 18.2 % | 55.5 % | 85.1 % | 65.1 % |

79 % of all positions get worse, and it takes the worst 10 % to account for 43 % of the damage.
For contrast, removing the recurrence alone — a small, genuinely localised effect — puts 85 % of
its damage in its worst 10 %. So my Part IV prediction ("small in variance but expensive in nats
usually means concentrated") was wrong: it is a broad, diffuse degradation.

What it does scale with is **how hard the position already was** (Δ_affine rises monotonically
from +0.26 in the easiest decile to +2.25 in the 9th) and **how rare the target is** (+0.83 for
targets seen >1000 times in the fit corpus, +2.87 for those seen 1–10 times).

Two further diagnostics confirm there are two distinct failure modes rather than one:

- affine vs token-type mean: **r = +0.956** — they damage the same positions. The affine *form*
  adds essentially nothing beyond being a context-free estimate.
- affine vs token-local: **r = +0.494** — different positions. Losing context and mis-estimating
  the token-wise map hurt different tokens.
- Position in sequence separates them cleanly: token-local damage is ~0 at positions 1–8
  (+0.19, there is no context to lose yet) and grows to +1.24 by position 128+; affine damage is
  *largest* early (+3.35 at positions 1–8) where entropy is high and the surrogate is worst.

## Revised picture of layer 0

Layer 0 is an affine re-embedding **plus a broad, redundant, context-smoothing term** — not a
detokeniser. Roughly 80–85 % of its value is a token-wise affine recode of the embedding
(Part IV, unchanged); the remaining 15–20 % is diffuse context sensitivity, supplied redundantly
by the 4-tap convolution and the recurrence, and spread across four fifths of all positions in
proportion to their difficulty rather than concentrated on any identifiable linguistic class.

**Next experiment.** The redundancy is the one crisp, unexplained fact left: conv-only and
recurrence-only each cost ~0.2–0.6 nats, but jointly 1.15 — they carry overlapping information.
Measure the overlap directly by sweeping the conv window (1/2/3/4 taps) crossed with the
recurrence reset window (1/2/4/8/∞) as a 2-D grid. If the surface is roughly additive in
`log(window)`, the two are supplying the same short-range signal by two routes, and the layer
could be simplified to one of them; if there is a sharp interaction, they are specialised and
the redundancy is only apparent.

---

# Part VI: layer 0 vs layer 1 — why the first GDN layer is different

Layers 0 and 1 are architecturally identical GDN blocks, three positions apart in the same
residual stream. Zeroing layer 0's mixer costs **+7.46 nats** (PPL 21.9 → 38,013); zeroing
layer 1's costs **+0.24** (→ 27.8). A 31× difference in nats. Five candidate explanations, each
with its own test.

Figure: `figures/fig8_layer01.png`. Scripts: `scripts/exp12_layer01.py`, `exp12b_matched_noise.py`.

## G — Geometry: layer 0's write dominates the residual stream ✔

| layer | ‖o_L‖ | ‖h_in‖ | ratio |
|---|---|---|---|
| **0** | 1.26 | 0.66 | **1.95** |
| 1 | 0.79 | 1.23 | 0.67 |
| 2 | 0.94 | 1.56 | 0.62 |
| 4–22 | 0.55–2.05 | 1.85–9.37 | 0.22–0.43 |

Layer 0 writes a vector twice the size of the stream it reads — after it, the residual stream
*is* mostly its output (0.66 + 1.26 → 1.23 entering layer 1). Every later layer contributes a
0.2–0.7 fraction. This reproduces Borobia et al.'s norm-change outlier (1.98) and is the single
largest factor.

## N — Content: mostly rejected, with an important residue ✔/✘

Step 12's first pass perturbed each mixer by `ε‖o_L‖u`, which is confounded — the same ε
injects a bigger absolute perturbation at layer 0 precisely because of G. Matching properly:

| perturbation | δ or ε | layer 0 | layer 1 | layer 2 | layer 4 |
|---|---|---|---|---|---|
| **absolute** `o+δu` | 0.4 | +0.034 ± .008 | +0.028 ± .008 | +0.024 ± .005 | +0.012 ± .004 |
| **absolute** | 1.6 | +1.99 ± .13 | +0.70 ± .04 | +0.44 ± .02 | +0.26 ± .02 |
| **relative to the stream** `o+ε‖h_in‖u` | 0.4 | **+0.015** ± .005 | +0.045 ± .008 | +0.054 ± .007 | +0.041 ± .006 |

At matched *absolute* noise, layers 0 and 1 are statistically indistinguishable up to δ = 0.4;
measured *relative to the residual stream*, layer 0 is the **least** sensitive layer of the four.
So layer 0's output is not intrinsically more fragile — the naive reading of the first pass was
wrong.

But deletion is not noise. Comparing deletion against random noise of exactly the same magnitude
gives a magnitude-controlled measure of how much the *specific direction* matters:

| structure index = ΔNLL(delete) / ΔNLL(random noise of equal size) | layer 0 | layer 1 | layer 2 | layer 4 |
|---|---|---|---|---|
| | **9.4×** | 2.0× | 1.8× | 2.1× |

For every other early layer, deleting the output is only ~2× worse than scrambling it by the same
amount; for layer 0 it is 9.4× worse. So there *is* a content effect — roughly 4.7× — it is just
much smaller than the raw 31×, and it sits on top of the geometry factor rather than replacing it.
(Single noise seed; treat the index as an order-of-magnitude statistic.)

## R — Redundancy: this is why layer 1 looks cheap ✔

| mixers zeroed | ΔNLL | matched random set of the same size |
|---|---|---|
| [1] | +0.240 | — |
| [2] | +0.223 | — |
| **[1, 2]** | **+1.435** | +0.237 (mean of 3) |
| [1, 2, 4] | +2.228 | +0.206 |
| [1, 2, 4, 5] | +2.281 | +0.518 |
| [1, 2, 4, 5, 6] | +2.306 | — |

Layers 1 and 2 are individually cheap and jointly expensive: removing both costs **6× either
alone** and **6× a random pair**. They substitute for each other. The effect saturates by
[1,2,4] — the early GDN block as a whole is worth ~2.3 nats, but no single member of it is.
Nothing substitutes for layer 0: [0,1] (+7.37) is no worse than [0] alone (+7.46).

This is the same redundancy pattern found in Part V *within* layer 0 (conv and recurrence each
cheap, jointly expensive), now appearing *between* layers. Single-layer ablation systematically
understates early-layer importance — a caution that applies directly to layer-sweep studies.

## S — Specialisation: the two layers do different jobs ✔

| | positions hurt at all | top 1 % | top 5 % | top 10 % of positions |
|---|---|---|---|---|
| zero layer 0 | **97.7 %** | 2.3 % | 10.1 % | 18.4 % of damage |
| zero layer 1 | 61.6 % | 17.5 % | 53.9 % | **82.0 %** of damage |

Layer 0's damage is almost perfectly uniform — it hurts essentially every token, and the worst
10 % of positions carry only 18 % of it (uniform would be 10 %). Layer 1's damage is focal: 82 %
of it lands on 10 % of positions. And the two are nearly independent: **r(Δ_L0, Δ_L1) = +0.13**.
They help different tokens.

## I — Information: the same ladder, ~30× smaller ✔

| surrogate for the mixer output | layer 0 ΔNLL | layer 1 ΔNLL |
|---|---|---|
| zeros | +7.461 | +0.240 |
| global mean (no token info) | +6.534 | +0.224 |
| token-type mean | +1.888 | +0.121 |
| token-local recomputation | +1.230 | +0.177 |
| conv 1 tap only | +0.632 | +0.153 |
| no recurrence only | +0.207 | +0.073 |

Layer 0: token identity alone recovers 75 % of the deletion damage (7.461 → 1.888) — the Part IV
result. Layer 1: 50 % (0.240 → 0.121), and its token-local recomputation is *worse* than a
token-type lookup, i.e. what little layer 1 contributes is relatively more contextual. But all
layer-1 effects are 0.07–0.24 nats, close enough to each other that the ordering within its
ladder should not be over-read.

## Answer

**Layer 0 is a token encoder; layer 1 is a redundant specialist.** The 31× gap decomposes into
three multiplicative factors, none of which alone explains it:

1. **Magnitude (largest).** Layer 0's write is 1.95× the stream it reads, versus 0.67× for
   layer 1, and the network's response to perturbation is strongly superlinear. Deleting layer 0
   leaves downstream layers reading the bare embedding.
2. **Structure (~4.7×).** Layer 0's particular direction matters 9.4× more than a random
   direction of the same size, against ~2× for every other early layer — it writes the token's
   usable identity (Part IV: an affine re-embedding), which nothing else reconstructs.
3. **Non-redundancy.** Layers 1 and 2 cover for each other (either alone +0.23, both +1.44);
   layer 0 has no substitute.

And they are functionally disjoint: layer 0 affects 98 % of positions almost uniformly, layer 1
affects a focal 10 %, and their damage profiles correlate at r = 0.13.

**Caveat.** This is one model, one corpus, and the structure index rests on a single noise seed.
The redundancy result also means the ladder numbers for layer 1 measure its *marginal* value
given layer 2, not its function.

**Next experiment.** Redundancy is now the recurring theme — within layer 0 (conv vs recurrence)
and between layers 1 and 2. The direct test is a pairwise joint-ablation matrix over all 18 GDN
layers: zero every pair, and compare ΔNLL(i,j) against ΔNLL(i) + ΔNLL(j). The supra-additive
entries map which layers substitute for each other, turning "early layers matter most" into a
statement about which specific groups form a redundant block — and that is what a compression
schedule actually needs.

---

# Part VII: GDN layer groups, and what survives with only layer 0

Two questions, both on the same two metrics — WikiText-2 perplexity and the Part I dictionary
task — because Part VI showed they can dissociate. Softmax layers and all MLPs are left intact
throughout; ablation zeroes a GDN layer's **token mixer** only.

The 18 GDN layers fall into exactly 6 blocks of 3, each sitting between two softmax layers:

```
G0 [0,1,2] | attn 3 | G1 [4,5,6] | attn 7 | G2 [8,9,10] | attn 11
G3 [12,13,14] | attn 15 | G4 [16,17,18] | attn 19 | G5 [20,21,22] | attn 23
```

Figure: `figures/fig9_groups.png`. Script: `scripts/exp13_groups.py`. Intact: PPL 21.86,
retrieval accuracy 1.000, answer gap +20.81.

## 1. Keep only the first GDN layer

All 6 softmax layers and all 24 MLPs intact; GDN mixers 1–22 zeroed:

| condition | PPL | retrieval acc | gap |
|---|---|---|---|
| intact | 21.86 | 1.000 | +20.81 |
| **keep only GDN layer 0** | **3,479** | **0.030** | +0.09 |
| keep only GDN layer 1 | 270,333 | 0.000 | +0.16 |
| keep only GDN layer 2 | 271,693 | 0.000 | +0.32 |
| keep only GDN layer 8 | 173,280 | 0.000 | −0.09 |
| keep only GDN layer 16 | 221,598 | 0.000 | −0.12 |
| keep only GDN layer 20 | 155,246 | 0.000 | +0.06 |
| all 18 GDN mixers zeroed | 174,236 | 0.000 | −0.03 |
| keep only G0 = layers [0,1,2] | 558 | 0.000 | +0.04 |
| *(stricter: other GDN layers skipped entirely, MLPs too)* | 62,744 | 0.000 | −0.04 |

**The answer is no — it does not survive.** Perplexity goes 21.9 → 3,479 (159×) and retrieval
collapses to chance. But the comparison is informative in two ways:

- **Layer 0 alone is worth more than the other 17 combined.** Keeping only layer 0 gives 3,479;
  keeping *everything except* layer 0 gives 38,013 (Part VI). Layer 0 on its own is 11× better
  than all its 17 siblings together.
- **It is ~50× better than keeping any other single layer** (155k–272k), all of which are no
  better than zeroing every GDN mixer (174,236) — i.e. a single mid-network GDN layer running on
  a residual stream that never received layer 0's write contributes essentially nothing.

So layer 0 is necessary and by far the most valuable single layer, but nowhere near sufficient:
the ~2.3 nats of distributed value in the rest of the GDN stack (Part VI) cannot be dispensed
with either.

## 2. Ablating each group of three

| group | layers | PPL | ΔNLL | retrieval acc | gap |
|---|---|---|---|---|---|
| **G0** | [0,1,2] | **26,019** | +7.082 | **0.000** | +0.08 |
| G1 | [4,5,6] | 44.13 | +0.703 | 0.840 | +7.50 |
| G2 | [8,9,10] | 24.84 | +0.128 | 0.600 | +5.93 |
| **G3** | [12,13,14] | 25.04 | +0.136 | **0.150** | +3.45 |
| G4 | [16,17,18] | 30.38 | +0.329 | 0.965 | +16.18 |
| G5 | [20,21,22] | 32.50 | +0.397 | 0.940 | +14.16 |
| *3 random GDN layers* | — | 28.3 | +0.249 ± .091 | — | — |

**On perplexity there is exactly one special group.** G0 costs 7.08 nats; every other group
costs 0.13–0.70, and **G2 and G3 are below the matched random triple** — removing three
specific mid-network GDN layers is *less* damaging than removing three random ones. The early
positional gradient reported by Borobia et al. is really a layer-0 effect plus a mild G1 effect,
not a smooth gradient.

**On the retrieval task the ordering is completely different.** G3 [12,13,14] is the second most
destructive group (accuracy 1.000 → 0.150) while being one of the *cheapest* on perplexity
(+0.136 nats, below random). G4 and G5 — the two most expensive groups after G0/G1 on perplexity
— are nearly free for retrieval (0.965, 0.940).

This is mechanistically coherent with Part I. G3 is the block immediately **before softmax
layer 15**, whose head L15H5 was the primary reader; G3 builds the representation that reader
attends to. It also resolves an apparent tension: Part I's module screen found that blocking
layers 12, 13 or 14 *individually* left the writer patch's recovery at ≥0.997. Removing all
three together breaks the task. The same redundancy pattern as Parts V and VI — within layer 0
(conv vs recurrence), between layers 1 and 2, and now within G3.

A second redundancy shows in the cumulative sweep: removing G4 alone leaves accuracy at 0.965
and removing G5 alone at 0.940, but removing **both** drops it to 0.295 — consistent with
Part II's finding that the GDN state from layer 16 onward carries attention-15's retrieved
answer forward to layer 19 and the readout.

## 3. Depth gradient, and a caution

| GDN layers kept | 1 | 2 | 3 | 4 | 6 | 9 | 12 | 15 | 18 |
|---|---|---|---|---|---|---|---|---|---|
| PPL | 3,479 | 1,241 | 558 | 360 | 165 | 98.9 | 59.6 | 32.5 | 21.86 |
| retrieval acc | 0.030 | 0.000 | 0.000 | 0.010 | 0.070 | 0.075 | 0.295 | 0.940 | 1.000 |

Dropping the last three GDN layers costs 49 % perplexity and 6 points of accuracy; dropping the
first three costs 1,190×. Value is heavily front-loaded, but note that retrieval needs 15 of the
18 layers to reach 0.94 — the task is not supported by the early layers alone.

**Caution — deeper ablation is not monotonically worse.** Removing groups from the front:
G0 → 26,019, G0–G1 → 437,948, G0–G2 → 689,705, G0–G3 → **775,812**, G0–G4 → 531,802,
G0–G5 → 174,236. Removing *all eighteen* is 4.5× better than removing the first twelve. Once the
early layers are gone the remaining ones inject actively harmful signal into a residual stream
they were never trained to see. Any ablation study that reads "more removed = worse" as a
monotone importance ranking will misorder these conditions.

## Summary

1. **Keeping only GDN layer 0 does not work** (PPL 3,479, retrieval at chance) — but layer 0
   alone beats the other 17 layers combined by 11×, and beats any other single layer by ~50×.
2. **Only G0 matters for perplexity**; G2 and G3 are *below* matched random triples.
3. **Retrieval and perplexity rank the groups differently.** G3 is near-free for likelihood and
   near-fatal for retrieval, because it feeds the reader head at softmax layer 15 — a concrete
   instance of Borobia et al.'s warning that benchmarks and perplexity measure different things.
4. **Redundancy within groups is pervasive**: every group whose removal matters is made of
   layers that are individually cheap to remove.

**Caveats.** One model, one corpus, one task; mixer-only ablation (whole-layer skip gives larger
numbers throughout); retrieval accuracy is measured on 200 prompts of a single template.

**Next experiment.** The G3-vs-perplexity dissociation is the most useful lead: a group that is
free to compress by the usual metric and fatal for retrieval. Run the group sweep against a
recall-oriented benchmark suite (needle-in-a-haystack / RULER-style, plus a few of the Borobia
benchmarks) to see whether "cheap on perplexity, critical for retrieval" holds for G3 generally
or only for this dictionary template. That is the direct test of whether perplexity-guided
compression would silently destroy a hybrid model's recall — which is exactly the property the
softmax layers are there to provide.

---

# Part VIII: the roles of [G1+G2] and [G4+G5]

G0 is critical for perplexity, G3 for retrieval. The two remaining blocks sit either side of
the reader head at softmax layer 15:

```
G1+G2 = [4,5,6,8,9,10]        BEFORE the reader
G3    = [12,13,14]            immediately before the reader
G4+G5 = [16,17,18,20,21,22]   AFTER the reader
```

Figure: `figures/fig10_supergroups.png`. Script: `scripts/exp14_supergroups.py`.

## Ablation alone does not separate them

| removed | PPL | ΔNLL | retrieval acc | gap |
|---|---|---|---|---|
| intact | 21.86 | — | 1.000 | +20.81 |
| **G1+G2** | 51.79 | +0.863 | **0.035** | +1.86 |
| **G4+G5** | 59.62 | +1.003 | **0.295** | +4.97 |
| G2+G3 | 32.88 | +0.408 | 0.000 | +1.17 |
| G3+G4 | 40.59 | +0.619 | 0.085 | +2.00 |
| G0+G1 | 437,948 | +9.905 | 0.000 | +0.03 |
| G0+G3 | 49,459 | +7.724 | 0.000 | −0.03 |
| *6 random GDN layers* | ~58 | +0.977 ± .582 | — | — |

On perplexity the two super-groups are **indistinguishable from each other and from six random
GDN layers** (+0.86 and +1.00 against a random-6 mean of +0.98). Perplexity says nothing about
their roles. Retrieval separates them — G1+G2 removal is far worse — but says only "more" and
"less", not *what*. Three probes settle it.

## Probe 1 — logit lens: where the answer is formed

`D = logit(clean answer) − logit(corrupt answer)`, read off the residual stream at the final
position after each layer (softmax layers in bold):

| condition | L11 | L14 | **L15** | L18 | **L19** | L22 | **L23** |
|---|---|---|---|---|---|---|---|
| intact | +0.2 | +0.5 | **+14.6** | +10.3 | **+27.8** | +24.5 | **+20.8** |
| remove G0 | +0.0 | −0.1 | −0.0 | −0.0 | +0.0 | +0.0 | +0.1 |
| **remove G1+G2** | +0.0 | −0.0 | **+1.8** | +1.1 | **+2.5** | +2.7 | **+1.9** |
| remove G3 | +0.2 | +0.2 | **+0.6** | +0.7 | **+2.8** | +5.4 | **+3.4** |
| **remove G4+G5** | +0.2 | +0.5 | **+14.6** | +13.6 | **+28.5** | +23.1 | **+5.0** |

Two completely different failure modes:

- **Without G1+G2 the answer is never retrieved.** The jump at layer 15 — the moment the reader
  head writes the value into the residual stream — drops from +14.6 to +1.8, and nothing later
  recovers it.
- **Without G4+G5 retrieval works perfectly and the answer is then lost.** The trajectory is
  *identical* to intact through layer 15 (+14.6) and layer 19 (+28.5, marginally above intact),
  and then collapses across the last block, +23.1 → **+5.0** at the output.

## Probe 2 — reader attention: can the head still find the source?

P(final token → source value token):

| condition | L15H5 | L19H1 |
|---|---|---|
| intact | 0.782 | 0.404 |
| remove G0 | 0.007 | 0.009 |
| **remove G1+G2** | **0.187** | 0.085 |
| remove G3 | **0.056** | 0.113 |
| **remove G4+G5** | **0.782** | 0.340 |

G1+G2 and G3 destroy the reader's aim; G4+G5 leave it *exactly* untouched. So G1+G2 and G3 act
on the **query/address side**, and G4+G5 act entirely **downstream of retrieval**.

## Probe 3 — is the block's recurrence local or long-range?

ΔNLL when the recurrent state is reset every `w` tokens *inside that block only*:

| block | w=1 | w=8 | w=64 | w=64 / w=1 |
|---|---|---|---|---|
| G0 | +1.155 | +0.364 | +0.097 | 0.08 |
| **G1+G2** | +0.329 | +0.128 | **+0.007** | **0.02** |
| G3 | +0.077 | +0.037 | +0.010 | 0.13 |
| **G4+G5** | +0.460 | +0.389 | **+0.213** | **0.46** |

G1+G2's recurrence is the most local in the model — an 8-token window already recovers most of
it and by 64 tokens its value is gone (0.007 nats). G4+G5's is the most long-range by far:
still 0.21 nats at w=64, i.e. 46 % of its w=1 damage remains. These two blocks are doing
opposite things with the recurrence.

## Roles

**G1+G2 — building the address.** Local, short-horizon processing that turns layer 0's token
encodings into the query and key representations the softmax reader matches on. Removing them
leaves the reader unable to find the source token (attention 0.78 → 0.19), so the answer is
never retrieved at all (logit lens flat through the whole network). Its recurrence is
short-range only; it is doing local feature construction, not memory. G3 is the final and
sharpest stage of this same pipeline: it costs even less perplexity (+0.136, below random) and
damages the reader's aim more (attention → 0.056).

**G4+G5 — carrying the retrieved answer to the output.** Removing them leaves retrieval fully
intact through layer 19 and then loses the answer in the last block. Reader attention is
unchanged. Its recurrence is the longest-range in the model, consistent with a transport and
consolidation role rather than a computation one. This matches Part II: a probe on the GDN
recurrent state first decodes the answer at layer 16 — the first layer of G4 — i.e. exactly
where this block picks the answer up from attention-15 and starts carrying it.

The collapse is specifically at layer 23, which is also where Part I found the anti-reader head
L23H3 (contribution −0.34). Without G4+G5 maintaining the retrieved value, that competing
last-layer signal dominates.

So the GDN stack around the reader is a **pipeline**: `G0` encodes tokens → `G1+G2`, `G3` build
the address → softmax 15/19 retrieve → `G4+G5` transport the result to the output.

**Caveats.** "Address construction" is inferred from the attention drop plus the logit-lens
flatline, not from a direct read of the query representation; a Q/K path-patch through G1+G2
would confirm it. The retrieval half of this rests on one task template, and the L23 collapse
is a single-layer observation on 200 prompts.

**Next experiment.** The pipeline predicts a clean double dissociation that has not been tested
directly: patching **only the reader's query** at layer 15 from an intact run into a
G1+G2-ablated run should restore retrieval, while the same patch in a G4+G5-ablated run should
do nothing (there the query is already fine). Running both patches in both ablated models is a
2×2 that would confirm the role assignment causally rather than by elimination.

---

# Part IX: resolving G3 to a layer, and layers 0 and 14 to heads

Scripts: `scripts/exp15_heads.py` (log `logs/exp15.log`, results `results/exp15_heads.json`)
and `scripts/exp15b_head8.py` (results `results/exp15b_head8.json`). Same 200 retrieval prompts
as Parts VII–VIII; perplexity on 12 × 512-token WikiText-2 test chunks (intact PPL 22.36).
"attn" is L15H5's attention from the final token to the source value in the corrupted prompt.

A GDN layer has 16 value heads of width 128. Head *h* owns slice `[128h, 128(h+1))` of the
`out_proj` input (after the gated RMSNorm), so zeroing that slice removes exactly that head's
write to the residual stream.

## 1. Which layer of G3 matters?

| removed | PPL | retrieval acc | gap | L15H5 attn |
|---|---|---|---|---|
| nothing | 22.36 | 1.000 | +20.81 | 0.782 |
| [12] | 23.30 | 1.000 | +20.36 | 0.789 |
| [13] | 22.53 | 0.995 | +15.44 | 0.675 |
| **[14]** | 23.76 | **0.870** | **+12.10** | **0.354** |
| [12, 13] — *only 14 kept* | 23.70 | **1.000** | +14.91 | 0.633 |
| [12, 14] — only 13 kept | 25.01 | 0.945 | +11.94 | 0.336 |
| [13, 14] — only 12 kept | 24.07 | 0.220 | +3.48 | 0.081 |
| [12, 13, 14] | 25.63 | 0.150 | +3.45 | 0.056 |

**Layer 14 is the critical layer of G3.** It is the only single layer whose removal clearly hurts
(and halves the reader's attention on the source), and keeping it alone restores accuracy to
1.000. Layer 13 is a partial backup: the task only collapses once 13 and 14 are both gone. Layer
12 adds nothing measurable. As for G3 as a whole, none of this costs much perplexity (≤ +0.14 nats).

## 2. Per-head ablation in layers 0 and 14

Zero one head, everything else intact:

| layer 0 head | PPL | ΔNLL | acc | gap | | layer 14 head | PPL | ΔNLL | acc | gap |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 34.42 | +0.432 | 0.995 | +18.26 | | 0 | 22.38 | +0.001 | 0.985 | +14.37 |
| 4 | 29.41 | +0.274 | 1.000 | +21.97 | | 15 | 22.68 | +0.015 | 0.965 | +19.66 |
| **8** | **65.96** | **+1.082** | **0.000** | **+2.36** | | 14 | 22.84 | +0.021 | 1.000 | +19.91 |
| other 13 heads | ≤ 23.6 | ≤ +0.054 | 1.000 | +19.5 to +22.4 | | other 13 heads | ≤ 22.6 | ≤ +0.011 | 1.000 | +18.6 to +21.3 |

**Layer 0 has a single point of failure: head 8.** Zeroing it alone takes retrieval from 1.000
to 0.000 and triples perplexity. It is also the most important head for perplexity, but heads 3
and 4 show the two metrics dissociate: they cost 0.43 / 0.27 nats yet leave retrieval intact
(and zeroing 3+4 together still leaves 0.52 accuracy, see §3).

**Layer 14 has none.** No head's removal costs more than 0.035 accuracy or 0.02 nats; head 0
dents the gap most (+14.4).

Keep one head, zero the other 15 (Part C of the script):

- **Layer 0:** every single-head model fails (acc 0.000, PPL 4,119–36,663). Head 8 alone gives
  the lowest PPL (4,119), still 180× intact. Head 8 is necessary but far from sufficient.
- **Layer 14:** every single-head model gives acc 0.84–0.96 and gap +10.3 to +14.8 — i.e. the
  same as removing layer 14 entirely (0.870, +12.10). **No single head carries layer 14's
  function**; it only appears when the heads act together. (The script's "best single head =
  15, acc 0.960" is within this band and should not be read as a rescue.)

## 3. What the heads look like

Measured on 200 × 512 WikiText tokens: **|o_h|** = mean norm of the head's output slice;
**half-life** = tokens for the head's mean forget gate to halve its state (`ln 0.5 / mean log-decay`,
capped at 10⁶); **R² token-id** = fraction of the head's output variance explained by token
identity alone (between-type / total).

| | |o_h| | half-life | R² token-id |
|---|---|---|---|
| **layer 0, head 8** | **2.80** (largest in layer) | **≥ 10⁶** (no measurable decay) | **0.996** |
| layer 0, head 3 | 2.01 | 0.21 | 0.993 |
| layer 0, head 4 | 0.48 | 3.9 | 0.585 |
| layer 0, other heads | 0.13–2.02 | 0.2–6,900 | 0.57–0.996 |
| layer 14, all heads | 0.24–1.90 | 3–1,900 | **0.10–0.49** |

Head 8 writes the largest vector in layer 0, almost entirely determined by the current token
(R² 0.996), and its gate essentially never forgets. Layer 0 is token-identity dominated overall
(most heads R² > 0.9), which is Part III's "learned second embedding" resolved to heads. Layer 14's
heads are not token encoders (R² ≤ 0.49): whatever they write depends on context.

## 4. Is head 8 the Part I writer?

**Head-resolved writer patch** (clean → corrupted, GDN-0 output at the source value token only;
all 16 heads together recover +1.043, as in Part I):

| head | 3 | 8 | 2 | 15 | 7 | 5 | 13 | 1 | others |
|---|---|---|---|---|---|---|---|---|---|
| recovery | **+0.453** | +0.191 | +0.030 | +0.024 | +0.021 | +0.020 | +0.019 | +0.016 | ≤ 0.01 |

The source-value write that Part I found is mostly **head 3**, with head 8 second. The per-head
recoveries sum to ~0.80 of the whole-layer 1.04, so heads also interact.

**Where head 8 is needed** (zero head 8 at selected positions only):

| head 8 zeroed at | acc | gap | L15H5 attn |
|---|---|---|---|
| nowhere | 1.000 | +20.81 | 0.782 |
| **every position** | **0.000** | +2.36 | 0.096 |
| source value token only | 1.000 | +19.25 | 0.766 |
| all 8 value tokens | 1.000 | +19.40 | 0.791 |
| query key (in the question) | 1.000 | +17.81 | 0.762 |
| final token | 1.000 | +20.68 | 0.782 |
| **every position except the value tokens** | **0.050** | +3.91 | 0.149 |
| *control: heads 3+4 everywhere* | 0.520 | +9.71 | 0.364 |

So head 8 is **not** the writer in the Part I sense: removing it at the value tokens costs
nothing, and it carries only 0.19 of the source-value patch. It is needed at the *other*
positions, and no single one of the probed positions (query key, final token) is enough to
break the task — the damage comes from removing it across the non-value context as a whole.
Without it, the reader loses its aim (attention 0.78 → 0.10–0.15), the same symptom as removing
G3 (Part VIII probe 2).

## Reading

Two opposite architectures inside the same stack:

- **Layer 0 is concentrated.** One head (8) — large, non-decaying, almost purely token-identity —
  is indispensable, because the rest of the network uses its write at the context positions
  (keys, question, template) to locate the answer. A second head (3) supplies most of the
  value-token content the reader copies. Heads 3/4 carry much of the perplexity cost without
  being required for retrieval.
- **Layer 14 is distributed.** Its effect needs the layer as a whole, no head is necessary or
  sufficient, and its heads encode context rather than token identity. Part X localises what it
  feeds: mostly the reader's query at the final token.

**Caveats.** "Head 8 is needed at non-value positions" is established by exclusion; which of them
(dictionary keys, `=`/`,` delimiters, question, template tokens) matter has not been narrowed. The
half-life for head 8 hits the 10⁶ cap, so it means "no decay measurable", not a precise value.
All retrieval numbers are one template, 200 prompts, 8 pairs.

---

# Part X: does G3 build the reader's query or its keys?

Script: `scripts/exp16_qk_patch.py`, log `logs/exp16.log`, results `results/exp16_qk_patch.json`.
Same 200 prompts. Patches act on layer 15's `q_proj`/`k_proj`/`v_proj` outputs (before
q/k-norm and RoPE). Q is patched at the final token, K/V at all positions unless stated.
`lens@L15` is the clean−corrupt gap read off the residual stream right after layer 15, i.e. the
reader's own output, before downstream layers can add their own damage.

## Rescue: ablate, then restore one side of the dot product from the intact run

| ablation → | L14 attn | L14 lens@15 | L13+14 attn | L13+14 lens@15 | G3 attn | G3 lens@15 | G3 acc |
|---|---|---|---|---|---|---|---|
| intact | 0.782 | +14.6 | 0.782 | +14.6 | 0.782 | +14.6 | 1.000 |
| ablated, no patch | 0.354 | +5.0 | 0.081 | +1.2 | 0.056 | +0.6 | 0.150 |
| **Q@final** | **0.534** | **+10.3** | **0.420** | **+10.4** | **0.425** | **+11.0** | **0.680** |
| K@all | 0.388 | +9.4 | 0.122 | +2.1 | 0.084 | +0.9 | 0.380 |
| K@value tokens | 0.329 | +7.3 | 0.127 | +2.2 | 0.089 | +0.9 | 0.250 |
| K@key tokens | 0.368 | +5.3 | 0.074 | +1.1 | 0.050 | +0.5 | 0.165 |
| V@all | 0.354 | +4.8 | 0.081 | +1.1 | 0.056 | +0.6 | 0.235 |
| Q@final + K@all | 0.782 | +17.2 | 0.782 | +20.6 | 0.782 | +20.8 | 0.900 |
| Q+K+V@all at L15 and L19 | 0.782 | +16.5 | 0.782 | +19.1 | 0.782 | +19.2 | 0.995 |

## Necessity: intact run, insert the ablated run's Q or K

| inserted from → | L14 attn | L13+14 attn | G3 attn | G3 lens@15 |
|---|---|---|---|---|
| (nothing) | 0.782 | 0.782 | 0.782 | +14.6 |
| **ablated Q@final** | **0.388** | **0.122** | **0.084** | **+0.8** |
| ablated K@all | 0.534 | 0.420 | 0.425 | +5.9 |
| ablated K@value tokens | 0.598 | 0.516 | 0.484 | +7.0 |

The two tables check each other: "rescue Q" and "insert ablated K" are the same configuration
(intact Q, ablated K) and give the same attention (0.425 / 0.425 for G3).

## Reading

1. **G3 mainly builds the query.** Restoring only the final-token query recovers reader attention
   from 0.056 to 0.425 and the reader's output from +0.6 to +11.0 (G3). Restoring the keys alone
   does almost nothing (0.084). Conversely, giving an intact model G3-less queries is almost as
   destructive as removing G3 (attention 0.084, lens +0.8), whereas G3-less keys only halve it.
   One reader head is enough to see it: patching the query of **H5 alone** gives the same
   rescue as all eight heads.
2. **The keys contribute too, and they interact with the query.** Q alone restores about half
   the attention; Q+K restores it exactly (0.782, as it must, since attention depends only on
   Q and K). The key-side contribution sits mostly at the **value tokens** (ablated K at value
   tokens alone costs 0.30 of the 0.36 that ablated K@all costs), not at the key-word tokens.
3. **V does nothing** (rescue ≤ 0.04). Consistent with Part I: G3 does not carry value content.
4. **G3 also feeds reader(s) after layer 15.** With Q+K+V fully restored at layer 15 the reader's
   output is even above intact (+19.2 vs +14.6), yet the final gap stays at +8.7. Also restoring
   layer 19's Q/K/V brings accuracy to 0.995 but the gap only to +13.0 (intact +20.8). G3's
   effect is therefore spread across the second reader L19 and later layers, not only L15.

**Revised picture.** G3 — and layer 14 within it — writes the "what am I asking for?" feature at
the final token that the reader's query uses to find the matching entry, with a smaller
contribution to the value-token keys it matches against. When the query is missing, the reader
falls back to position, which is why the worked example attends to the first entries.

**Caveats.** Patches are made at projection outputs, so "the query" here includes whatever
q_norm and RoPE do to it downstream. The Q-vs-K asymmetry is clear under the G3 and L13+14
ablations but smaller for L14 alone (Q rescue 0.53 vs K 0.39). The residual downstream deficit
is not localised beyond "L19 and later".

---

# Part XI: G2 builds the keys, G1 and G3 the query

Same script as Part X with other ablation sets:
`scripts/exp16_qk_patch.py --abl "L8=8;L9=9;L10=10;G2=8,9,10;G1=4,5,6;G1+G2=4,5,6,8,9,10"`
(log `logs/exp16b.log`, results `results/exp16b_qk_patch_g2.json`).

## Layers of G2

| removed | acc | gap | lens@L15 | L15H5 attn |
|---|---|---|---|---|
| nothing | 1.000 | +20.81 | +14.6 | 0.782 |
| [8] | 1.000 | +18.68 | +15.4 | 0.752 |
| [9] | 1.000 | +17.00 | +15.4 | 0.702 |
| **[10]** | **0.935** | **+10.87** | **+8.6** | **0.533** |
| G2 [8, 9, 10] | 0.600 | +5.93 | +4.3 | 0.284 |

The same pattern as G3: one layer (10, the last before softmax 11) carries most of the effect and
the other two back it up. G2 costs +0.128 nats of perplexity (Part VII), below a random triple.

## Query or keys? Attention of L15H5 on the source value

| | G1 | G2 | L10 | G1+G2 | *G3 (Part X)* |
|---|---|---|---|---|---|
| ablated, no patch | 0.640 | 0.284 | 0.533 | 0.187 | *0.056* |
| rescue **Q@final** | **0.725** | 0.291 | 0.492 | **0.404** | ***0.425*** |
| rescue **K@all** | 0.599 | **0.555** | **0.749** | **0.386** | *0.084* |
| rescue K@value tokens | 0.616 | **0.562** | **0.753** | 0.446 | *0.089* |
| rescue K@key tokens | 0.641 | 0.283 | 0.535 | 0.164 | *0.050* |
| necessity: insert ablated Q | 0.599 | 0.555 | 0.749 | 0.386 | *0.084* |
| necessity: insert ablated K@all | 0.725 | **0.291** | **0.492** | 0.404 | *0.425* |
| necessity: insert ablated K@value tokens | 0.473 | **0.244** | **0.436** | 0.080 | *0.484* |

(As in Part X, "rescue Q" and "insert ablated K" are the same configuration and agree.)

- **G2 acts on the keys, not the query** — the mirror image of G3. Restoring the keys recovers
  attention 0.28 → 0.56; restoring the query does nothing (0.29). Giving an intact model the
  G2-less keys *at the value tokens only* is as damaging as removing G2 outright (0.244 vs 0.284).
  Layer 10 alone shows the same thing more cleanly (K rescue 0.53 → 0.75, Q rescue none).
- **The relevant keys are at the value tokens**, not the key words: patching K at the key-word
  tokens does nothing for any ablation. The reader attends to `gray`, so it is `gray`'s key that
  must say "I am grape's value".
- **G1 acts mainly on the query** (Q rescue 0.64 → 0.73, K rescue none), like G3.
- **G1+G2 together damage both sides**, which is why Part VIII saw only "the reader loses its
  aim". Neither side alone gets attention past ~0.45; both together restore it exactly.
- As for G3, restoring all of layer 15's inputs does not restore the final output (G2: acc 0.88,
  gap +10.0); also restoring layer 19 gives acc 1.000, gap +14.6. G2 feeds the L19 reader too.

## Reading

The address stage splits by side of the dot product:

```
G1 [4,5,6]    → query  (what is being asked for, at the final token)
G2 [8,9,10]   → keys   (at each value token: which key this value belongs to), mostly layer 10
G3 [12,13,14] → query  (final and sharpest query stage), mostly layer 14
```

A plausible mechanism for G2 is **key→value binding**: copying the identity of the key word
(`grape`, two tokens back across `=`) into the value token (`gray`) so the value's key vector
carries it. That fits Part VIII probe 3 (G1+G2's recurrence is the most local in the model; an
8-token window suffices), but it is an inference — no probe has yet read the key identity out of
the value-token residual with and without G2.

**Caveats.** The G1 necessity row for K@value tokens (0.473) is lower than for K@all (0.725),
i.e. G1's effect on keys at value tokens is partly cancelled by its effect on keys elsewhere —
G1 is not purely a query stage. One template, 200 prompts.

---

# Part XII: G2 does not create the key→value binding — it makes keys distinguishable

Scripts: `scripts/exp17_binding_probe.py` (log `logs/exp17.log`, `results/exp17_binding_probe.json`)
and `scripts/exp17b_attn_targets.py` (log `logs/exp17b.log`, `results/exp17b_attn_targets.json`).

Part XI proposed that G2 copies each key word's identity into its value token. Test: a linear
probe (22-way logistic regression, chance 0.045) decodes the **own key** of every value token.
Keys and values are sampled independently, so the value token's own identity says nothing
about its key. 4,800 train / 1,600 test value tokens from fresh random dictionaries. "fixed" =
probe trained on the intact model applied to the ablated one; "retrain" = trained and tested
on the ablated model. K15 = layer 15's `k_proj` output at the value token.

## Own key at the value token

| condition | mode | resid 3 | resid 7 | resid 9 | resid 10 | resid 11 | resid 14 | K15 |
|---|---|---|---|---|---|---|---|---|
| intact | — | **0.996** | 0.997 | 0.998 | 1.000 | 0.999 | 0.999 | 1.000 |
| remove G2 | fixed | 0.996 | 0.997 | 0.436 | 0.315 | 0.319 | **0.995** | **0.990** |
| remove G2 | retrain | 0.996 | 0.997 | 0.996 | 0.994 | 0.991 | 1.000 | 1.000 |
| remove L10 | fixed | 0.996 | 0.997 | 0.998 | 0.499 | 0.589 | 0.999 | 1.000 |
| remove G1 | fixed | 0.996 | 0.072 | 0.134 | 0.700 | 0.338 | 0.987 | 0.991 |
| remove G3 | fixed | 0.996 | 0.997 | 0.998 | 1.000 | 0.999 | 0.615 | 0.552 |
| remove G1+G2 | fixed | 0.996 | 0.072 | 0.061 | 0.061 | 0.057 | 0.869 | 0.764 |
| remove G1+G2 | retrain | 0.996 | 0.941 | 0.909 | 0.894 | 0.835 | 1.000 | 1.000 |

The value token's own identity decodes at 1.000 everywhere (control). The previous entry's key
also decodes, less well (intact 0.38 at layer 3, 0.93 at K15).

**The binding hypothesis is rejected.** The key's identity is already at the value token after
layer 3 — before G1 or G2 — so layers 0–2 (whose conv spans the two tokens back to the key)
and/or softmax 3 put it there. Removing G2 does not remove it: a retrained probe still reads it
at 0.99–1.00 at every layer, and the intact model's own probe still reads it at 0.99 **in the
reader's key vector (K15)**. The drop of the fixed probe at layers 9–11 is expected — those
residuals are directly missing G2's write — and it recovers by layer 14.

## Where the reader's attention goes instead

L15H5, final token, clean prompts, share of attention:

| condition | target value | other value tokens* | key words | query key | `<|im_start|>` | rest |
|---|---|---|---|---|---|---|
| intact | **0.794** | 0.058 | 0.017 | 0.000 | 0.063 | 0.069 |
| remove G1 | 0.649 | **0.249** | 0.016 | 0.005 | 0.000 | 0.081 |
| **remove G2** | 0.298 | **0.404** | 0.073 | 0.010 | 0.054 | 0.161 |
| remove L10 | 0.553 | **0.311** | 0.028 | 0.001 | 0.038 | 0.068 |
| **remove L14** | 0.351 | 0.034 | 0.068 | 0.041 | **0.376** | 0.129 |
| remove G3 | 0.048 | 0.411 (0.220 on the first entry) | 0.040 | 0.004 | **0.216** | 0.281 |
| remove G1+G2 | 0.201 | 0.331 | 0.101 | 0.111 | 0.000 | 0.257 |

\* previous + next + first + other entries' values.

Two distinct failure modes:

- **Without G2 (or layer 10) the reader still finds the value tokens but cannot tell them
  apart.** Attention leaks to the *other* values (0.06 → 0.40), spread over neighbours and
  distant entries alike; the BOS sink is unchanged. The keys still look like "value tokens", and
  still contain their key's identity, but no longer single out the right one.
- **Without layer 14 the query matches nothing.** Attention falls back to the `<|im_start|>`
  sink (0.06 → 0.38), not to other values. Removing all of G3 mixes both: sink (0.22) plus the
  first entry (0.22) — the "reads the first entries" behaviour of the worked example.

## Reading

G2's contribution to the keys is not *what* key a value belongs to — that is present from
layer 3 — but making that identity **discriminable along the direction the reader's query
reads**. Linear decodability by a 22-way probe is a much weaker requirement than a single
q·k dot product separating one entry from seven others, and G2 is what bridges the two.

```
G0 (+ softmax 3)  token identity; each value token already carries its key's identity
G1                query, and some key sharpening (attention leaks to other values without it)
G2 (layer 10)     keys: make the key identity at value tokens selective for the query
G3 (layer 14)     query: without it the query matches nothing (attention sink)
```

**Caveats.** "Discriminable along the query direction" is the remaining explanation, not a
measurement. The direct test is the reader's pre-softmax score for each value token, with an
intact query, against intact vs G2-less keys (selectivity of q·k), and the geometry of those
keys (e.g. how similar the keys of different entries become). Probes show information is
present, not that the model uses it. Same template, 100 clean prompts for the attention breakdown.

---

# Part XIII: what G1 [4,5,6] does

Scripts: `scripts/exp18_g1.py` (log `logs/exp18.log`, `results/exp18_g1.json`) and
`scripts/exp18b_g1_cancel.py` (log `logs/exp18b.log`, `results/exp18b_g1_cancel.json`).
Retrieval columns as before; attention = L15H5 from the final token on clean prompts, split
into target value / other value tokens / `<|im_start|>` sink.

## 1. General text: G1 is for prediction, not copying

WikiText-2, 40 × 512 tokens. An *induction* target is a token that completes a bigram already
seen earlier in the sequence (28.7 % of targets; intact NLL 0.74 vs 3.80 on the rest).

| removed | ΔNLL induction | ΔNLL other | ratio |
|---|---|---|---|
| G0 | +8.689 | +6.929 | 1.25 |
| **G1** | **+0.069** | **+0.982** | **0.07** |
| G2 | +0.174 | +0.149 | 1.17 |
| G3 | +0.236 | +0.168 | 1.40 |
| G4 | +0.111 | +0.419 | 0.26 |
| G5 | +0.338 | +0.401 | 0.84 |

G1 is the most **prediction-specific** group in the model. It is the largest non-G0 cost on
ordinary tokens (+0.98 nats) and nearly free on in-context copying (+0.07). G2 and G3 are the
opposite: slightly more important for copying than for ordinary tokens — consistent with their
roles as the key and query stages of retrieval.

## 2. Retrieval: small, distributed, and on the dictionary

| removed | acc | gap | lens@L15 | attn target | other values | sink |
|---|---|---|---|---|---|---|
| nothing | 1.000 | +20.81 | +14.55 | 0.794 | 0.057 | 0.063 |
| [4] | 1.000 | +20.45 | +14.34 | 0.782 | 0.080 | 0.042 |
| [5] | 1.000 | +17.61 | +13.96 | 0.735 | 0.101 | 0.077 |
| [6] | 1.000 | +18.13 | +16.35 | 0.897 | 0.052 | 0.000 |
| [5, 6] | 0.985 | +11.72 | +13.29 | 0.788 | 0.151 | 0.000 |
| [4, 5, 6] | 0.840 | +7.50 | +9.32 | 0.649 | 0.249 | 0.000 |

No single layer matters much; the effect builds up over all three (unlike G2 → layer 10 and
G3 → layer 14).

Zeroing G1's write only at some positions:

| G1 zeroed at | acc | gap | lens@L15 | attn target | other values | sink |
|---|---|---|---|---|---|---|
| keys / `=` / values / `,` (each alone) | ≥ 0.990 | +17.8 to +20.0 | ≥ 13.3 | ≥ 0.76 | ≤ 0.07 | ≈ 0.06 |
| **whole dictionary** | **0.930** | **+9.29** | **+10.06** | 0.650 | **0.217** | 0.037 |
| query key (question) | 1.000 | +19.84 | +14.62 | 0.778 | 0.068 | 0.070 |
| everything after the dictionary | 1.000 | +15.45 | +12.70 | 0.701 | 0.093 | 0.110 |
| final token | 1.000 | +20.86 | +13.12 | 0.732 | 0.061 | 0.072 |
| everything except the dictionary | 1.000 | +17.35 | +15.15 | 0.880 | 0.046 | **0.000** |

G1's retrieval-relevant write is over **the dictionary as a whole** — removing it there
reproduces the full-removal pattern (attention leaks to other values, 0.22 vs 0.25) — and no
single token type (keys, `=`, values, delimiters) carries it. Its write at the question and
final token hardly matters. So Part XI's "G1 acts on the query" is indirect: the query at the
final token is affected because the dictionary context it is built from is.

**Side finding: layer 6 makes the attention sink.** Whenever layer 6's write at the
non-dictionary positions (which include `<|im_start|>`) is removed, L15H5's sink attention goes
from 0.06 to 0.000 — and target attention *rises* (0.88–0.90).

## 3. G1 works through what comes after it

Remove G1, then restore downstream components to their intact outputs:

| restored | acc | gap | lens@L15 | attn target |
|---|---|---|---|---|
| none (G1 removed) | 0.840 | +7.50 | +9.32 | 0.649 |
| **softmax 7 + 11** | **0.990** | **+13.04** | **+13.97** | 0.663 |
| G2 mixers | 0.515 | +3.26 | +4.56 | 0.436 |
| G3 mixers | 0.800 | +11.23 | +6.98 | 0.344 |
| all mixers 7–14 | 0.790 | +11.00 | +8.26 | 0.378 |
| MLPs 4–14 | **0.000** | +0.38 | +0.06 | 0.008 (sink 0.71) |
| mixers 7–14 + MLPs 4–14 (G1's direct write is all that is missing) | 0.770 | +10.95 | +6.35 | 0.223 |

Converse — keep G1's own write, but give downstream components their G1-less outputs: acc
0.73–0.81, gap +8.1 to +8.7, i.e. **the same as removing G1**. G1's own write is worth nothing
for retrieval unless downstream layers have processed it.

Two things follow.

- **The only clean rescue is through softmax layers 7 and 11** (acc 0.99, reader output +14.0 vs
  +14.6 intact). G1's contribution to retrieval runs mostly through the two softmax layers that
  sit inside and right after it.
- **Restoring the GDN mixers or MLPs alone makes things worse**, and restoring MLPs is fatal. The
  reason is measured in 18b (intact run, WikiText, medians over positions):

| part of the MLP output that exists because of G1 | cos with G1's write | size relative to G1's write | |G1 write + that part| / |G1 write| |
|---|---|---|---|
| MLPs 4–6 | **−0.51** | 0.79 | 0.90 |
| MLPs 7–14 | −0.14 | 1.44 | 1.65 |
| MLPs 4–14 | −0.38 | 1.58 | 1.52 |

The MLPs right after G1 respond to its write with a vector of similar size that points
**against** it (cos −0.51), and the total MLP response is ~1.6× the write itself. G1's write and
the MLP computation are one coupled unit: put back only the MLP response and a large
anti-aligned vector is left with nothing to cancel.

## Reading

G1 is **not a retrieval component**. It is the network's main mid-depth **language-modelling
block**: the largest non-G0 contributor to predicting ordinary tokens, nearly irrelevant to
copying, spread over three layers, and tightly coupled to the MLPs that follow it. Retrieval
uses it only indirectly: G1 refines the representation of every dictionary token, softmax 7 and
11 use that to move context across the dictionary, and G2 (keys) and G3 (query) build on the
result. Without it, the reader still aims at values but confuses them (the same failure as G2,
milder).

```
G0 (+ softmax 3)   token identity; each value already carries its key's identity
G1 (+ softmax 7)   general contextual features over the whole context (language modelling)
G2 (layer 10)      keys at the value tokens made selective for the query
G3 (layer 14)      the query itself
softmax 15 / 19    retrieval
G4 + G5            carry the answer to the output
```

**Caveats.** The mediation results are non-additive (restoring more components can do worse
than restoring fewer), so "through softmax 7 and 11" is the best-supported route, not a clean
decomposition; what softmax 7/11 compute from G1 has not been examined. The induction split
uses exact bigram repetition, a crude definition of copying.

---

# Part XIV: where the key→value binding is made

Script: `scripts/exp19_binding_origin.py` (log `logs/exp19.log`, `results/exp19_binding_origin.json`).
Same probe as Part XII (own key of each value token, 22 classes; chance acc 0.045, CE 3.09),
plus test cross-entropy, which still separates conditions at ceiling accuracy.

## A. Where it appears (intact model)

| site at the value token | own key acc | CE | prev key acc |
|---|---|---|---|
| embedding | 0.043 | 3.13 | 0.053 |
| residual after layer 0 | 0.983 | 0.56 | 0.264 |
| residual after layer 1 | 1.000 | 0.18 | 0.344 |
| residual after layer 2 | 1.000 | 0.08 | 0.490 |
| residual after softmax 3 | 0.996 | 0.11 | 0.381 |
| layer 0 mixer output alone | 0.996 | 0.38 | 0.329 |
| layer 1 mixer output alone | 1.000 | 0.03 | 0.476 |
| **layer 2 mixer output alone** | **1.000** | **0.01** | **0.741** |
| softmax 3 output alone | 0.988 | 0.09 | 0.348 |

The binding is **first written by layer 0** and then **rewritten, more sharply, by layers 1 and
2** — each GDN mixer's own output at the value token encodes the key, layer 2's most cleanly
(and with the widest reach: it also carries the previous entry's key at 0.74). Softmax 3 writes
it too.

## B. What is necessary (probe after layer 3, retrained on the intervened model)

| intervention | own key acc | CE |
|---|---|---|
| intact | 0.996 | 0.11 |
| layer 0 mixer zeroed at value tokens | 0.879 | 0.64 |
| layer 1 mixer zeroed at value tokens | 0.981 | 0.23 |
| layer 2 mixer zeroed at value tokens | 0.943 | 0.33 |
| softmax 3 zeroed at value tokens | 0.999 | 0.13 |
| layers 1+2 zeroed at value tokens | 0.726 | 0.98 |
| layer 0 token-local (conv 1 tap + no recurrence) | **1.000** | 0.05 |
| layer 1 token-local | 0.988 | 0.16 |
| layer 2 token-local | 0.932 | 0.36 |
| layers 0–2 **conv 1 tap** | 0.888 | **0.91** |
| layers 0–2 **no recurrence** | 0.980 | 0.19 |
| layers 0–2 token-local | 0.576 | 1.80 |

- **No single layer is necessary.** Any one layer's contribution can be removed and the key is
  still read at ≥ 0.93. Layer 0's *own* context mixing is entirely dispensable (token-local
  layer 0: 1.000) — layers 1–2 rebuild the binding. (Zeroing layer 0 at the value tokens hurts
  more, 0.879, but that also removes the value's own token encoding.)
- **Layers 1+2 together carry most of it** (0.726 without their value-token writes), with layer 2
  the larger single contributor (CE 0.33–0.36 vs 0.15–0.23 for layer 1).
- **The mechanism is the convolution, not the recurrence.** Reducing the conv of layers 0–2 to
  one tap costs 5× more (CE 0.91) than removing their recurrence (0.19). The key is two tokens
  back, inside the 4-tap window.
- **Softmax 3 is a backup.** Unnecessary when G0 is intact (0.999), it is the likely source of
  what remains (0.576) when all of layers 0–2 are token-local.

## C. Retrieval under the same interventions

| intervention | acc | gap |
|---|---|---|
| intact | 1.000 | +20.81 |
| layer 0 conv 1 tap | 0.855 | +11.71 |
| layer 0 no recurrence | 1.000 | +18.75 |
| layer 0 token-local | **0.170** | +6.01 |
| layer 1 conv 1 tap / no recurrence / token-local | 1.000 | +20.8 to +21.5 |
| layer 2 no recurrence | 0.995 | +15.88 |
| layer 2 token-local | 0.965 | +13.90 |
| layers 0–2 conv 1 tap | 0.000 | +0.21 |
| layers 0–2 no recurrence | 0.150 | +3.38 |
| layers 0–2 token-local | 0.000 | +0.58 |

Retrieval and binding **dissociate**: token-local layer 0 leaves the binding perfect (1.000) but
retrieval collapses (0.170); layers 0–2 conv 1 tap leave it largely intact (0.888) but retrieval
is 0.000. Retrieval therefore needs G0's sequence mixing for something besides the value-token
binding. One candidate is an artefact: Part III found a context-free *token-type mean*
surrogate for layer 0 kept retrieval at 1.000, so the collapse under a context-free
*recomputation* may reflect off-distribution outputs of the modified conv rather than a genuine
need for context. This was not resolved here.

## Reading

The key→value binding is a **redundant, convolution-driven product of G0**: layer 0 writes it
first, layers 1 and 2 (mainly 2) rewrite it more sharply from their own 4-tap convs, and softmax
3 can partly substitute. No single layer owns it. This completes the address picture: G0 makes
the binding, G2 makes it selectively readable in the keys, G1/G3 build the query.

**Caveats.** Probes show presence, not use. Zeroing a mixer at the value tokens removes all of its
write there, not only the binding. The retrieval collapses in C are not explained.

---

# Part XV: q·k selectivity — G2 shapes both sides of the match (correction to Parts XI–XII)

Script: `scripts/exp20_qk_selectivity.py` (log `logs/exp20.log`, `results/exp20_qk_selectivity.json`).

Parts XI–XII concluded that G2 acts on the reader's **keys**, and inferred that it makes them
selective along the query's read direction. That rested on attention *mass*, which also depends
on competition with the `<|im_start|>` sink. Here the reader's actual scores are measured.

**Method.** In the intact model, L15's Q at the final token and K at all positions are
overwritten with the same prompt's Q/K from the intact or ablated run (2 × 2). L15H5's attention
log-probabilities from the final token give exact pre-softmax score differences (after
q/k-norm, RoPE, scaling). Over the 8 value tokens, 100 clean prompts:
- **selectivity** = s_target − mean(s_other values), and **margin** = s_target − max(s_other values)
  — both reference-free;
- **top-1** = target scores highest among the values;
- tgt−sink / oth−sink = scores relative to `<|im_start|>` (note: the sink's key is also replaced
  in K patches, so these two move together with any change at the sink).

| ablation | Q from | K from | selectivity | margin (±SEM) | top-1 | tgt−sink | oth−sink |
|---|---|---|---|---|---|---|---|
| — | intact | intact | **5.57** | +3.54 ± 0.10 | 1.000 | +2.68 | −2.90 |
| G2 | intact | **G2-less** | 3.28 | +1.91 ± 0.07 | 1.000 | −0.24 | −3.52 |
| G2 | **G2-less** | intact | 2.95 | +1.61 ± 0.08 | 0.980 | +3.71 | **+0.76** |
| G2 | G2-less | G2-less | 1.85 | +0.80 ± 0.08 | 0.840 | +1.69 | −0.17 |
| L10 | intact | L10-less | 3.87 | +2.44 ± 0.07 | 1.000 | +0.91 | −2.96 |
| L10 | L10-less | intact | 4.24 | +2.63 ± 0.10 | 1.000 | +3.90 | −0.33 |
| L10 | L10-less | L10-less | 2.96 | +1.67 ± 0.10 | 0.960 | +2.78 | −0.18 |
| G3 | intact | G3-less | 7.44 | +4.29 ± 0.16 | 0.990 | +0.83 | −6.60 |
| G3 | **G3-less** | intact | **0.19** | **−1.39** ± 0.11 | **0.110** | −0.62 | −0.81 |
| G3 | G3-less | G3-less | 0.22 | −2.89 ± 0.20 | 0.090 | −2.63 | −2.85 |

Key geometry (k_proj output of H5's KV head at the 8 value tokens):

| keys from | mean pairwise cosine between entries | entry-specific share of key energy |
|---|---|---|
| intact | 0.806 | 0.172 |
| G2-less | 0.843 | 0.139 (−19 %) |
| L10-less | 0.849 | 0.134 (−22 %) |
| G3-less | 0.721 | 0.245 |

## Reading

1. **G2 contributes to both sides, about equally.** Removing G2 from the keys only cuts
   selectivity 5.57 → 3.28; from the query only, 5.57 → 2.95; from both, 1.85 (margin 0.80,
   top-1 0.84). Layer 10 alone shows the same split. Part XI's "keys, not query" was wrong in
   selectivity terms.
2. **The two sides fail differently.**
   - **G2-less keys:** the target still wins among the values (top-1 1.000), and the other
     values barely move, but the target's own score falls (by ≥ 2.3 relative to the others, and
     below the sink). The keys become more alike (entry-specific energy −19 %). The target key
     *loses its match strength* — enough to lose to the sink, not to other values.
   - **G2-less query:** every value's score rises (oth−sink −2.90 → +0.76), the target's rises
     less. The query *stops being specific* and matches all values. **This is the source of the
     "attention leaks to other values" failure in Part XII**, not the keys.
   - That is also why attention mass pointed at the keys in Part XI: a key-side loss drops the
     target below the sink (large loss of mass), a query-side loss spreads the mass over values
     while leaving plenty of it on the value tokens.
3. **G3 is purely query-side, and cleanly so.** G3-less keys are *more* selective than intact
   (7.44; entry energy 0.245), but a G3-less query no longer points at the target at all
   (selectivity 0.19, top-1 0.11 — at chance among 8 values, 0.125). G3 supplies the query's
   *content* (which entry); G2 supplies its *specificity* and the keys' match strength.

## Corrected roles

```
G0 (L0→L2 conv)   bind each value token to its key (Part XIV)
G2 (mainly L10)   both sides of the match: strengthens the target key's match and makes the
                  query specific (suppresses non-target values); keys become more distinct
G3 (mainly L14)   the query's content: which entry to look for
```

**Caveats.** Scores are from one head (L15H5); L19 is not examined. Q/K patches replace the
projections at all positions for K, including the sink, so the absolute (vs-sink) columns are
indicative; the selectivity and margin columns do not depend on a reference. The key-geometry
measure is on pre-norm, pre-RoPE projections.

---

# Part XVI: generalisation — does the circuit survive other templates, sizes and tasks?

Scripts: `scripts/exp21_generalize.py` (one model × one variant), launchers
`scripts/run_exp21_08b.sh` and `scripts/run_exp21_9b.sh`, tables `scripts/summarize_exp21.py`
→ `results/exp21/summary.md`. Logs `logs/exp21/*.log`, results `results/exp21/*.json`.
Models: **Qwen3.5-0.8B** (24 layers, softmax at 3/7/11/15/19/23, 6 GDN groups G0–G5, 16 GDN
value heads) and **Qwen3.5-9B** (`/mnt/yuang/models/Qwen3.5-9B`, 32 layers, softmax at
3/7/…/31, 8 groups G0–G7, 32 value heads, 16 query heads). Both float32, eager attention
(same as Parts I–XV).

Parts VII–XV used one template, 8 pairs, ~62 tokens. Here the circuit is **re-discovered** per
variant rather than assumed: the same probes are run on every task **and on both model sizes**,
and the reader head and the groups tested at it are chosen from the data.

## Variants (`src/gen.py`, `make_items`)

| variant | change from `chat8` | items | tokens |
|---|---|---|---|
| `chat8` | — (Parts VII–XV task, new seed) | 100 | 62 |
| `chat4` / `chat16` | 4 / 16 pairs | 100 / 100 | 46 / 94 |
| `list8` | second template: `Lookup table:` with `* key=value` lines; "Which value does K map to?" / "The entry K maps to **" | 100 | 69 |
| `rev8` | reversed entries: "lime is the value of banana" | 100 | 86 |
| `perm8` | target at each of the 8 positions, random distractor | 200 | 62 |
| `long512` | ~512 tokens of WikiText between dictionary and question | 100 | 575 |
| `mmlu` | MMLU question, options A–D, `Answer:`; corruption swaps the correct option with a same-length one, so the answer letter changes | 96 | ~82 |
| `mmlu_hint` | as `mmlu`, prefixed "Hint: the answer is ⟨option text⟩." | 219 | ~62 |

Corrupted prompts always have the clean prompt's token length and differ only inside the target
and distractor entries (asserted). MMLU keeps only items the intact model answers correctly in
both forms: 96 of 1,399 scanned (`mmlu`), 219 items in 200 of 249 scanned position groups
(`mmlu_hint`). MMLU prompts are plain text, **not** chat-templated. `long2048` is implemented
but was not run.

## Probes (per variant)

- **B1** baseline acc / gap. **B2** zero each GDN group G0–G5. **B3** zero each GDN layer.
- **B4** reader: rank all softmax heads by attention from the final token to the target entry;
  of the top 8, the reader is the one whose zeroing lowers the gap most.
- **B5** zero each of the 16 value heads of GDN layer 0.
- **B6** q·k selectivity at the reader (Part XV design, 2 × 2 Q/K from intact or ablated run),
  for the group right before the reader layer ("Gr-1") and the one before it ("Gr-2"). Entry
  score = log attention mass on the whole entry span. **B7** reader attention split.
- Options `--reader L,H` and `--qk_groups i,j,...` override the reader and the groups in B6/B7.

# A. Qwen3.5-0.8B

## A1. Group ablations (acc / gap)

| variant | baseline | G0 | G1 | G2 | G3 | G4 | G5 |
|---|---|---|---|---|---|---|---|
| chat8 | 1.00 / +21.0 | 0.00 / −0.1 | 0.86 / +8.3 | 0.66 / +5.7 | **0.13** / +3.4 | 0.98 / +16.4 | 0.95 / +14.3 |
| chat4 | 1.00 / +17.3 | 0.00 / −0.1 | 0.88 / +7.7 | 0.74 / +6.1 | 0.37 / +3.5 | 0.94 / +13.5 | 0.91 / +11.2 |
| chat16 | 1.00 / +23.4 | 0.00 / +0.0 | 0.73 / +7.5 | 0.52 / +6.1 | **0.14** / +3.7 | 0.98 / +18.7 | 0.98 / +16.4 |
| list8 | 1.00 / +20.4 | 0.00 / +0.0 | 0.79 / +5.8 | 0.56 / +3.5 | **0.03** / +0.8 | 1.00 / +17.9 | 0.99 / +14.7 |
| rev8 | 1.00 / +18.7 | 0.00 / +0.0 | **0.04** / +1.1 | **0.01** / +2.3 | **0.01** / +0.5 | 0.99 / +17.8 | 0.91 / +13.7 |
| perm8 | 1.00 / +21.2 | 0.00 / −0.1 | 0.85 / +8.0 | 0.68 / +6.4 | 0.23 / +3.7 | 0.97 / +17.3 | 0.94 / +14.3 |
| long512 | 1.00 / +20.1 | 0.00 / +0.0 | **0.05** / +4.7 | 0.43 / +5.9 | 0.20 / +3.7 | 0.94 / +15.1 | 0.94 / +13.2 |
| mmlu | 1.00 / +2.6 | 0.00 / −0.1 | 0.15 / +0.2 | 0.38 / +0.4 | 0.42 / +1.2 | 0.65 / +2.3 | 0.32 / +1.9 |
| mmlu_hint | 1.00 / +4.8 | 0.00 / +0.0 | 0.32 / +0.5 | 0.35 / +0.6 | 0.63 / +2.1 | 0.92 / +4.4 | 0.59 / +4.0 |

`chat8` reproduces Part VII (G0 0.000, G1 0.840, G2 0.600, G3 0.150, G4 0.965, G5 0.940) on a
fresh sample.

## A2. Single layers, layer-0 heads, reader

| variant | critical GDN layers (gap < 60 % of baseline) | L0 head 8 zeroed (acc / gap) | reader | attn on target |
|---|---|---|---|---|
| chat8 | L0, L2, L10, L14 | 0.01 / +2.8 | L15H5 | 0.84 |
| chat4 | L0, L2 | 0.00 / +2.1 | L15H5 | 0.88 |
| chat16 | L0, L10 | 0.00 / +2.3 | L15H5 | 0.86 |
| list8 | L0, L10, L14 | **0.66** / +6.2 | L15H5 | 0.82 |
| rev8 | L0, L2, L10, L14 | 0.01 / +0.6 | L15H2 (L19H6 has most attention, 0.80) | 0.43 |
| perm8 | L0, L2, L10 | 0.01 / +2.6 | L15H1 (H0/H1/H5 tie at 0.86–0.88) | 0.88 |
| long512 | L0, L2, L10 | 0.00 / +3.7 | L15H5 | 0.84 |
| mmlu | L0, L1, L2, L4, L5, L8 | 0.00 / +0.0 | L11H5 | 0.12 |
| mmlu_hint | L0, L2, L5, L10 | 0.00 / −0.0 | L3H0 | 0.11 |

## A3. q·k selectivity at the L15 reader (s_target − mean s_other; top-1 among entries)

| variant | group | intact | ablated K only | ablated Q only | both |
|---|---|---|---|---|---|
| chat8 | G3 | 5.20 (1.00) | 7.01 (1.00) | **0.15 (0.11)** | 0.07 (0.10) |
| chat8 | G2 | 5.20 (1.00) | 2.94 (0.99) | 2.83 (1.00) | 1.72 (0.85) |
| chat4 | G3 / G2 | 4.70 | 6.04 / 3.06 | **0.40** / 2.62 | 0.81 / 1.85 |
| chat16 | G3 / G2 | 6.04 | 8.14 / 2.86 | **0.08** / 3.35 | −0.08 / 1.77 |
| list8 | G3 / G2 | 5.24 | 6.72 / 2.94 | **−0.66** / 1.98 | −0.89 / 1.23 |
| perm8 | G3 / G2 | 6.60 | 8.45 / 4.05 | **0.28** / 2.76 | 0.39 / 1.92 |
| long512 | G3 / G2 | 4.82 | 6.47 / 2.85 | **0.69** / 2.74 | 1.26 / 1.69 |

`chat8` reproduces Part XV (intact 5.57, G2-less K 3.28, G2-less Q 2.95, both 1.85; G3-less Q
0.19, top-1 0.11). For MMLU the chosen "readers" (L11H5, L3H0) attend to the target with
≤ 0.12 and selectivity is ≈ 0 in every condition, so B6 is not meaningful there.

## A4. `rev8` at the L19 readers

The automatic pick (L15H2) was driven by a small gap drop; L19H6/H7/H5 carry the most attention
to the target (0.80 / 0.71 / 0.60). B6/B7 were rerun at each with `--reader 19,h --qk_groups
4,3,2,1` (`results/exp21/0.8B-L19H{6,7,5}_rev8.json`). L19H6 (H7, H5 agree within ±1):

| group removed | ablated K only | ablated Q only | both | attn target / others |
|---|---|---|---|---|
| — (intact 4.80, top-1 1.00) | | | | 0.80 / 0.10 |
| G4 [16–18] | 5.49 (0.99) | 5.80 (1.00) | 6.68 (1.00) | 0.91 / 0.03 |
| G3 [12–14] | 5.10 (1.00) | **−0.57 (0.00)** | −0.53 (0.00) | 0.02 / 0.82 |
| G2 [8–10] | 3.79 (1.00) | **1.41 (0.72)** | 1.25 (0.83) | 0.20 / 0.43 |
| G1 [4–6] | 4.43 (0.99) | **0.16 (0.08)** | 0.19 (0.07) | 0.08 / 0.65 |

No single L19 head is necessary: zeroing any of them leaves the gap ≥ 17.9 (baseline 18.7).

## Reading — 0.8B

1. **The core circuit holds across the dictionary variants** (`chat4/8/16`, `list8`, `perm8`,
   `long512`): G0 is necessary everywhere, L0 head 8 is a single point of failure (except
   `list8`, 0.66), the reader sits in layer 15, **G3 is purely query-side** (a G3-less query
   drops selectivity by 4.1–6.3 to chance; G3-less keys are *more* selective), and **G2 acts on
   both sides about equally** (≈ 2 each).
2. **More pairs lean more on G1 and G2** (G1 0.88 → 0.86 → 0.73, G2 0.74 → 0.66 → 0.52 for
   4 / 8 / 16 pairs); with 4 pairs G3 matters less (0.37).
3. **Long context recruits G1** (0.86 → 0.05 in `long512`) and G2 (0.43), consistent with
   Part XIII's reading of G1 as a general language-modelling block.
4. **Reversed entries change the route.** In `rev8` G1, G2 and G3 are each necessary, and the
   reading moves to layer 19. At the L19 readers the **query** depends on G1, G2 and G3 (each
   drives top-1 to chance or near it), while the **keys** barely depend on any of G1–G4
   (largest key-side loss 1.0, G2). G4, the group right before L19, is not needed — removing it
   sharpens the aim. So "the group right before the reader builds the query" is not a general
   rule; in `rev8` the query is built by G1–G3 and read two softmax layers later. G2 shifts
   towards the query side (Q loss 3.4 vs K loss 1.0; in `chat8` 2.4 vs 2.3).
5. **MMLU letter retrieval does not use this circuit.** Gaps are small (+2.6 / +4.8), no head
   attends to the correct option (≤ 0.12), many single GDN layers are critical and G5 matters.
   With such margins, part of the damage is likely generic zero-ablation harm; the 0.8B model
   also solves only 96 of 1,399 unhinted items in both forms.

# B. Qwen3.5-9B

Same nine variants, `scripts/run_exp21_9b.sh` (6–67 min per variant on one L20X). MMLU keeps
items solved in both forms: **14 of 1,348** scanned (`mmlu`) and 215 items in 200 of 239 scanned
position groups (`mmlu_hint`).

## B1. Group ablations (acc / gap)

| variant | baseline | G0 | G1 | G2 | G3 | G4 | G5 | G6 | G7 |
|---|---|---|---|---|---|---|---|---|---|
| chat8 | 1.00 / +23.8 | **0.01** / +0.2 | 1.00 / +17.6 | 1.00 / +18.4 | 1.00 / +15.4 | 1.00 / +11.6 | 1.00 / +23.2 | 1.00 / +24.6 | 1.00 / +21.8 |
| chat4 | 1.00 / +22.2 | 0.03 / +0.5 | 0.99 / +13.1 | 1.00 / +16.3 | 1.00 / +12.3 | 0.99 / +7.9 | 1.00 / +22.1 | 1.00 / +23.7 | 1.00 / +21.1 |
| chat16 | 1.00 / +25.0 | 0.00 / +0.2 | 0.99 / +19.7 | 0.98 / +20.2 | 0.99 / +16.6 | 0.99 / +13.4 | 1.00 / +24.1 | 1.00 / +25.5 | 1.00 / +21.8 |
| list8 | 1.00 / +20.8 | 0.00 / −0.0 | 1.00 / +19.8 | 0.99 / +17.5 | 1.00 / +15.4 | 1.00 / +9.7 | 0.99 / +20.4 | 1.00 / +21.0 | 1.00 / +19.3 |
| rev8 | 1.00 / +20.7 | 0.00 / +0.1 | 0.94 / +19.1 | 0.96 / +15.1 | 0.99 / +14.9 | 0.76 / +10.9 | 1.00 / +18.4 | 1.00 / +20.6 | 1.00 / +19.1 |
| perm8 | 1.00 / +23.1 | 0.01 / +0.6 | 0.99 / +16.4 | 1.00 / +17.9 | 1.00 / +14.7 | 0.99 / +10.4 | 1.00 / +22.6 | 1.00 / +24.0 | 1.00 / +21.2 |
| long512 | 1.00 / +24.7 | 0.00 / +0.0 | 0.99 / +18.5 | 0.99 / +18.0 | 1.00 / +17.4 | 0.96 / +11.7 | 1.00 / +24.0 | 1.00 / +23.9 | 1.00 / +20.6 |
| mmlu (n=14) | 1.00 / +7.2 | 0.00 / +0.1 | 0.04 / +0.9 | 0.32 / +3.6 | 0.29 / +3.3 | 0.71 / +5.7 | 0.68 / +5.9 | 0.93 / +7.1 | 0.11 / +6.4 |
| mmlu_hint | 1.00 / +7.2 | 0.01 / +0.0 | 0.49 / +3.8 | 0.97 / +6.0 | 0.98 / +4.7 | 0.97 / +5.8 | 0.97 / +6.1 | 1.00 / +6.9 | 0.85 / +6.0 |

No mid-network group is a bottleneck: on the dictionary variants every group except G0 leaves
acc ≥ 0.94 (worst: G4 in `rev8`, 0.76). The gap still moves most for **G4 [16,17,18]**
(chat8 +23.8 → +11.6).

## B2. Single layers, layer-0 heads, reader

| variant | critical GDN layers (gap < 60 % of baseline) | worst L0 value head (acc / gap) | reader (by causal test) | attn on target | most attention |
|---|---|---|---|---|---|
| chat8 | L0 (0.05 / +0.6) | h22 (1.00 / +22.7) | L19H11 | 0.83 | L19H15 0.92 |
| chat4 | L0 (0.04 / +0.8) | h22 (1.00 / +21.3) | L19H11 | 0.78 | L19H15 0.92 |
| chat16 | L0 (0.03 / +0.4) | h22 (1.00 / +24.0) | L19H11 | 0.83 | L19H15 0.91 |
| list8 | L0 (0.01 / +0.1) | h27 (1.00 / +19.9) | L27H1 | 0.89 | L19H15 0.91 |
| rev8 | L0 (0.03 / −0.1) | h19 (1.00 / +20.0) | L27H1 | 0.82 | L27H1 0.82 |
| perm8 | L0 (0.09 / +0.6) | h22 (1.00 / +21.9) | L19H11 | 0.82 | L19H15 0.92 |
| long512 | L0 (0.00 / +0.4) | h4 (1.00 / +23.1) | L19H11 | 0.92 | L19H15 0.93 |
| mmlu (n=14) | L0 (0.00 / −0.0) | h22 (0.32 / +6.9) | L15H11 | 0.20 | L19H11 0.22 |
| mmlu_hint | L0 (0.00 / −0.1) | h3 (0.96 / +7.0) | L19H10 | 0.16 | L23H9 0.25 |

**L0 is the only critical single layer, and no single L0 value head matters** — zeroing any of
the 32 leaves acc 1.00 on the dictionary variants (0.8B: head 8 → 0.00).

## B3. q·k selectivity at the automatic reader

| variant | group | intact | ablated K only | ablated Q only | both | K-loss | Q-loss |
|---|---|---|---|---|---|---|---|
| chat8 | G4 [16–18] | 5.45 (1.00) | 4.80 (1.00) | **1.36 (0.53)** | 1.32 (0.58) | +0.65 | +4.09 |
| chat8 | G3 [12–14] | 5.45 (1.00) | 5.31 (1.00) | 3.71 (1.00) | 3.65 (1.00) | +0.14 | +1.74 |
| chat4 | G4 / G3 | 3.79 | 3.32 / 3.80 | **0.84 (0.68)** / 2.54 | 0.76 / 2.58 | +0.47 / −0.01 | +2.95 / +1.24 |
| chat16 | G4 / G3 | 6.84 | 6.05 / 6.26 | **2.19 (0.51)** / 4.84 | 2.06 / 4.50 | +0.79 / +0.58 | +4.65 / +2.00 |
| perm8 | G4 / G3 | 5.37 | 4.72 / 5.27 | **1.55 (0.65)** / 3.69 | 1.46 / 3.64 | +0.64 / +0.10 | +3.82 / +1.68 |
| long512 | G4 / G3 | 6.41 | 5.88 / 6.09 | **2.24 (0.89)** / 4.71 | 2.17 / 4.63 | +0.53 / +0.32 | +4.17 / +1.70 |

For `list8` and `rev8` the automatic reader is L27H1, whose two preceding groups (G6, G5) do
nothing (all conditions 5.2–7.0); see B4. For MMLU the readers attend to the correct option with
≤ 0.20 and selectivity is ≤ 1.25 in every condition, so B6 is not informative there.

## B4. `list8` and `rev8` at their top readers (groups set by hand)

`--reader L,H --qk_groups 1,…` at the heads with the most attention to the target
(`results/exp21/9B-L{19H15,23H12,27H1}_*.json`).

| variant | reader | intact | G4: K only | **G4: Q only** | other groups tested (Q-side range) |
|---|---|---|---|---|---|
| list8 | L19H15 | 5.28 (0.99) | 4.99 (0.99) | **0.80 (0.52)** | G1–G3: 4.72–5.55 |
| list8 | L27H1 | 5.73 (1.00) | 5.67 (1.00) | **4.00 (0.99)** | G1–G3, G5, G6: 5.41–5.99 |
| rev8 | L23H12 | 3.87 (1.00) | 3.72 (1.00) | **2.29 (0.83)** | G1–G3, G5: 2.99–4.03 |
| rev8 | L27H1 | 6.58 (1.00) | 6.62 (1.00) | **3.38 (0.73)** | G1–G3, G5, G6: 5.34–6.54 |

Attention on the target with G4 removed: L19H15 0.91 → 0.15, L23H12 0.66 → 0.42,
L27H1 0.89 → 0.75 (`list8`) and 0.82 → 0.48 (`rev8`).

## Reading — 9B, and what scales

1. **Same skeleton, same relative depth.** G0 (and within it L0) is the only necessary group;
   the main reader is in layer 19 of 32 (0.59 of depth; 0.8B: layer 15 of 24, 0.63); the group
   immediately before it (**G4**) builds the **query** and nothing builds the keys
   (K-side loss ≤ 0.8 for every group tested).
2. **Redundancy grows with scale.** At 0.8B, removing G3 dropped acc to 0.13 and L0 head 8 to
   0.00. At 9B no mid group and no L0 head changes acc at all, although the same interventions
   still move the gap and still blind the first reader (G4 removed: L19H15 attention
   0.91 → 0.15). Later readers (L23, L27) rebuild a usable query — top-1 0.99 (`list8`) and
   0.73 (`rev8`) at L27H1 with a G4-less query — which is why accuracy survives.
3. **The template no longer changes the route.** The 0.8B `rev8` result (G1, G2, G3 each
   necessary, reading moved to L19) does not reproduce: at 9B `rev8` and `list8` use the same
   G4 → L19/L23/L27 path as `chat8`, with G1–G3 costing ≤ 1.3 on the query side.
4. **The 0.8B "G2 shapes both sides" result does not transfer.** The group two before the
   reader (G3 at 9B) is query-side only (Q-loss 1.2–2.0, K-loss ≈ 0), whereas 0.8B's G2 cost
   both sides ≈ 2.3 each (Part XV).
5. **MMLU at 9B is not interpretable as run.** The model solved both forms of only **14 of
   1,348** items — far below its MMLU ability — which points at the plain-text `Answer:` prompt
   (not chat-templated) rather than at retrieval. `mmlu_hint` (215 items) is cleaner and shows
   only G1 (0.49) and G7 (0.85) mattering, with no head attending to the correct option (≤ 0.25).

**Caveats.** Two model sizes, one family; n = 100–219 per variant (14 for 9B `mmlu`), no
confidence intervals on acc; zero-ablation throughout; entry scores cover whole entry spans
(for `rev8` "lime is the value of banana"), not just the value token; the per-head analyses show
what a head's aim depends on, not that any one head is required — at 9B no single softmax head
is necessary (0.8B `rev8`: zeroing any L19 head leaves the gap ≥ 17.9 of 18.7); MMLU prompts
are not chat-templated on either model.

---

# Part XVII: the query transplant — the group's write *is* the query

Script: `scripts/exp22_query_transplant.py`, results `results/exp22_transplant_{0.8B,9B}_*.json`,
logs `logs/exp22_*.log`. Models and settings as in Part XVI.

Parts X–XVI argue "group G builds the reader's query" from **damage**: ablate G and the reader
stops aiming at the target. That is necessity, and necessity cannot separate the component that
*carries* the query from the ones that merely *feed* it. This is the sufficiency test.

## Method

Two prompts share one dictionary and differ only in the queried key ("What is the value of
banana?" vs "... of mango?"), so they have identical token length and identical entry positions.
Run **A** (asks entry A) is the donor; run **B** (asks entry B) is the host. For one GDN group,
every layer's mixer output is recorded in run A and substituted into run B at one of four
position sets:

| set | positions |
|---|---|
| `final` | the last token only — where the reader forms its query |
| `question` | the question + assistant prefix |
| `dict` | the dictionary span (control: the query is not built there) |
| `all` | every position |

Measured at the final token: the rate of answering **A's** value (`ansA`), of answering B's
(`ansB`), `D(A−B) = logit(A) − logit(B)`, and the reader's attention on entries A and B.
100 prompt pairs per condition (4 target/distractor configs × 25), single-token values,
assertion-checked equal lengths. Baselines: run A answers A (1.00), run B answers B (0.99–1.00),
with 0.78–0.91 of the reader's attention on its own target.

## Results — transplanting at the **final token only**

| model | variant | reader | steering group | `ansA` | attn on A (from) | `D(A−B)` | every other group |
|---|---|---|---|---|---|---|---|
| 0.8B | chat8 | L15H5 | **G3 [12,13,14]** | 0.99 | 0.84 (0.00) | +4.69 | no change |
| 0.8B | list8 | L15H5 | **G3** | 1.00 | 0.81 (0.00) | +6.41 | no change |
| 0.8B | rev8 | L19H6 | **G3** | 0.97 | 0.77 (0.01) | +4.67 | no change |
| 9B | chat8 | L19H11 | **G4 [16,17,18]** | 0.99 | 0.78 (0.00) | +2.40 | no change |
| 9B | list8 | L19H15 | **G4** | 0.97 | 0.89 (0.01) | +6.18 | no change |
| 9B | rev8 | L27H1 | **G4** | 0.98 | 0.80 (0.00) | +2.44 | no change |

Over the `question` span the same group reaches `ansA` 0.99–1.00 and `D(A−B)` +6.2 … +10.1.
The `dict` control changes nothing for any group in any run (`ansB` stays 0.99–1.00). Full
tables per group and position set are in the JSON.

**G0 is the expected exception.** Transplanted over `question` or `all`, G0 also flips the answer
(`ansA` 0.99–1.00) — it carries the re-embedded *key token*, which is the one thing the two
prompts differ in. At `final` alone G0 does nothing. G0 supplies token identity; G3 / G4 supplies
the query.

## Reading

1. **Sufficiency, in one vector.** Substituting one group's write at a **single position** makes
   the model answer a question that is not in its prompt, at 0.97–1.00 accuracy, and moves the
   reader's attention with it. The query is a vector at the final token, not a property spread
   over the question.
2. **The carrier is template-independent.** One group per model — G3 at 0.8B, G4 at 9B, in both
   cases the group immediately before the first strong reader — carries the query in all three
   templates, `rev8` included.
3. **Necessity was hiding the division of labour.** Part XVI found that in 0.8B `rev8`, removing
   G1, G2 *or* G3 each destroys retrieval, and the q·k split showed each of them damaging the
   query. The transplant separates them: **only G3 is sufficient** (0.97); transplanting G1 or G2
   changes nothing (`ansB` 0.99). G1 and G2 feed the query that G3 assembles; they do not carry
   it. The same holds at 9B, where G1–G3 are necessary-ish for the gap but only G4 steers.
4. **Scale does not move the carrier, only the redundancy.** The 9B result is the same mechanism
   with a smaller logit swing (`D(A−B)` +2.4 vs +4.7 at `final`), consistent with Part XVI: later
   readers partly rebuild the host prompt's own query.

**Caveats.** One seed (`--seed 0`), n = 100 pairs, no confidence intervals; donor and host share
a dictionary, so this shows the query selects *which entry*, not that it carries content
independent of the dictionary; substitution is exact-copy (not resample), so the transplanted
write is on-distribution for run A but not necessarily for run B; the reader for each variant is
the head with the most attention on the target (Part XVI B4), and for 0.8B `rev8` that is L19H6,
not the causally-picked L15H2.

---

# Part XVIII: a second family — IBM Granite 4.0 H (Mamba-2 hybrid)

Models (downloaded to `models/`): **`granite-4.0-h-1b`** (tag `G1b`, 1B dense, 40 layers,
hidden 1536, 12 query / 4 KV heads, `head_dim` 128, Mamba-2 with 48 heads × 64) and
**`granite-4.0-h-tiny`** (tag `Gtiny`, 6.9B total / ~1B active, 64 experts, 6 per token, same
layer geometry). Softmax attention sits at layers **5, 15, 25, 35**; every other layer is
Mamba-2. Groups are therefore *all* linear layers since the previous softmax layer
(`--group_size 0`): **G0 [0–4], G1 [6–14], G2 [16–24], G3 [26–34]**.

Launchers `scripts/run_exp21_granite.sh` (battery, 18 runs) and `scripts/exp22_query_transplant.py`
(transplant, 6 runs); results `results/exp21/{G1b,Gtiny}_*.json`,
`results/exp22_transplant_{G1b,Gtiny}_*.json`; logs in `logs/exp21/` and `logs/exp22_*`.

**Porting.** The experiment scripts are now family-agnostic rather than forked:
`src/runner.py` treats `mamba` like `linear_attention`, resolves the mixer by layer type
(Granite's attention layers carry a `mamba` attribute set to `None`), and exposes `head_dim`,
`n_linear_heads`, `linear_head_dim`, `q_gated` and `groups(size)`; `src/gen.py` gains
`chat_wrap(tok)`, which keeps Qwen's hand-written wrapper (so Parts I–XVII stay comparable) and
derives the wrapper from the tokenizer otherwise; `src/qkv.py`'s `ProjPatch` takes `gated`,
because Qwen's `q_proj` emits `[query; gate]` per head while Granite's emits the query only.
*That last one was a silent failure: until it was fixed the Q patch was a no-op and B6 reported
exactly zero query-side effect.* Regression check: Qwen3.5-0.8B `chat8` reproduces Part XVI
exactly (baseline +21.04; groups 0.00 / 0.86 / 0.66 / 0.13 / 0.98 / 0.95; selectivity 5.20).

## 1. Group ablations (acc / gap)

| variant | model | baseline | G0 [0–4] | G1 [6–14] | G2 [16–24] | G3 [26–34] |
|---|---|---|---|---|---|---|
| chat8 | G1b | 0.99 / +28.4 | **0.00** / +0.2 | 0.43 / +6.6 | **0.00** / +1.8 | 0.57 / +25.4 |
| chat8 | Gtiny | 0.99 / +24.5 | **0.00** / −0.1 | 0.53 / +10.3 | 0.49 / +8.3 | 0.70 / +19.4 |
| chat4 | G1b / Gtiny | 0.99 | 0.00 / 0.00 | 0.44 / 0.54 | **0.00** / 0.64 | 0.58 / 0.66 |
| chat16 | G1b / Gtiny | 0.98 / 0.99 | 0.00 / 0.00 | 0.49 / 0.56 | **0.00** / 0.40 | 0.67 / 0.60 |
| list8 | G1b / Gtiny | 1.00 | 0.00 / 0.00 | **0.06** / 0.26 | **0.01** / 0.55 | 0.62 / 0.81 |
| rev8 | G1b / Gtiny | 0.96 / 0.90 | 0.00 / 0.00 | **0.01** / **0.02** | **0.00** / 0.21 | 0.44 / 0.65 |
| perm8 | G1b / Gtiny | 1.00 | 0.00 / 0.00 | 0.47 / 0.53 | **0.00** / 0.57 | 0.61 / 0.71 |
| long512 | G1b / Gtiny | 1.00 / 0.99 | 0.00 / 0.00 | 0.25 / 0.35 | **0.00** / 0.43 | 0.22 / 0.60 |
| mmlu | G1b / Gtiny | 1.00 | 0.03 / 0.00 | 0.07 / 0.36 | 0.24 / 0.27 | 0.85 / 0.92 |
| mmlu_hint | G1b / Gtiny | 1.00 | 0.02 / 0.00 | 0.07 / 0.44 | 0.26 / 0.31 | 0.94 / 0.97 |

MMLU here is the standard 5-shot format of Part XVI's fix (≈410 tokens, 201–206 items kept).

## 2. Critical layers, layer-0 heads, readers

| variant | model | critical single layers (gap < 60 % of baseline) | worst L0 head | reader (causal) | attn | most attention |
|---|---|---|---|---|---|---|
| chat8 | G1b | L0 (0.00), L13, L23, L24 (0.73–0.86) | h25 (0.97) | L25H1 | 0.81 | L25H1 0.81 |
| chat8 | Gtiny | **L0 only** (0.00) | h33 (1.00) | L25H2 | 0.76 | L25H2 0.76 |
| list8 | G1b / Gtiny | L0, L23, L24 / **L0 only** | h25 (0.92) / h33 (1.00) | L25H1 / L25H2 | 0.82 / 0.74 | L25H1 / L25H8 0.81 |
| rev8 | G1b / Gtiny | L0, L13, **L14 (0.03)**, L23 / **L0 only** | h25 (0.93) / h8 (0.90) | L25H8 / L25H2 | 0.39 / 0.42 | **L35H6 0.66 / L35H1 0.71** |
| long512 | G1b / Gtiny | L0, L13, L14, L24 / L0, L24 | h25 (0.96) / h8 (0.99) | L25H1 / L25H2 | 0.89 / 0.77 | same |
| mmlu_hint | G1b / Gtiny | L0, L14, L24 / **L0 only** | h25 (0.88) / h8 (0.96) | L25H8 / L25H2 | 0.40 / **0.68** | same |

Layer 25 is 0.63 of depth — the same relative position as the readers in Qwen3.5-0.8B (L15/24 =
0.63) and 9B (L19/32 = 0.59).

## 3. q·k selectivity at the reader (`Gr-1` = G2 [16–24], `Gr-2` = G1 [6–14])

| variant | model | group | intact | ablated K only | ablated Q only | both |
|---|---|---|---|---|---|---|
| chat8 | G1b | G2 | 6.09 (1.00) | 2.75 (0.97) | **−0.13 (0.01)** | 0.02 (0.13) |
| chat8 | G1b | G1 | 6.09 (1.00) | 4.07 (0.96) | 2.32 (0.79) | 2.51 (0.71) |
| chat8 | Gtiny | G2 | 5.29 (1.00) | 2.72 (0.93) | **−0.20 (0.00)** | −0.04 (0.06) |
| chat8 | Gtiny | G1 | 5.29 (1.00) | 3.29 (0.89) | 2.16 (0.66) | 2.27 (0.65) |
| list8 | G1b / Gtiny | G2 | 5.50 / 6.18 | 2.77 / 3.43 | **−0.01 (0.14) / −0.06 (0.01)** | 0.01 / 0.02 |
| rev8 | G1b / Gtiny | G2 | 6.08 / 4.95 | 3.21 / 3.04 | **0.01 (0.03) / −0.10 (0.00)** | 0.09 / −0.04 |
| long512 | G1b / Gtiny | G2 | 5.78 / 4.36 | 2.49 / 2.61 | **−0.04 (0.02) / −0.27 (0.00)** | 0.05 / −0.21 |
| mmlu_hint | G1b / Gtiny | G2 | 2.92 / 3.67 | 1.30 / 3.00 | 1.37 / 1.22 | 0.37 / 1.29 |

Chance top-1 among 8 entries is 0.125. **G2 is query-side** (top-1 → 0.00–0.14 with a G2-built
query) **but also carries the keys**: ablated keys alone cost 2.3–3.3 selectivity, where the
matching Qwen groups cost ≤ 0.8 (Part XVI B3/B4).

## 4. Query transplant (donor = the other question; Part XVII method)

| model | variant | reader | group | at `final` only | over `question` |
|---|---|---|---|---|---|
| G1b | chat8 | L25H1 | **G2** | ansA **1.00**, attn A 0.83 | ansA 1.00, D +13.45 |
| G1b | list8 | L25H1 | **G2** | ansA **0.98**, attn A 0.84 | ansA 1.00, D +12.57 |
| G1b | rev8 | L35H6 | **G2** | ansA **0.92**, attn A 0.39 | ansA 0.96, attn A 0.66 |
| Gtiny | chat8 | L25H2 | G2 | **ansA 0.03**, attn A **0.80** | ansA 0.63, attn A 0.79 |
| Gtiny | list8 | L25H2 | G2 | **ansA 0.06**, attn A **0.76** | ansA 0.57, attn A 0.75 |
| Gtiny | rev8 | L35H1 | G2 | ansA 0.05, attn A 0.08 | ansA 0.39, attn A 0.31 |

G0, G1 and G3 do nothing at `final` in either model; the `dict` control is null throughout.

## Reading

1. **Three findings replicate across families.** (i) The first linear block is necessary and
   **layer 0 alone** accounts for it (0.00 in every variant, both models). (ii) The reader sits
   at ≈ 0.6 of depth — now in four models and two families. (iii) The block immediately before
   the reader supplies the reader's **query**: a query built without it leaves the reader at or
   below chance (top-1 0.00–0.14).
2. **The transplant replicates in G1b**: one block's write at a **single position** makes the
   model answer the other question (0.92–1.00 across three templates), and no other block does.
   So "the linear layers hand the softmax reader its query" is not a Qwen artefact.
3. **But the clean query/key split is.** Granite's query block also shapes the keys (K-side loss
   2.3–3.3 vs ≤ 0.8 in Qwen 9B). Whether one block owns *only* the query is architecture-specific;
   that it owns the query is not.
4. **No layer-0 head is a single point of failure** in either Granite model (worst 0.90–1.00),
   as in Qwen 9B but unlike Qwen 0.8B (head 8 → 0.00). The Part IX single-head result does not
   generalise.
5. **Gtiny dissociates attention from behaviour — the most interesting result here.** The G2
   transplant moves the reader's attention onto the donor's entry exactly as designed
   (0.01 → 0.80) while the model still answers the *host's* value (ansA 0.03, D −7.67). Steering
   the tracked reader is necessary but not sufficient; some other path keeps the answer on B.
   Transplanting over the whole question span gets ansA to 0.57–0.63, i.e. partway. The same
   direction appeared in Qwen 9B (smaller logit swing than 0.8B, later readers rebuilding the
   query); the MoE model makes it a clean dissociation. Which heads hold the answer when the
   reader has been steered is **not yet measured**.
6. **`rev8` moves the reading deeper in both Granite models** (most attention at L35, not L25),
   as it did in Qwen3.5-0.8B; and in G1b, **L14 alone** drops `rev8` to 0.03, the only
   single-layer catastrophe outside layer 0 anywhere in Parts XVI–XVIII.
7. **Granite shows the circuit on MMLU where Qwen did not.** With the hint, the reader attends to
   the correct option at 0.40 (G1b) and 0.68 (Gtiny) with selectivity 2.9–3.7, against ≤ 0.25 and
   ≤ 1.25 for Qwen. The mechanism is not confined to the synthetic task in this family.

**Caveats.** `Gtiny` is an MoE model, so its differences from `G1b` confound scale, sparsity and
routing (no dense 7B Granite H exists). Groups here span 5 or 9 layers against Qwen's 3, so
"group" ablations remove more of the network — the G2/G1 comparisons are within-model, not
across families. One seed, n = 100–206, no confidence intervals, zero-ablation throughout.
Granite's chat template inserts a default system prompt (from its tokenizer), which Qwen's does
not; prompt lengths therefore differ by ~17 tokens for the same dictionary. The `rev8`
transplants use the top-attention head (L35) rather than the causally chosen one.

---

# Part XIX: a minimal hybrid designed from the circuit (paper exp 2)

Scripts: `scripts/exp25_minimal.py` (phases P1–P5), `scripts/exp25b_lmeval_matched.py`
(lm-eval at matched size), `scripts/exp25c_shard.py` (parallel finish of 9B); shared code
`src/minimal.py`. Results: `results/exp25_minimal_<tag>.json`, `results/exp25b_lmeval_<tag>.json`.
Tables: `scripts/summarize_exp25_27.py` → `results/exp25_27_summary.md`.

**Recipe: GnA's training-free "minimal model"** (arXiv:2504.18574 §3.2–3.4), applied to the
*linear* layers. GnA prune a network to its smallest sub-network that still retrieves, then
show that retrieval (MMLU, KV-retrieval) survives while knowledge-heavy benchmarks fall. Here:

- **Removing a linear layer's mixer.** Only the token mixer goes; softmax layers and every MLP stay.
  Two forms:
  - **mean**: the mixer's output is replaced by its WikiText-2 mean, a constant bias per layer.
    This removes the layer's token mixing and its recurrent state.
  - **zero**: the output is zeroed (Parts VII–XVIII).
- **Truncation.** Keep layers `0..c` and send the residual straight to the final norm and LM head.
  `c*` is the smallest softmax layer whose dev *candidate* accuracy reaches 95 % of full depth.
- **Metrics:**
  - `acc`: full-vocabulary argmax on the dictionary suite (`chat8/16`, `list8`, `rev8`, `long512`).
  - `cacc`: GnA-style choice scoring among the dictionary's 8 values.
  - `kv20`: GnA's KV-retrieval, 20 pairs, trailing space restored.
  - WikiText-2 PPL.
  - lm-eval: ARC-e/c, PIQA, HellaSwag, WinoGrande (0-shot, 500 each); MMLU 5-shot (10 per
    subject); FDA and SWDE (real-document recall, 200 each).
- **Selection vs report.** Every choice (cut, greedy set) is made on a dev dictionary sample
  (seed 1), and every number below is on the seed-0 test sample.

**Which linear layers to keep.**
- **Circuit rule, fixed in advance:** keep the linear layers upstream of the first reader,
  i.e. Bind, the feeders, and Query (`pre-reader`). Everything after the reader, which
  Part VIII identified as "transport", is replaced by its mean.
- **Circuit-refined:** greedy backward elimination from the full stack (mean removal), dropping the
  layer whose loss hurts dev retrieval least, until dev accuracy falls below 90 % of intact.
- **Baselines at the same size:**
  - PPL-ranked: keep the layers whose single removal costs most perplexity (Borobia et al.'s
    importance).
  - First-k, last-k.
  - Random, 5 seeds (3 in lm-eval).

## 1. Truncation reproduces GnA's minimal model (P1)

Candidate accuracy on the dictionary suite, keeping layers 0..cut:

| model | cut after softmax layer | mean cacc | full-vocab acc | kv20 | note |
|---|---|---|---|---|---|
| 0.8B | 11 / **15** (first reader) / **19** (= c*) / full | 0.12 / 0.89 / **1.00** / 1.00 | 0.00 / 0.00 / 0.39 / 1.00 | 0.08 / 0.17 / 0.64 / 0.83 | |
| G1b | 15 / **25** (first reader) / **35** (= c*) / full | 0.11 / 0.62 / **0.99** / 0.99 | 0.00 / 0.00 / 0.01 / 0.99 | 0.08 / 0.84 / 0.99 / 1.00 | |
| Gtiny | 15 / 25 / **35** (= c*) / full | 0.11 / 0.28 / **0.98** / 0.98 | 0.00 / 0.00 / 0.00 / 0.98 | 0.04 / 0.62 / 0.99 / 1.00 | |
| 9B | 15 / **19** (first reader) / **23** (= c*) / 27 / full | 0.12 / 0.71 / **0.95** / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 / 0.00 / 1.00 | 0.06 / 0.22 / 0.80 / 0.82 / 0.83 | |

Choosing among the dictionary's values already works right after the reader, and is complete one
softmax layer later. Full-vocabulary argmax needs the rest of the network: the answer is *in*
the residual stream but not yet *top-1 over the vocabulary*. That is the gap between GnA's
answer scoring and generation. On lm-eval the truncated models show GnA's signature:

| model (truncated at c*) | knowledge avg (intact) | MMLU-5 (intact) | FDA (intact) |
|---|---|---|---|
| 0.8B, layers 0–19 | 0.44 (0.55) | **0.49 (0.51)** | 0.10 (0.83) |
| G1b, layers 0–35 | 0.52 (0.63) | **0.61 (0.61)** | 0.13 (0.73) |
| Gtiny, layers 0–35 | 0.62 (0.73) | **0.72 (0.72)** | 0.14 (0.77) |
| 9B, layers 0–23 | 0.52 (0.72) | **0.75 (0.69)** | 0.07 (0.76) |

Knowledge falls 11 points (9B: 20) while MMLU is unchanged (9B: +6). FDA/SWDE are scored by generating text, so
they need full-vocabulary answers and fall with it.

## 2. Which single linear layers matter (P2, mean removal, full depth)

Layers whose removal costs > 5 points of dev accuracy:

| model | layers |
|---|---|
| 0.8B | **L0** (0.00), L14 (0.84), L10 (0.90) |
| G1b | L13 (0.37), L23 (0.48), L14 (0.62), **L0** (0.81), L24 (0.83) |
| Gtiny | **L0** (0.00) |
| 9B | **L0** (0.33); no other single layer |

With mean instead of zero removal, only the Bind layer (L0) and the query-side layers of
Parts XVI–XVIII (0.8B L10/L14; G1b L13/L14/L23/L24) matter individually.

## 3. The minimal sets lie on the Bind → Query path (P3)

Greedy elimination on dev retrieval, full depth (reader in brackets):

| model | kept / linear total | minimal set | before the reader |
|---|---|---|---|
| 0.8B [L15] | **8 / 18** | 0, 1 · 4, 5, 6, 9, 10 · 14 | 8 of 8 |
| 9B [L19] | **9 / 24** | 0 · 5, 9, 10, 12, 13, 14 · 17, 18 | 9 of 9 |
| G1b [L25] | **15 / 36** | 0, 4 · 6, 8, 9, 10, 11, 13, 14 · 19, 20, 22, 23, 24 · 34 | 14 of 15 |
| Gtiny [L25] | **14 / 36** | 0, 2, 4 · 7, 8, 10, 14 · 19, 21, 22, 23, 24 · 34, 39 | 12 of 14 |

(· separates Bind block / feeder blocks / Query block / after the reader.) The data-driven sets are
not arbitrary. Every one contains layer 0 and the query block's key layers: 0.8B L14,
9B L17/L18, and Granite L23/L24, which Parts XVI–XVIII found by ablation and transplant. Apart
from Granite's L34/L39, every kept layer sits upstream of the first reader. Granite's truncated
search (layers 0–35) returns nearly the same sets (G1b 15 layers, Gtiny 13).

## 4. At matched size, the circuit sets retrieve and the baselines do not (P4)

Dictionary suite, full-vocabulary accuracy (mean over the five variants), full depth, mean
removal. Random = mean ± sd over 5 seeds.

| model | set | k | dict acc | kv20 | PPL |
|---|---|---|---|---|---|
| 0.8B | intact | 18 | 1.00 | 0.83 | 21.9 |
| | **pre-reader (rule)** | 12 | **1.00** | **0.83** | 38.9 |
| | **greedy** | 8 | **0.96** | **0.83** | 51.4 |
| | PPL-ranked | 8 | 0.11 | 0.27 | 42.8 |
| | random | 8 | 0.06 ± 0.10 | 0.12 ± 0.05 | 4669 ± 6515 |
| | bind + query blocks only | 6 | 0.04 | 0.13 | 77.1 |
| | PPL-ranked / first-6 / last-6 / random-6 | 6 | 0.07 / 0.01 / 0.00 / 0.00 | ≤ 0.16 | |
| G1b | intact | 36 | 0.99 | 1.00 | 15.1 |
| | **pre-reader (rule)** | 23 | **0.87** | **1.00** | 86.4 |
| | **greedy** | 15 | **0.84** | **0.97** | 62.9 |
| | PPL-ranked | 15 | 0.01 | 0.10 | 67.4 |
| | random | 15 | 0.09 ± 0.16 | 0.26 ± 0.35 | 682 ± 607 |
| | bind + query blocks only | 14 | 0.02 | 0.37 | 394 |
| Gtiny | intact | 36 | 0.98 | 1.00 | 11.7 |
| | pre-reader (rule) | 23 | 0.08 (cacc 0.70) | 0.90 | 77.3 |
| | **greedy** | 14 | **0.83** | **0.98** | 30.2 |
| | PPL-ranked | 14 | 0.04 | 0.03 | 33.9 |
| | random | 14 | 0.03 ± 0.05 | 0.07 ± 0.05 | 3844 ± 2930 |
| 9B | intact | 24 | 1.00 | 0.82 | 11.3 |
| | **pre-reader (rule)** | 15 | **1.00** | **0.82** | 21.0 |
| | **greedy** | 9 | **0.97** | **0.77** | 36.1 |
| | PPL-ranked | 9 | 0.67 | 0.72 | 28.4 |
| | random | 9 | 0.00 ± 0.00 | 0.09 ± 0.04 | 261 ± 128 |
| | bind + query blocks only | 6 | 0.01 | 0.18 | 62.6 |

- **9B** repeats it: 15 of 24 GDN layers (the rule) keep every dictionary variant at 1.00 and
  KV-retrieval at the intact level; 9 (greedy) keep 0.97.
  - The same-size PPL-ranked set does better here than in the smaller models (0.67), because it
    happens to contain L0, the query layer L18 and three feeders ([0, 1, 6, 8, 10, 13, 18, 20, 24]).
  - It still fails the harder variants: `rev8` 0.36 and `long512` 0.34, against greedy 0.94 / 0.92.
  - Random 9-sets are at 0.00.
- **At 8 of 18 GDN layers in 0.8B**, the circuit set keeps dictionary retrieval (0.96) and GnA
  KV-retrieval at the intact level (0.83). A PPL-ranked set of the same size keeps better
  perplexity (42.8 vs 51.4) and loses retrieval (0.11 / 0.27). Perplexity-guided compression keeps
  the wrong linear layers for retrieval, as Part VII predicted from G3 alone.
- **Mean removal is essential for the small sets.** The same sets with *zeroed* mixers retrieve at
  0.19–0.44 (greedy: 0.8B 0.21, 9B 0.19, G1b 0.44; 0.8B pre-reader 0.31). Only 9B's 15-layer
  pre-reader set survives zeroing (1.00). A removed linear layer must leave its average
  contribution behind; only its token mixing is dispensable.
- **Bind + Query alone is not enough** without training (0.02–0.04). The feeder blocks upstream
  of the query block (0.8B G1/G2, Granite G1) are needed, in line with Part VIII
  (G1+G2 feed the address) and Part XVII (they feed the query block, which carries it).
- **Gtiny is the exception to the fixed rule.** Keeping everything before the reader is not
  enough there (0.08), and its greedy set keeps two post-reader layers (L34, L39). This fits
  Part XVIII, where Gtiny's first reader can be steered without moving the answer: in this model
  the answer is also held after L25.

## 5. lm-eval at matched size (exp25b; full depth, mean removal)

| model | set | k | knowledge avg | MMLU-5 | FDA | SWDE |
|---|---|---|---|---|---|---|
| 0.8B | intact | 18 | 0.55 | 0.51 | 0.83 | 0.86 |
| | **greedy** | 8 | 0.44 | **0.35** | **0.64** | **0.67** |
| | PPL-ranked | 8 | 0.43 | 0.22 | 0.39 | 0.58 |
| | random ×3 | 8 | 0.35–0.43 | 0.21–0.26 | 0.00–0.20 | 0.00–0.43 |
| | **pre-reader** | 12 | 0.47 | **0.54** | 0.73 | **0.83** |
| | PPL-ranked | 12 | 0.47 | 0.26 | 0.77 | 0.73 |
| | random ×3 | 12 | 0.43–0.47 | 0.22–0.45 | 0.29–0.74 | 0.65–0.84 |
| G1b | intact | 36 | 0.63 | 0.61 | 0.73 | 0.91 |
| | **greedy** | 15 | 0.44 | **0.30** | **0.38** | **0.65** |
| | PPL-ranked | 15 | 0.41 | 0.24 | 0.03 | 0.29 |
| | random ×3 | 15 | 0.38–0.41 | 0.22–0.24 | 0.00–0.15 | 0.01–0.39 |
| | **pre-reader** | 23 | 0.48 | **0.59** | **0.20** | **0.62** |
| | PPL-ranked | 23 | 0.47 | 0.27 | 0.14 | 0.50 |
| | random ×3 | 23 | 0.41–0.46 | 0.23–0.25 | 0.02–0.17 | 0.24–0.38 |
| Gtiny | intact | 36 | 0.73 | 0.72 | 0.77 | 0.91 |
| | **greedy** | 14 | 0.55 | **0.45** | **0.25** | **0.70** |
| | PPL-ranked | 14 | 0.49 | 0.27 | 0.08 | 0.34 |
| | random ×3 | 14 | 0.36–0.49 | 0.24–0.26 | 0.00–0.01 | 0.00–0.09 |
| | pre-reader | 23 | 0.58 | **0.71** | 0.08 | 0.26 |
| | PPL-ranked | 23 | 0.64 | 0.49 | 0.71 | 0.86 |
| | random ×3 | 23 | 0.38–0.58 | 0.21–0.54 | 0.00–0.23 | 0.16–0.45 |
| 9B | intact | 24 | 0.72 | 0.69 | 0.76 | 0.91 |
| | greedy | 9 | 0.52 | 0.25 | 0.26 | 0.75 |
| | PPL-ranked | 9 | 0.56 | 0.32 | 0.31 | 0.71 |
| | random ×3 | 9 | 0.40–0.54 | 0.27–0.30 | 0.01–0.29 | 0.15–0.84 |
| | **pre-reader** | 15 | 0.66 | **0.73** | 0.43 | 0.65 |
| | PPL-ranked | 15 | 0.67 | 0.57 | 0.59 | 0.87 |
| | random ×3 | 15 | 0.37–0.64 | 0.21–0.63 | 0.01–0.51 | 0.03–0.79 |

1. **MMLU is carried by the pre-reader linear layers.** Keeping only the linear layers upstream of
   the first reader leaves MMLU at the intact level in all four models:
   - 0.8B / G1b / Gtiny / 9B: 0.54 / 0.59 / 0.71 / 0.73, against intact 0.51 / 0.61 / 0.72 / 0.69.
   - Same-size PPL-ranked sets: 0.26 / 0.27 / 0.49 / 0.57.
   - Random sets: 0.21–0.63. Knowledge declines to about the same level either way. This is GnA's MMLU result
   restated for the linear layers of a hybrid: the linear layers the retrieval circuit uses are
   the ones MMLU needs, and all linear layers downstream of the reader can become constants.
2. **Real-document recall follows the greedy circuit set at small k**, in the three models up to
   1B active parameters (not 9B, below). FDA/SWDE:
   - 0.8B: 0.64 / 0.67 vs 0.39 / 0.58
   - G1b: 0.38 / 0.65 vs 0.03 / 0.29
   - Gtiny: 0.25 / 0.70 vs 0.08 / 0.34

   Knowledge is equal or better than the PPL-ranked set in every case.
3. **Limits.**
   - At k = 12 in 0.8B, FDA does not separate the sets (pre-reader 0.73, PPL-ranked 0.77, one
     random set 0.74).
   - In Gtiny the fixed rule loses FDA/SWDE, where the PPL-ranked set of 23 keeps them. The
     generation tasks use the post-reader layers that Gtiny's greedy set also keeps.
   - Minimal sets selected on the dictionary task do not keep MMLU at small k (0.25–0.45).
   - **9B does not separate on lm-eval.** Its 9-layer greedy set wins on the dictionary suite
     (0.97 vs 0.67 PPL-ranked, 0.00 random). On MMLU, FDA and SWDE it is level with or below
     same-size baselines (0.25 / 0.26 / 0.75 vs PPL-ranked 0.32 / 0.31 / 0.71). At k = 15 the
     rule wins on MMLU but trails on FDA/SWDE. This is the scale redundancy of Part XVI: at 9B
     later readers rebuild the query, so which pre-reader layers survive matters less for
     natural recall.

**Efficiency.** A GDN mixer in 0.8B is 10.5 M parameters (all 18: 190 M, 25 % of the model), and its
state is 16 × 128 × 128 floats. The 8-layer hybrid drops 105 M parameters (14 %) and 56 % of the
recurrent state. The 12-layer rule drops 63 M (8 %) and 33 %. In Granite-1B a Mamba-2 mixer is
14.6 M (all 36: 527 M, 36 %). The 15-layer set drops 307 M (21 %) and 58 % of the state; the
23-layer rule drops 190 M (13 %) and 36 %. Each removed mixer leaves one constant vector.

## Reading

1. **The circuit tells you which linear layers a hybrid can do without.** Every linear layer
   downstream of the first reader can be replaced by a constant, with no training, while
   dictionary retrieval and KV-retrieval stay at the intact level (0.8B, 9B; G1b 0.87), and so
   does MMLU (all four models). A greedy search that knows nothing about the circuit lands on
   the Bind → feeders → Query path, and in 0.8B keeps 8 of 18 linear layers (9B: 9 of 24).
2. **Perplexity is the wrong guide.** The layers that matter most for perplexity are not the ones
   retrieval needs. Same-size PPL-ranked hybrids have similar or better perplexity and knowledge,
   but lose the dictionary task in all four models. In the three ≤ 1B-active models they also
   lose MMLU and FDA/SWDE. In 9B they hold natural recall (see Limits in §5).
3. **Training-free has limits:**
   - The two circuit blocks alone (Bind + Query) are not enough without their feeders.
   - Knowledge-heavy scores fall, as in GnA.
   - Gtiny needs two post-reader layers.

**Caveats.**
- One seed per configuration (random baselines: 5 / 3 seeds).
- lm-eval limits: 500 per knowledge task, 10 per MMLU subject, 200 for FDA/SWDE, so differences
  below about 0.05 are not resolved.
- The greedy selection uses the dictionary dev set, so its dictionary numbers are optimistic. Its
  KV/lm-eval numbers are not.
- Mean removal needs one pass over WikiText to set the constants.
- No recovery training was done. A distilled version of these hybrids is the natural next step.

---

# Part XX: transferring the circuit between families (paper exp 3)

Scripts: `scripts/exp26_xfamily_query.py` (A, query vector), `scripts/exp27_xfamily_block.py`
(B, query block), shared prompt material `src/xfam.py`; launchers `scripts/run_exp26.sh`,
`scripts/run_exp27.sh`; results `results/exp26_xquery_*.json`, `results/exp27_xblock_*.json`;
tables `scripts/summarize_exp25_27.py` → `results/exp25_27_summary.md`.

Parts XVII–XVIII showed the same Bind–Query layout in Qwen3.5 (Gated DeltaNet) and Granite-4.0-H
(Mamba-2). Here the circuit is moved *between* the families. The two residual streams have
different widths (1024 / 4096 / 1536), tokenizers and bases, so everything that crosses is passed
through linear maps fitted in closed form (ridge) on intact runs of both models. There is **no
gradient training anywhere**. The inspiration is GnA's hybrid replacement (arXiv:2504.18574
§5.4): swap one layer from another model in without fine-tuning, and see whether the capability
that layer carries comes back.

**Shared prompts.** 167 keys and 30 values that are single tokens, with and without a leading space,
in all three tokenizers (`src/xfam.py`). Every model reads the same dictionary text, each wrapped in
its own chat template. Keys are split in half with a fixed seed: **maps are fitted only on `K_fit`
dictionaries and tested only on `K_test` dictionaries**, so no test key (queried or distractor)
was ever seen by a map. Test n = 200. Query blocks and readers are those of Parts XVI–XVIII
(0.8B G3 [12–14] → L15H5; 9B G4 [16–18] → L19H11; Granite G2 [16–24] → L25H1/H2).

## A. The query vector (`exp26`)

Two runs share one dictionary: the donor asks key A, the receiver (other family) asks key B.
A ridge map f takes the donor's query-block write at the final token to the receiver's query-block
write, fitted on 6,000 paired prompts. f(donor asks A) replaces the receiver's query-block write,
at the final token only. `ansA` is the rate at which the receiver answers **A**, i.e. the donor's
question.

| donor → receiver | variant | in-family (Part XVII) | **cross-family** | reader attn on A | self-map keeps B | shuffled map | mean query |
|---|---|---|---|---|---|---|---|
| 0.8B → G1b | chat8 | 1.00 | **0.98** | 0.72 | 0.99 | 0.03 | 0.03 |
| 0.8B → G1b | list8 | 0.97 | **0.97** | 0.76 | 0.99 | 0.04 | 0.04 |
| 0.8B → G1b | rev8 | 0.92 | **0.87** | 0.36 | 0.98 | 0.01 | 0.01 |
| 9B → G1b | chat8 | 1.00 | **0.98** | 0.74 | 0.99 | 0.02 | 0.03 |
| 9B → G1b | list8 | 0.97 | **0.94** | 0.78 | 0.99 | 0.04 | 0.04 |
| G1b → 0.8B | chat8 | 0.97 | **0.91** | 0.68 | 0.98 | 0.00 | 0.00 |
| G1b → 0.8B | list8 | 1.00 | **0.96** | 0.67 | 0.99 | 0.04 | 0.04 |
| G1b → 0.8B | rev8 | 0.98 | **0.93** | 0.46 | 0.92 | 0.04 | 0.04 |
| Gtiny → 0.8B | chat8 | 0.97 | **0.91** | 0.69 | 0.96 | 0.00 | 0.00 |
| Gtiny → G1b | chat8 | 1.00 | **1.00** | 0.74 | 0.99 | 0.02 | 0.03 |
| G1b → 9B | chat8 | 0.92 | **0.73** | 0.41 | 0.99 | 0.00 | 0.00 |
| G1b → 9B | list8 | 0.97 | **0.77** | 0.52 | 1.00 | 0.00 | 0.00 |
| Gtiny → 9B | chat8 | 0.92 | **0.78** | 0.41 | 1.00 | 0.00 | 0.00 |
| 0.8B → 9B (same family, other scale) | chat8 | 0.92 | **0.79** | 0.43 | 1.00 | 0.00 | 0.00 |
| 9B → 0.8B (same family, other scale) | chat8 | 0.97 | **0.94** | 0.73 | 0.97 | 0.00 | 0.00 |

A query built by one family, linearly re-expressed, is used by the other family's reader as its own
query. The receiver answers the donor's question on keys the map never saw, at 0.73–1.00 against
0.92–1.00 for the in-family transplant, and the reader's attention moves with it. The controls
hold: mapping the donor's *own-question* run leaves the receiver on B (0.92–1.00), and a map
fitted on shuffled pairs or the mean write does nothing (≤ 0.04). The weakest receiver is 9B
(0.73–0.79), the model whose later readers rebuild the host's own query (Part XVI).

**Is it the query block, or anything that knows the key?** With 6,000 fit prompts, maps from
most donor blocks eventually steer the receiver as well: by the final token, blocks after the
query block also carry the queried key. The donor's static input embedding of the key never does
(≤ 0.40), so context matters. The discriminating measure is **how little data a block needs**,
i.e. how close its geometry already is to the receiver's query. Rate at 25 paired prompts, and the
fit size at which each source first reaches 0.8 (— = never):

| donor → receiver | variant | query block @25 | best other block @25 | fit prompts to reach 0.8: query block / other blocks |
|---|---|---|---|---|
| 0.8B → G1b | chat8 | **0.83** | 0.62 (G2) | **25** / G2 100, G4 200, G5 400, G0/G1/emb — |
| 0.8B → G1b | list8 | **0.81** | 0.47 | **25** / G2 100, G4 200, G5 400 |
| 0.8B → G1b | rev8 | **0.54** | 0.34 | **100** / G4 800, G5 800, G2 3200 |
| 9B → G1b | chat8 | **0.82** | 0.75 (G3) | **25** / G3 50, G5 100, G6 200, G2/G7 400 |
| 9B → G1b | list8 | **0.79** | 0.64 | **50** / G3 100, G5 100, G2 200, G6/G7 800 |
| 9B → 0.8B | chat8 | **0.52** | 0.45 | **50** / G3 100, G5 200, G6 400, G2/G7 800 |
| G1b → 0.8B | chat8 | **0.59** | 0.11 (G3) | **100** / G3 400 |
| G1b → 0.8B | list8 | **0.74** | 0.30 | **50** / G3 200 |
| G1b → 0.8B | rev8 | **0.65** | 0.46 | **100** / G3 100 |
| Gtiny → 0.8B | chat8 | **0.49** | 0.20 | **200** / G3 400 |
| Gtiny → G1b | chat8 | **0.86** | 0.56 | **25** / G3 50, G1 800 |
| G1b → 9B, Gtiny → 9B, 0.8B → 9B | chat8 (G1b also list8) | **0.22–0.32** | 0.03–0.06 | (9B receiver: only the query block approaches 0.8) |

In **all 15 runs** the donor's query block is the best source at 25 prompts. It is also the first
to reach 0.8, once tied (G1b → 0.8B `rev8`, with G3). Twenty-five dictionaries suffice to align
the 0.8B query (G3) with Granite-1B's (0.83 on unseen keys). The block after the reader needs 8–16×
more, and early blocks (G0, G1) never get there. Of everything a model computes about the queried
key, the query block's write is the most linearly aligned with the other family's query.

## B. The query block (`exp27`)

GnA §5.4 across families. The receiver's query block is removed. In its place the donor's block
runs on the receiver's own residual stream, mapped in and out:

`h_out = h_in + M_out( B_donor(M_in(h_in)) − M_in(h_in) )`

- `M_in` maps the receiver residual to the donor residual at the block input. It is fitted on tokens
  that cover identical characters in both tokenisations (1,500 `K_fit` dictionaries + 200 WikiText
  chunks).
- `B_donor` is the donor's decoder layers run unchanged, over the whole sequence.
- `M_out` maps the donor block's write to the intact receiver's block write (h_out − h_in), at every
  position. It is fitted with `B_donor` running on `M_in(h_in)`, as at test.

Controls:
- **bypass:** the best position-wise linear stand-in, `h_out = h_in + M(h_in)`, with no token mixing.
- **every other donor block,** stitched in the same way.
- **the donor's query block with its linear mixers zeroed** (MLPs only).

Accuracy is full-vocabulary argmax on 200 `K_test` prompts; "attn" is the receiver reader's
attention on the target.

| donor → receiver | variant | intact | block removed | bypass | **donor query block** (attn) | same, mixers zeroed | best other donor block |
|---|---|---|---|---|---|---|---|
| 0.8B G3 → G1b | chat8 | 1.00 | 0.01 | 0.24 | **0.92** (0.34) | 0.28 | 0.39 (G2) |
| 0.8B G3 → G1b | list8 | 1.00 | 0.00 | 0.28 | **0.94** (0.55) | 0.26 | 0.34 |
| 0.8B G3 → G1b | rev8 | 0.97 | 0.00 | 0.14 | **0.83** (0.24) | 0.14 | 0.26 |
| 9B G4 → G1b | chat8 | 1.00 | 0.01 | 0.24 | **0.96** (0.50) | 0.38 | 0.79 (G3) |
| G1b G2 → 0.8B | chat8 | 0.99 | 0.30 | 0.28 | **0.95** (0.46) | 0.28 | 0.32 |
| G1b G2 → 0.8B | list8 | 1.00 | 0.18 | 0.31 | **0.93** (0.39) | 0.29 | 0.34 |
| G1b G2 → 0.8B | rev8 | 1.00 | 0.15 | 0.10 | **0.77** (0.37) | 0.11 | 0.12 |
| 9B G4 → 0.8B (same family) | chat8 | 0.99 | 0.30 | 0.28 | **0.99** (0.67) | 0.38 | 0.69 |
| Gtiny G2 → 0.8B | chat8 | 0.99 | 0.30 | 0.28 | 0.54 (0.18) | 0.32 | 0.64 |
| 0.8B G3 → 9B | chat8 | 1.00 | **1.00** | 0.98 | 1.00 | 0.96 | 0.98 |
| G1b G2 → 9B | chat8 | 1.00 | **1.00** | 0.98 | 1.00 | 0.97 | 0.97 |
| 0.8B G3 → Gtiny | chat8 | 1.00 | 0.07 | **0.95** | 0.97 | 0.95 | 0.98 |

**Where the receiver needs its query block** (Qwen-0.8B and Granite-1B), the other family's query
block restores retrieval: 0.92–0.96 on `chat8`/`list8`, 0.77–0.83 on `rev8`, from 0.00–0.30.
Every control stays at the floor:
- **Another block of the donor:** ≤ 0.39, except the query feeder next to it in 9B (G3, 0.79).
- **The query block without its linear mixers:** 0.11–0.38.
- **The bypass:** 0.10–0.31. That is despite fitting the missing write with R² 0.93: a linear
  function of each position's own residual cannot make the query.

The transplanted layers must *mix tokens*. What transfers is the linear-attention computation of
the query block, and it works across Gated DeltaNet ↔ Mamba-2 in both directions.

**Two receivers cannot test it**, for reasons exp 1 already exposed:
- 9B does not need its G4 at all when accuracy is the measure (removed: 1.00; Part XVI redundancy).
- In Granite-Tiny removal drops accuracy to 0.07, but the linear bypass alone already restores 0.95.
  Its query-block contribution is position-local enough to be linear.

**One donor fails:** Granite-Tiny's block into 0.8B (0.54, below its G3's 0.64), although Tiny's
query *vector* transfers fine in A (0.91). Tiny's block contains the mixture-of-experts MLPs, whose
routing sees mapped, off-distribution inputs. This is untested, and it is the one place where A
and B disagree.

## Seed replicates

Seeds 1 and 2 redraw the key split (`K_fit` / `K_test`), the fit dictionaries and the test
dictionaries (`scripts/run_exp26_27_seeds.sh`; `results/exp2{6,7}_*_seed{1,2}.json`):

| experiment | seed 0 | seed 1 | seed 2 | mean ± sd |
|---|---|---|---|---|
| A: query vector, 0.8B → G1b (ansA) | 0.98 | 0.98 | 0.97 | 0.980 ± 0.004 |
| A: query vector, G1b → 0.8B (ansA) | 0.91 | 0.86 | 0.94 | 0.903 ± 0.029 |
| B: query block, 0.8B G3 → G1b (acc) | 0.92 | 0.94 | 0.95 | 0.938 ± 0.014 |
| B: query block, G1b G2 → 0.8B (acc) | 0.95 | 0.93 | 0.90 | 0.925 ± 0.020 |

Across the three seeds the B controls stay at the floor:

| direction | block removed | best other donor block | query block, mixers zeroed | bypass |
|---|---|---|---|---|
| 0.8B → G1b | 0.01–0.02 | 0.28–0.39 | 0.18–0.28 | 0.20–0.24 |
| G1b → 0.8B | 0.22–0.30 | 0.27–0.32 | 0.23–0.28 | 0.21–0.28 |

## Reading

1. **The query is shared across families up to a linear map.** A query formed by Gated DeltaNet
   layers drives a Mamba-2 model's reader, and the reverse, on keys never used to fit the map.
   The query block is the donor representation closest to the other family's query: 25–100
   paired prompts, against 4–30× more for any other block.
2. **The query block's computation transplants, GnA-style, across families.** Granite-1B without
   its query block retrieves at 0.01. With Qwen-0.8B's three GDN layers in its place, linearly
   stitched and never trained, it retrieves at 0.92–0.94 (`rev8` 0.83); the reverse gives
   0.30 → 0.95. Other blocks, the same block without token mixing, and the best linear stand-in do
   not.
3. **What does not transfer is predictable from exp 1:** redundancy (9B), a query block that is
   effectively position-local (Tiny as receiver), and the MoE model as a block donor.

**Caveats.**
- Three seeds for the headline pair (0.8B ↔ G1b, `chat8`); one seed elsewhere.
- The ridge λ is chosen on held-out fit data.
- `M_in` fits only the tokens whose character spans match (61–84 % of a prompt: 48–72 of
  62–103). The wrapper tokens (Granite's system prompt) are extrapolated.
- Attention is recorded at the `chat8` reader, even for `rev8`, where reading moves later.
- The cross-family maps use a single dictionary template per run. Transfer across templates (fit
  on `chat8`, test on `list8`) was not tested.

---

# Part XXI: the Bind side — sufficiency and transfer (completes paper exp 2 and exp 3)

Scripts: `scripts/exp28_xfamily_circuit.py` (Bind-block and whole-circuit transplant),
`scripts/exp29_bind_transplant.py` (token-level Bind swaps, in-family and cross-family); shared
stitching code `src/stitch.py`; launcher `scripts/run_exp28_29.sh`. Results
`results/exp28_xcircuit_*.json`, `results/exp29_bind_*.json`. `chat8`, test keys (and, in exp29,
values) never seen when fitting any map; n = 200 test prompts.

Parts XVII and XX tested sufficiency and transfer for the **Query** stage only. Here the same two
tests are run on the **Bind** stage, and on the two stages together.

## A. Is the Bind block's write sufficient to re-bind an entry? (exp29, in-family)

Two token-level swaps, host and donor prompts of identical length:
- **value swap:** the host has entry t's value exchanged with entry d's. Write the donor's
  (unswapped) Bind-block output at t's *value* token. Success = the host answers t's original value.
- **key swap:** the donor has the keys of t and d exchanged. Write its Bind output at both *key*
  tokens. Success = the host, asked about key t, answers d's value.

Two controls: patch only the token **embedding** at those positions (Bind then recomputed), and
patch another block's write instead of Bind's.

| host | swap | Bind write | embedding only | any other block | no patch |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | value / key | **1.00 / 1.00** | 1.00 / 1.00 | ≤ 0.00 | 0.00 |
| Qwen3.5-9B | value / key | **0.98 / 0.98** | 1.00 / 1.00 | ≤ 0.00 | 0.00 |
| Granite-1B | value / key | **0.00 / 0.00** | 1.00 / 1.00 | ≤ 0.00 | 0.00 |
| Granite-Tiny | value / key | **0.10 / 0.01** | 0.99 / 1.00 | ≤ 0.01 | 0.00 |

The information the Bind stage handles, which value sits at an entry and which key it is filed
under, is **token identity**, and the token embedding alone carries all of it in every model.
- **In Qwen,** the Bind write carries it too: layer 0's write dominates the residual stream (Part VI).
- **In Granite,** the Bind write carries none of it. The identity reaches later layers through the
  embedding left in the residual stream, and rewriting the Bind block's output does not override it.

So Bind is "sufficient" only in the sense that an embedding patch is. It re-encodes identity; it
does not compute a binding that exists nowhere else.

## B. Does the Bind block transfer across families? (exp28 stage 1, exp29 cross-family)

**Block transplant.** The receiver's Bind block is removed and a donor block is stitched in, as in
Part XX-B.

| donor → receiver | seeds | intact | Bind removed | **donor Bind block** | any other donor block | donor Bind, mixers zeroed | linear bypass |
|---|---|---|---|---|---|---|---|
| 0.8B → G1b | 3 | 1.00 | 0.00 | 0.99–1.00 | 0.98–1.00 | 0.97–0.99 | 0.98–0.99 |
| 9B → G1b | 1 | 1.00 | 0.00 | 0.99 | 0.99–1.00 | 0.99 | 0.98 |
| G1b → 0.8B | 3 | 0.98–0.99 | 0.00 | 0.98–1.00 | 0.98–0.99 | 0.96–0.99 | 0.99–1.00 |
| G1b → 9B | 1 | 1.00 | 0.00 | 1.00 | 0.98–0.99 | 0.99 | 1.00 |

Removing the Bind block destroys retrieval in every receiver (0.00, 9B included). But **anything**
put in its place restores it: the donor's Bind block, any other donor block, the donor's Bind block
without its token mixing, or a per-position linear map. For retrieval, the Bind slot needs a
position-local re-encoding of each token and nothing more. This is the opposite of the Query slot
(Part XX-B: only the donor's query block, with its mixing, restores it).

The Bind block's linear attention does matter for **language modelling**. WikiText perplexity of
the same receivers (intact G1b 14, 0.8B 21, 9B 11):

| donor → receiver | donor Bind block | other donor blocks | donor Bind, mixers zeroed | linear bypass |
|---|---|---|---|---|
| 0.8B → G1b | **32–33** | 37–77 | 229–251 | 46–48 |
| 9B → G1b | 25 | 23–27 | 66 | 48 |
| G1b → 0.8B | **28** | 37–44 | 116–119 | 71–73 |
| G1b → 9B | **19** | 24–29 | 87 | 45 |

The donor's own Bind block is the best stand-in for general text (except 9B → G1b, a tie), and
without its token mixing it is by far the worst. Bind's linear attention is local context for
prediction, not information retrieval needs.

**Token-level transfer (exp29).** Map the donor's Bind write at a value/key token into the
receiver's Bind write at the same token, fitted on aligned tokens of fit dictionaries and WikiText,
with test keys and values unseen. Compare with the same map from the donor's *static embedding*.

| donor → host | swap | Bind write, mapped | static embedding, mapped | shuffled map / mean |
|---|---|---|---|---|
| G1b → 0.8B | value / key | 0.81 / 0.74 | **1.00 / 0.95** | 0.04–0.05 |
| G1b → 9B | value / key | 0.55 / 0.34 | 0.51 / 0.42 | 0.00 |
| 0.8B, 9B → G1b; 0.8B → Gtiny | both | 0.00–0.13 | 0.00–0.10 | ≤ 0.02 |

Into Qwen, the Bind write transfers, but a map from the donor's context-free embedding does as well
or better. Into Granite nothing transfers at the Bind block, since not even the host's own Bind
write is sufficient there (A). What crosses families at the Bind stage is lexical identity, not a
computation.

## C. The whole circuit (exp28 stage 2)

The receiver's Bind **and** Query blocks are both replaced. Bind is stitched from the donor's Bind
block (or the linear bypass). The Query-stage maps are fitted with that Bind replacement already in
place.

| donor → receiver | seeds | **donor Bind + donor Query** (reader attn) | donor Bind + Query mixers zeroed | donor Bind + bypass Query | donor Bind + Query removed | bypass Bind + donor Query |
|---|---|---|---|---|---|---|
| 0.8B → G1b | 3 | **0.90 ± 0.02** (0.32–0.35) | 0.19–0.23 | 0.17–0.23 | 0.01 | 0.85–0.92 |
| 9B → G1b | 1 | **0.92** (0.44) | 0.28 | 0.26 | 0.03 | 0.91 |
| G1b → 0.8B | 3 | **0.95 ± 0.02** (0.51) | 0.24–0.32 | 0.23–0.31 | 0.23–0.28 | 0.93–0.97 |
| G1b → 9B | 1 | 1.00 | 0.97 | 0.99 | 0.99 | 0.99 (9B needs neither: redundancy) |

Replacing both stages of the circuit with the other family's layers, stitched and untrained, keeps
retrieval at 0.90–0.95 (9B → G1b 0.92). Whatever is in the Bind slot, success is decided by the
Query slot: donor Query 0.85–0.97, anything else 0.01–0.46.

## Reading

1. **The circuit is asymmetric.**
   - **Bind:** necessary everywhere (removal → 0.00), but what it provides is token identity,
     re-encoded per position. The embedding alone re-binds an entry. Any block, or a linear map,
     can fill its slot for retrieval. Across families it transfers no better than the static
     embedding.
   - **Query:** the only stage where linear attention performs a specific, transferable computation
     (Parts XVII, XX). Only the donor's query layers, with their token mixing, can fill that slot.
2. **Bind's linear attention serves language modelling, not retrieval.** Its token mixing is
   dispensable for retrieval, but zeroing it multiplies the stitched receiver's perplexity 2.6–7.5×,
   and the donor's own Bind block is the best stand-in.
3. **The whole circuit transfers.** Both stages from the other family, with no training, retrieve at
   0.90–0.95, and the result is carried entirely by the Query stage.

**Caveats.**
- Three seeds for 0.8B ↔ G1b, one for the 9B pairs.
- `chat8` only.
- The key swap patches two positions and the value swap one.
- In Granite the Bind write is defined as the mixer outputs of layers 0–4 (its MLPs recompute).

---

# Part XXII: the rank of the linear-attention memory, by circuit role

Scripts: `scripts/state_rank.py` (dumps the singular values of every head's state) and
`scripts/state_rank_report.py` (ranks, per-head CSV, heatmaps, markdown tables; effective rank and
σ₁ share), adapted from LinearSwap `tools/state_rank.py` / `tools/state_rank_report.py`; circuit roles
from `src/circuit.py`. Launcher `scripts/run_state_rank.sh`. Results `results/state_rank/<name>.json`,
reports `results/state_rank/report_{dclm,longbench,dict,dict_end}.md`, per-head CSVs alongside,
figures `figures/state_rank{,_longbench,_dict,_dict_end}.png`.

**What is measured.** Each head's recurrent state is a matrix: Gated DeltaNet key × value = 128 × 128
(Qwen3.5); Mamba-2 head × state = 64 × 128 (Granite-4.0-H, so full rank is 64). It is read from the HF
cache and summarised by its median over the batch:
- **numerical rank:** singular values above n·eps_fp32·σ_max;
- **significant rank:** singular values above 1 % of σ_max;
- **effective rank:** exp of the entropy of σ/Σσ;
- **σ₁ share:** σ₁²/Σσ².

Following the LinearSwap blog, the numerical rank is close to uninformative (almost every state
eventually counts as full rank). The effective rank is the measure to read.

**Sources** (fp32, all four models):
- **DCLM:** 8 packed sequences, documents separated by each model's own EOS. This is LinearSwap's setting
  and reads states at 64 / 128 / 256 / 1K / 4K / 16K tokens. Each 16K window holds 11–19 documents.
- **LongBench:** 8 single long documents (LongBench-v2, Single-Document QA, English) at the same marks.
- **dict:** the 100 `chat16` dictionary prompts (≈94 tokens), read at the end of the dictionary and at
  the final token.

## 1. Effective rank by circuit role

Median effective rank / median σ₁ share:

| model | source (mark) | **Bind** | feeder | **Query** | after the reader |
|---|---|---|---|---|---|
| 0.8B | DCLM (16K) | **5.1 / 0.96** | 10.5 / 0.88 | 7.7 / 0.87 | 8.1 / 0.87 |
| | LongBench (16K) | **5.7 / 0.94** | 12.2 / 0.85 | 9.8 / 0.85 | 10.9 / 0.82 |
| | dict (final) | **3.3 / 0.97** | 6.1 / 0.92 | 5.9 / 0.87 | 6.1 / 0.89 |
| 9B | DCLM (16K) | **2.3 / 0.98** | 14.0 / 0.82 | 17.9 / 0.78 | 11.9 / 0.84 |
| | LongBench (16K) | **2.4 / 0.98** | 17.8 / 0.78 | 21.8 / 0.72 | 19.1 / 0.77 |
| | dict (final) | **1.9 / 0.98** | 9.8 / 0.86 | 9.6 / 0.81 | 9.4 / 0.82 |
| G1b | DCLM (16K) | **2.7 / 0.96** | 3.6 / 0.95 | 3.3 / 0.96 | 2.4 / 0.97 |
| | LongBench (16K) | **2.8 / 0.95** | 3.9 / 0.94 | 3.5 / 0.96 | 2.8 / 0.97 |
| | dict (final) | **2.5 / 0.96** | 3.0 / 0.96 | 2.8 / 0.97 | 2.3 / 0.98 |
| Gtiny | DCLM (16K) | **1.6 / 1.00** | 2.4 / 0.99 | 3.6 / 0.97 | 3.4 / 0.96 |
| | LongBench (16K) | **1.6 / 1.00** | 2.5 / 0.99 | 3.7 / 0.97 | 4.0 / 0.95 |
| | dict (final) | **1.6 / 1.00** | 2.0 / 0.99 | 3.3 / 0.97 | 3.0 / 0.97 |

## 2. Rank against length (median effective rank over all states)

| model | source | 64 | 128 | 256 | 1K | 4K | 16K |
|---|---|---|---|---|---|---|---|
| 0.8B | DCLM (packed) | 6.5 | 6.8 | 7.6 | 8.6 | 8.5 | 8.1 |
| | LongBench (one document) | 6.3 | 7.2 | 7.6 | 8.1 | 9.2 | **10.2** |
| 9B | DCLM | 10.2 | 11.1 | 11.9 | 13.0 | 12.4 | 11.5 |
| | LongBench | 10.0 | 12.2 | 12.5 | 13.3 | 15.3 | **16.4** |
| G1b | DCLM / LongBench | 3.0 / 3.2 | 3.3 / 3.2 | 3.4 / 3.3 | 3.6 / 3.2 | 3.1 / 3.5 | 3.0 / 3.2 |
| Gtiny | DCLM / LongBench | 2.8 / 2.9 | 2.9 / 3.0 | 3.0 / 3.1 | 3.3 / 3.0 | 2.8 / 3.1 | 2.8 / 3.0 |

## Reading

1. **The Bind layers keep the lowest-rank memory, in every model and on every source.**
   - Effective rank is 1.6–5.7, against 2.0–21.8 elsewhere.
   - σ₁ carries 94–100 % of the energy, so the Bind state is close to one outer product.
   - In 9B the gap is sixfold: 2.3 against 11.9–17.9 on DCLM.

   This is the state-side picture of Part XXI. Bind's job for retrieval is a per-position
   re-encoding of token identity; a linear map can stand in for it, and its token mixing is
   dispensable for retrieval. Its recurrent memory accordingly holds almost a single direction.
2. **The Query layers are not distinguished by memory rank.**
   - In 9B they hold the highest-rank states on text (17.9 / 21.8).
   - On the dictionary task they are level with the feeders (9.6 vs 9.8), and in 0.8B they sit
     below them.

   What the Query block contributes (Parts XVII, XX) is a computation at the final position, not a
   larger memory. This is consistent with Part II, where the GDN state does not hold the binding.
3. **The families differ by an order of magnitude.**
   - Mamba-2 states carry an effective rank of about 3 out of 64 everywhere (σ₁ share 0.95–1.00).
   - Gated DeltaNet carries 5–22 out of 128.
   - Granite's numerical rank saturates almost immediately (thousands of states at 64 after
     ~100 tokens), so for Granite only the effective rank is informative.
4. **Length.** The plateau after ~1K tokens on packed DCLM (LinearSwap's setting, reproduced here:
   0.8B 8.6 → 8.5 → 8.1) comes from switching documents.
   - Within one long document, Qwen's effective rank is still rising at 16K: 0.8B +11 %, 9B +7 %
     from 4K to 16K.
   - Granite is flat on both sources.
   - We stop at 16K. A 32K mark on LongBench would show where Qwen saturates. It does not affect the
     role comparison above, which is the same at every source.
5. **One dead head.** Granite-Tiny layer 12, head 24 has an exactly zero state on 7 of 8 sequences
   (both sources). It is counted as rank 0.

**Caveats.**
- 8 sequences per text source (LinearSwap's setting) and 100 dictionary prompts.
- Per-head medians.
- Marks count each model's own tokens, so 16K covers slightly different text in Qwen and Granite.
- Singular values are computed in fp32 LAPACK on the CPU. cuSOLVER's GPU path is pathologically slow
  for Mamba-2's 64 × 128 states. CPU and GPU agree to 2e-5 σ_max, and numerical ranks are identical
  to an fp64 reference.
