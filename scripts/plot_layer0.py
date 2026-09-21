"""Figure 5: what GDN layer 0 computes -- information ladder, mechanism split, retrieval."""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default="results/exp9_layer0.json")
ap.add_argument("--out", default="figures/fig5_layer0.png")
a = ap.parse_args()

D = json.load(open(a.inp))
rows = D["rows"]
by = {r["name"]: r for r in rows if "name" in r}
base = by["intact (rung 4)"]["ppl"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.35, 1.1, 1]})

# --- A. information ladder ---------------------------------------------------
ax = axes[0]
LAY = [0, 1, 2, 4]
RUNGS = [("zero", "0: zeros\n(no info, no scale)", "#8c2d04"),
         ("global_mean", "1: global mean\n(scale only, no info)", "#cc4c02"),
         ("rand_dir", "2: norm-matched\nrandom direction", "#fe9929"),
         ("token_mean", "3: token-type mean\n(identity only, no context)", "#41ab5d"),
         ("token_sample", "3b: same token,\nother context", "#238b45")]
w = 0.17
for j, (rung, lab, c) in enumerate(RUNGS):
    v = [next(r["ppl"] for r in rows if r.get("layer") == L and r.get("rung") == rung)
         for L in LAY]
    ax.bar(np.arange(len(LAY)) + (j - 2) * w, v, w, label=lab, color=c)
ax.axhline(base, ls="--", c="k", lw=1.3, label=f"4: intact (PPL {base:.1f})")
ax.set_yscale("log")
ax.set_xticks(np.arange(len(LAY)))
ax.set_xticklabels([f"GDN layer {L}" for L in LAY])
ax.set_ylabel("WikiText-2 perplexity (held-out)")
ax.set_title("A. Information ladder: replace the mixer output\nwith progressively more informative surrogates",
             fontsize=10.5)
ax.legend(fontsize=7, ncol=2, loc="upper right")
ax.grid(alpha=.3, axis="y", which="both")
ax.set_ylim(top=ax.get_ylim()[1] * 40)

# --- B. mechanism split + channel sweep --------------------------------------
ax = axes[1]
names = [("L0 mixer <- rung 0 zeros", "whole mixer output removed", "#8c2d04"),
         ("L0 conv window forced to 1 tap", "conv window 4 -> 1 tap", "#4c72b0"),
         ("L0 recurrent state reset every token", "recurrence removed", "#55a868"),
         ("L0 recurrence reset every 8 tokens", "recurrence limited to 8 tok", "#8fbc8f")]
v = [by[n]["ppl"] for n, _, _ in names]
ax.barh(np.arange(len(names))[::-1], v, .6, color=[c for _, _, c in names])
ax.axvline(base, ls="--", c="k", lw=1.3)
ax.text(base * 1.12, 3.35, f"intact {base:.1f}", fontsize=8)
ax.set_yticks(np.arange(len(names))[::-1])
ax.set_yticklabels([l for _, l, _ in names], fontsize=8.5)
ax.set_xscale("log"); ax.set_xlabel("WikiText-2 perplexity")
ax.set_title("B. Which part of layer 0 matters?\nSequence mixing is nearly irrelevant", fontsize=10.5)
for i, x in enumerate(v):
    ax.text(x * 1.15, len(names) - 1 - i, f"{x:,.0f}" if x > 100 else f"{x:.1f}",
            va="center", fontsize=8)
ax.set_xlim(right=max(v) * 12)

# --- C. retrieval ladder -----------------------------------------------------
ax = axes[2]
rl = next(r for r in rows if r["name"] == "retrieval_ladder")
modes = [("zero", "zeros"), ("global_mean", "global\nmean"),
         ("token_mean", "token-type\nmean"),
         ("token_sample", "same token,\nother ctx"), ("intact", "intact")]
acc = [rl[m]["acc"] for m, _ in modes]
gap = [rl[m]["gap"] for m, _ in modes]
x = np.arange(len(modes))
ax.bar(x - .2, acc, .4, color="#4c72b0", label="accuracy (clean & corrupted)")
ax2 = ax.twinx()
ax2.bar(x + .2, gap, .4, color="#c44e52", label="$D_{clean}-D_{corrupt}$")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in modes], fontsize=8.5)
ax.set_ylabel("retrieval accuracy"); ax2.set_ylabel("answer logit gap")
ax.set_ylim(0, 1.15); ax2.set_ylim(0, 26)
ax.set_title("C. The retrieval circuit needs only\nlayer 0's token identity", fontsize=10.5)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

fig.tight_layout(); fig.savefig(a.out, dpi=150)
print("wrote", a.out)
