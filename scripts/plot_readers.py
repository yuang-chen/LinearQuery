"""Figure 2: reader screen, path opening/blocking, and head-level Q/K/V localisation."""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--readers", default="results/exp4_readers.json")
ap.add_argument("--heads", default="results/exp4b_headpath.json")
ap.add_argument("--out", default="figures/fig2_readers.png")
a = ap.parse_args()

R = json.load(open(a.readers))
rows = R["rows"]
ATTN = [3, 7, 11, 15, 19, 23]
agg = {}
for r in rows:
    agg.setdefault((r["kind"], r["layer"], r["head"]), []).extend(r["rec"])
ms = lambda v: (np.mean(v), np.std(v) / len(v) ** .5)
base = ms(agg[("writer only", R["meta"]["writer_layer"], None)])[0]

hagg = {}
for r in json.load(open(a.heads))["rows"]:
    hagg.setdefault((r["layer"], r["head"], r["kind"]), []).extend(r["rec"])

fig = plt.figure(figsize=(14.5, 8.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.62, wspace=0.28)

# --- A. module blocking screen -------------------------------------------------
ax = fig.add_subplot(gs[0, :])
layers = list(range(1, 24))
mix = np.array([ms(agg[("writer+block mixer", L, None)])[0] for L in layers])
mixe = np.array([ms(agg[("writer+block mixer", L, None)])[1] for L in layers])
mlp = np.array([ms(agg[("writer+block mlp", L, None)])[0] for L in layers])
ctl = np.array([ms(agg[("cleanblock mixer", L, None)])[0] for L in layers])
x = np.arange(len(layers))
cols = ["#c44e52" if L in ATTN else "#4c72b0" for L in layers]
ax.bar(x - .22, mix, .44, yerr=mixe, capsize=2, color=cols, label="block token mixer (red = softmax layer)")
ax.bar(x + .22, mlp, .44, color="#55a868", alpha=.8, label="block MLP")
ax.plot(x - .22, ctl, "kv", ms=5, ls="none", label="control: same mixer block, clean run (no writer patch)")
ax.axhline(base, ls="--", c="crimson", lw=1.2, label=f"writer patch, nothing blocked ({base:+.2f})")
ax.set_xticks(x); ax.set_xticklabels(layers)
ax.set_xlabel("layer"); ax.set_ylabel("recovery of logit difference")
ax.set_title("A. Reader screen: block each downstream module inside the writer-patched run "
             "(lower = module mediates the writer's effect)", fontsize=10.5)
ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.20))
ax.set_ylim(0, 1.32)

# --- B. layer-level path open / block ------------------------------------------
ax = fig.add_subplot(gs[1, 0])
kinds = ["pathopen K@src", "pathopen V@src", "pathopen Q@final", "pathopen KV@src"]
w = 0.2
for j, k in enumerate(kinds):
    m = [ms(agg[(k, L, None)])[0] for L in ATTN]
    e = [ms(agg[(k, L, None)])[1] for L in ATTN]
    ax.bar(np.arange(len(ATTN)) + (j - 1.5) * w, m, w, yerr=e, capsize=2,
           label=k.replace("pathopen ", ""))
ax.axhline(0, c="k", lw=.8)
ax.set_xticks(np.arange(len(ATTN))); ax.set_xticklabels(ATTN)
ax.set_xlabel("softmax layer"); ax.set_ylabel("recovery")
ax.set_title("B. Path opening: transplant one input\nfrom the writer-patched run", fontsize=10.5)
ax.legend(fontsize=8, title="input transplanted", title_fontsize=8)

ax = fig.add_subplot(gs[1, 1])
kinds = ["pathblock K@src", "pathblock V@src", "pathblock KV@src"]
for j, k in enumerate(kinds):
    m = [ms(agg[(k, L, None)])[0] for L in ATTN]
    e = [ms(agg[(k, L, None)])[1] for L in ATTN]
    ax.bar(np.arange(len(ATTN)) + (j - 1) * 0.27, m, 0.27, yerr=e, capsize=2,
           label=k.replace("pathblock ", ""))
ax.axhline(base, ls="--", c="crimson", lw=1.2, label="no block")
ax.set_xticks(np.arange(len(ATTN))); ax.set_xticklabels(ATTN)
ax.set_xlabel("softmax layer"); ax.set_ylabel("recovery")
ax.set_title("C. Path blocking: restore one input\nto its corrupted value", fontsize=10.5)
ax.legend(fontsize=8, title="input restored", title_fontsize=8)

# --- D. head-level Q/K/V -------------------------------------------------------
ax = fig.add_subplot(gs[1, 2])
heads = sorted({(L, h) for (L, h, k) in hagg})
kinds = [("head-path V@src", "V @ source"), ("head-path K@src", "K @ source"),
         ("head-path Q@final", "Q @ final"), ("head-path QKV@all", "Q+K+V, all positions")]
w = 0.2
for j, (k, lab) in enumerate(kinds):
    m = [ms(hagg[(L, h, k)])[0] for (L, h) in heads]
    e = [ms(hagg[(L, h, k)])[1] for (L, h) in heads]
    ax.bar(np.arange(len(heads)) + (j - 1.5) * w, m, w, yerr=e, capsize=2, label=lab)
ax.axhline(0, c="k", lw=.8)
ax.set_xticks(np.arange(len(heads)))
ax.set_xticklabels([f"L{L}H{h}" for L, h in heads], fontsize=9)
ax.set_ylabel("recovery through this head")
ax.set_title("D. Head-level localisation of the input\nthat carries the effect", fontsize=10.5)
ax.legend(fontsize=8)

fig.savefig(a.out, dpi=150, bbox_inches="tight")
print("wrote", a.out)
