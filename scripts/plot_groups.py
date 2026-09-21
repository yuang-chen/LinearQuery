"""Figure 9: GDN layer groups -- ablate each block of three, and keep-only conditions."""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default="results/exp13_groups.json")
ap.add_argument("--out", default="figures/fig9_groups.png")
a = ap.parse_args()

J = json.load(open(a.inp))
rows, GR = J["rows"], J["groups"]
sel = lambda k: [r for r in rows if r["kind"] == k]
BP, BA = J["base_ppl"], J["base_acc"]
rnd = next(r for r in rows if r["kind"] == "random3_summary")
LBL = [f"G{i}\n{g}" for i, g in enumerate(GR)]

fig, axes = plt.subplots(2, 2, figsize=(14, 9.4))

# --- A. group ablation, perplexity ------------------------------------------
ax = axes[0, 0]
ga = sorted(sel("group_ablate"), key=lambda r: r["group"])
x = np.arange(6)
ax.bar(x, [r["ppl"] for r in ga], .62,
       color=["#c44e52"] + ["#4c72b0"] * 5)
ax.axhline(BP, ls="--", c="k", lw=1.3, label=f"intact ({BP:.1f})")
band = np.exp(J["base_nll"] + rnd["mean"])
ax.axhspan(np.exp(J["base_nll"] + rnd["mean"] - rnd["std"]),
           np.exp(J["base_nll"] + rnd["mean"] + rnd["std"]),
           color="k", alpha=.12, label="3 random GDN layers (mean ± sd)")
ax.axhline(band, c="k", lw=1, alpha=.5)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(LBL, fontsize=8)
ax.set_ylabel("WikiText-2 perplexity")
ax.set_title("A. Remove one group of 3 consecutive GDN layers\n"
             "Only the first group is special; G2/G3 are below random", fontsize=10.5)
ax.legend(fontsize=8)
for i, r in enumerate(ga):
    ax.text(i, r["ppl"] * 1.35, f"{r['ppl']:,.0f}" if r["ppl"] > 100 else f"{r['ppl']:.1f}",
            ha="center", fontsize=8)
ax.set_ylim(top=ax.get_ylim()[1] * 12)

# --- B. group ablation, retrieval -------------------------------------------
ax = axes[0, 1]
ax.bar(x - .2, [r["acc"] for r in ga], .4, color="#4c72b0", label="retrieval accuracy")
ax2 = ax.twinx()
ax2.bar(x + .2, [r["gap"] for r in ga], .4, color="#c44e52", label="answer logit gap")
ax.axhline(BA, ls="--", c="#4c72b0", lw=1.2, alpha=.7)
ax2.axhline(J["base_gap"], ls="--", c="#c44e52", lw=1.2, alpha=.7)
ax.set_xticks(x); ax.set_xticklabels(LBL, fontsize=8)
ax.set_ylabel("retrieval accuracy"); ax2.set_ylabel("$D_{clean}-D_{corrupt}$")
ax.set_ylim(0, 1.1); ax2.set_ylim(0, 23)
ax.set_title("B. The same ablations on the dictionary task\n"
             "G3 is near-free for perplexity but breaks retrieval", fontsize=10.5)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
ax.annotate("softmax reader\nL15 reads G3", xy=(3, .15), xytext=(3.1, .55), fontsize=7.5,
            ha="center", arrowprops=dict(arrowstyle="->", lw=1))

# --- C. keep-only conditions -------------------------------------------------
ax = axes[1, 0]
k1 = sel("keep_one")
kg = sorted(sel("group_keep"), key=lambda r: r["group"])
allz = next(r for r in sel("ref") if r["name"] == "all 18 GDN mixers zeroed")
zero0 = None
names = [("keep only\nlayer 0", next(r for r in k1 if r["keep"] == 0)["ppl"], "#c44e52")]
names += [(f"keep only\nlayer {r['keep']}", r["ppl"], "0.75") for r in k1 if r["keep"] in (1, 2, 8, 16)]
names += [(f"keep only\n{['G0','G1','G2','G3','G4','G5'][r['group']]}", r["ppl"],
           "#dd8452" if r["group"] == 0 else "0.82") for r in kg if r["group"] in (0, 2, 5)]
names += [("all GDN\nmixers zeroed", allz["ppl"], "0.35")]
xs = np.arange(len(names))
ax.bar(xs, [v for _, v, _ in names], .62, color=[c for _, _, c in names])
ax.axhline(BP, ls="--", c="k", lw=1.3, label=f"intact ({BP:.1f})")
ax.set_yscale("log"); ax.set_xticks(xs)
ax.set_xticklabels([n for n, _, _ in names], fontsize=7.2)
ax.set_ylabel("WikiText-2 perplexity")
ax.set_title("C. Keeping only layer 0 beats keeping only any other\nlayer by ~50x -- but is still far from usable",
             fontsize=10.5)
ax.legend(fontsize=8)
for i, (_, v, _) in enumerate(names):
    ax.text(i, v * 1.5, f"{v:,.0f}", ha="center", fontsize=7.5)
ax.set_ylim(top=ax.get_ylim()[1] * 15)

# --- D. keep first k / cumulative -------------------------------------------
ax = axes[1, 1]
kf = sorted(sel("keep_first_k"), key=lambda r: r["k"])
ax.plot([r["k"] for r in kf], [r["ppl"] for r in kf], "o-", color="#4c72b0", ms=5,
        label="keep the first $k$ GDN layers")
cf = sorted(sel("cum_front"), key=lambda r: r["k"])
ax.plot([18 - 3 * r["k"] for r in cf], [r["ppl"] for r in cf], "s--", color="#c44e52", ms=5,
        label="remove groups from the front")
ax.axhline(BP, ls="--", c="k", lw=1.2)
ax.set_yscale("log"); ax.set_xlabel("number of GDN layers kept")
ax.set_ylabel("WikiText-2 perplexity")
ax.set_title("D. Late GDN layers are nearly free to drop;\nearly ones are not (note the non-monotonicity)",
             fontsize=10.5)
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.3, which="both")
ax3 = ax.twinx()
ax3.plot([r["k"] for r in kf], [r["acc"] for r in kf], "^:", color="#55a868", ms=5,
         label="retrieval accuracy")
ax3.set_ylabel("retrieval accuracy", color="#55a868")
ax3.tick_params(axis="y", colors="#55a868"); ax3.set_ylim(0, 1.1)
ax3.legend(fontsize=8, loc="center left")

fig.tight_layout(); fig.savefig(a.out, dpi=150)
print("wrote", a.out)
