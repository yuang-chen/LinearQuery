"""Figure 6: is GDN layer 0 an affine re-embedding?  Fit quality, rank, and generalisation."""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--fit", default="results/exp10_affine.json")
ap.add_argument("--ood", default="results/exp10b_affine_ood.json")
ap.add_argument("--out", default="figures/fig6_affine.png")
a = ap.parse_args()

F = json.load(open(a.fit))
rows = F["rows"]
by = {r["name"]: r for r in rows if "name" in r}
O = {r["name"]: r for r in json.load(open(a.ood))["rows"]}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1.15, 1.1]})

# --- A. rank sweep -----------------------------------------------------------
ax = axes[0]
rk = sorted([(r["rank"], r["ppl"]) for r in rows if r.get("rung") == "affine"])
ax.plot([r for r, _ in rk], [p for _, p in rk], "o-", color="#1f77b4", lw=2, ms=6,
        label="affine  $A_r\\cdot$embed$(x)+b$")
ax.axhline(by["intact"]["ppl"], ls="--", c="k", lw=1.3,
           label=f"intact ({by['intact']['ppl']:.1f})")
ax.axhline(by["token-type mean (rung 3 ceiling)"]["ppl"], ls="-.", c="#2ca02c", lw=1.4,
           label=f"token-type lookup ceiling ({by['token-type mean (rung 3 ceiling)']['ppl']:.0f})")
ax.axhline(by["global mean (no token info)"]["ppl"], ls=":", c="crimson", lw=1.3,
           label=f"no token info ({by['global mean (no token info)']['ppl']:,.0f})")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("rank of $A$"); ax.set_ylabel("WikiText-2 perplexity (held-out split)")
ax.set_title("A. The map is affine but full-rank\n(low-rank truncation fails)", fontsize=10.5)
ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=.3, which="both")

# --- B. variance decomposition + ridge selection ------------------------------
ax = axes[1]
bars = [("token identity\n(between-type / total)", F["r2_identity"], "#4c72b0"),
        ("affine map of embedding\n(of total variance)", F["r2_affine"], "#55a868"),
        ("affine map\n(of the token-wise part)", F["r2_affine_of_identity"], "#8fbc8f"),
        ("affine map, token types\nheld out of the fit", F["r2_heldout_types"], "#c44e52")]
ax.bar(range(len(bars)), [v for _, v, _ in bars], .6, color=[c for _, _, c in bars])
for i, (_, v, _) in enumerate(bars):
    ax.text(i, v + .015, f"{v:.3f}", ha="center", fontsize=9)
ax.set_xticks(range(len(bars)))
ax.set_xticklabels([n for n, _, _ in bars], fontsize=7.8)
ax.set_ylabel("$R^2$ of layer-0 mixer output")
ax.set_ylim(0, 1.08)
ax.set_title(f"B. Variance decomposition\n({F['n_types']:,} token types, ridge {F['best_ridge']:.0e})",
             fontsize=10.5)

# --- C. generalisation across distribution ------------------------------------
ax = axes[2]
groups = ["WikiText eval,\nWikiText-fit $A$", "task prompts,\nWikiText-fit $A$",
          "task prompts,\ncombined-fit $A$"]
keys = ["WikiText eval, WikiText-fit A", "task prompts, WikiText-fit A",
        "task prompts, combined-fit A"]
err = [O[k]["rel_err_all"] for k in keys]
x = np.arange(3)
ax.bar(x - .2, err, .4, color="#c44e52", label="affine residual  $\\|o-\\hat o\\|/\\|o\\|$")
sp = [O[k]["rel_err_special"] for k in keys]
ax.plot(x - .2, [s if s is not None else np.nan for s in sp], "kv", ms=6, ls="none",
        label="…on chat-template tokens")
ax2 = ax.twinx()
accs = [np.nan, O["sub affine fitted on WikiText"]["acc"],
        O["sub affine fitted on WikiText + task"]["acc"]]
ax2.bar(x + .2, accs, .4, color="#4c72b0", label="retrieval accuracy under substitution")
ax2.axhline(O["sub intact"]["acc"], ls="--", c="#4c72b0", lw=1.2, alpha=.7)
ax2.text(2.42, O["sub intact"]["acc"] + .02, "intact", fontsize=7, color="#4c72b0", ha="right")
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8)
ax.set_ylabel("relative residual"); ax2.set_ylabel("retrieval accuracy")
ax2.set_ylim(0, 1.15); ax.set_ylim(0, max(max(err), max(s for s in sp if s)) * 1.35)
ax.set_title("C. The retrieval failure was extrapolation,\nnot non-linearity", fontsize=10.5)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper center")


fig.tight_layout(); fig.savefig(a.out, dpi=150)
print("wrote", a.out)
