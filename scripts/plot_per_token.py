"""Figure 7: per-token NLL damage -- where layer-0 surrogates lose their nats."""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default="results/exp11_per_token.json")
ap.add_argument("--raw", default="results/exp11_per_token_raw.npz")
ap.add_argument("--out", default="figures/fig7_per_token.png")
a = ap.parse_args()

J = json.load(open(a.inp))
rows = J["rows"]
Z = np.load(a.raw, allow_pickle=True)
CONDS = [("token_local", "layer 0 made token-local\n(conv 1 tap + no recurrence)", "#4c72b0"),
         ("affine", "affine  $A\\cdot$embed$(x)+b$", "#c44e52"),
         ("token_mean", "token-type mean lookup", "#dd8452")]
base = Z["base"]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), gridspec_kw={"width_ratios": [1, 1.25, 1.1]})

# --- A. overall + concentration ----------------------------------------------
ax = axes[0]
ov = {r["cond"]: r for r in rows if r["kind"] == "overall"}
co = {r["cond"]: r for r in rows if r["kind"] == "concentration"}
keys = [k for k, _, _ in CONDS] + ["conv1tap", "norecur"]
labs = [l.replace("\n", " ") for _, l, _ in CONDS] + ["conv 1 tap only", "no recurrence only"]
ppl = [ov[k]["ppl"] for k in keys]
y = np.arange(len(keys))[::-1]
ax.barh(y, ppl, .6, color=[c for _, _, c in CONDS] + ["#8fbc8f", "#b0c4b1"])
ax.axvline(np.exp(J["base_nll"]), ls="--", c="k", lw=1.3)
ax.text(np.exp(J["base_nll"]) * 1.1, y[0] + .45, f"intact {np.exp(J['base_nll']):.1f}", fontsize=8)
for i, p in enumerate(ppl):
    ax.text(p * 1.1, y[i], f"{p:.1f}", va="center", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("WikiText-2 perplexity")
ax.set_xlim(right=max(ppl) * 6)
ax.set_title("A. Recomputing layer 0 token-locally costs\nfar less than any context-free surrogate",
             fontsize=10.5)

# --- B. damage by target-token category --------------------------------------
ax = axes[1]
cat = [r for r in rows if r["kind"] == "category"]
order = [c["category"] for c in sorted(cat, key=lambda r: -r["share"])]
x = np.arange(len(order))
w = 0.26
for j, (k, lab, c) in enumerate(CONDS):
    v = [next(r[f"d_{k}"] for r in cat if r["category"] == o) for o in order]
    ax.bar(x + (j - 1) * w, v, w, label=lab, color=c)
shares = {r["category"]: r["share"] for r in cat}
ax.set_xticks(x)
short = {"word-initial": "word\ninitial", "word-continuation": "word\ncontinuation",
         "punctuation": "punct.", "whitespace/newline": "space/\nnewline",
         "digit": "digit", "other": "other"}
ax.set_xticklabels([f"{short[o]}\n{shares[o]:.0%}" for o in order], fontsize=8)
ax.axhline(0, c="k", lw=.8)
ax.set_ylabel("mean $\\Delta$NLL (nats)")
ax.set_title("B. Damage by target-token category\n(word continuations are NOT preferentially hurt)",
             fontsize=10.5)
ax.legend(fontsize=7.5)

# --- C. damage vs baseline difficulty and token frequency ---------------------
ax = axes[2]
dif = sorted([r for r in rows if r["kind"] == "difficulty"], key=lambda r: r["decile"])
for k, lab, c in CONDS:
    ax.plot([r["base"] for r in dif], [r[f"d_{k}"] for r in dif], "o-", color=c, ms=4, label=lab)
ax.set_xlabel("intact NLL of the position (nats, decile means)")
ax.set_ylabel("mean $\\Delta$NLL (nats)")
ax.set_title("C. Damage grows with how hard the\nposition already was -- it is broad, not focal",
             fontsize=10.5)
ax.grid(alpha=.3)
ax.legend(fontsize=7.5, loc="upper left")
ins = ax.inset_axes([0.55, 0.12, 0.42, 0.42])
fr = sorted([r for r in rows if r["kind"] == "frequency"], key=lambda r: r["lo"])
xf = np.arange(len(fr))
for k, lab, c in CONDS:
    ins.plot(xf, [r[f"d_{k}"] for r in fr], "o-", color=c, ms=3)
ins.set_xticks(xf)
ins.set_xticklabels([f"{r['lo']}" if r["hi"] < 10**8 else f"{r['lo']}+" for r in fr], fontsize=6)
ins.tick_params(labelsize=6)
ins.set_title("by target frequency\n(train count)", fontsize=6.5)

fig.tight_layout(); fig.savefig(a.out, dpi=150)
print("wrote", a.out)
