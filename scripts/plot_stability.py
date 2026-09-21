"""Figure 3: stability of the fixed candidates across conditions."""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default="results/exp5_stability.json")
ap.add_argument("--out", default="figures/fig3_stability.png")
a = ap.parse_args()

rows = json.load(open(a.inp))["rows"]
agg = {}
for r in rows:
    agg.setdefault((r["cond"], r["kind"], r["layer"], r["head"]), []).extend(r["rec"])
conds = list(dict.fromkeys(r["cond"] for r in rows))
ms = lambda v: (np.mean(v), np.std(v) / len(v) ** .5)

probes = [("writer out@value", 0, None, "writer: GDN-0 out @ value", "#1f77b4"),
          ("state@t ALL (reference)", -1, None, "reference: all GDN states", "0.6"),
          ("head-path V@src", 15, 5, "reader L15H5: V @ source", "#d62728"),
          ("head-path V@src", 19, 1, "reader L19H1: V @ source", "#ff7f0e"),
          ("head-path V@src", 19, 5, "reader L19H5: V @ source", "#8c564b"),
          ("pathblock V@src", 15, None, "path block V@src, layer 15", "#2ca02c"),
          ("pathblock V@src", 19, None, "path block V@src, layer 19", "#9467bd")]

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [2.3, 1]})
ax = axes[0]
x = np.arange(len(conds))
w = 0.115
for j, (k, L, h, lab, c) in enumerate(probes):
    m = [ms(agg[(cd, k, L, h)])[0] for cd in conds]
    e = [ms(agg[(cd, k, L, h)])[1] for cd in conds]
    ax.bar(x + (j - 3) * w, m, w, yerr=e, capsize=2, label=lab, color=c)
ax.axhline(0, c="k", lw=.8)
ax.axhline(1, c="crimson", ls=":", lw=1)
ax.set_xticks(x)
ax.set_xticklabels([c.replace(": ", ":\n") for c in conds], fontsize=8.5)
ax.set_ylabel("recovery of logit difference")
ax.set_title("A. Fixed candidates re-tested across conditions\n"
             "(candidates frozen from steps 3-4; every condition uses unseen examples)",
             fontsize=11)
ax.set_ylim(0, 1.42)
ax.legend(fontsize=8, ncol=4, loc="upper center")

ax = axes[1]
dc = [ms(agg[(cd, "D_clean", None, None)]) for cd in conds]
dx = [ms(agg[(cd, "D_corr", None, None)]) for cd in conds]
ax.barh(x + .2, [v[0] for v in dc], .4, xerr=[v[1] for v in dc], capsize=2,
        color="#4c72b0", label="clean prompt")
ax.barh(x - .2, [v[0] for v in dx], .4, xerr=[v[1] for v in dx], capsize=2,
        color="#c44e52", label="corrupted prompt")
ax.set_yticks(x); ax.set_yticklabels(conds, fontsize=8.5)
ax.axvline(0, c="k", lw=.8)
ax.set_xlabel("answer logit difference  D")
ax.set_title("B. Task difficulty per condition", fontsize=11)
ax.legend(fontsize=8)

fig.tight_layout(); fig.savefig(a.out, dpi=150)
print("wrote", a.out)
