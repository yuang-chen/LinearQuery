"""Figure 1: GDN writer effects by layer and intervention type (+ controls)."""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--scan", default="results/exp3_writers.json")
ap.add_argument("--ctrl", default="results/exp3b_controls.json")
ap.add_argument("--out", default="figures/fig1_writers.png")
a = ap.parse_args()


def load(p):
    agg = {}
    for r in json.load(open(p))["rows"]:
        agg.setdefault((r["kind"], r["layer"]), []).extend(r["rec"])
    return agg


scan, ctrl = load(a.scan), load(a.ctrl)
layers = sorted({L for (k, L) in scan if L >= 0})
ms = lambda v: (np.mean(v), np.std(v) / len(v) ** .5)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2.1, 1]})

ax = axes[0]
kinds = [("state@t", "clean recurrent state after the value token"),
         ("out@value", "GDN output @ value token"),
         ("out@delim", "GDN output @ following delimiter"),
         ("out@query+final", "GDN output @ query key + final token")]
w, cols = 0.2, plt.cm.viridis(np.linspace(0.05, 0.8, 4))
for j, (k, lab) in enumerate(kinds):
    src = scan if (k, layers[0]) in scan else ctrl
    m = np.array([ms(src[(k, L)])[0] if (k, L) in src else np.nan for L in layers])
    s = np.array([ms(src[(k, L)])[1] if (k, L) in src else np.nan for L in layers])
    ax.bar(np.arange(len(layers)) + (j - 1.5) * w, m, w, yerr=s, capsize=2,
           label=lab, color=cols[j])
ax.axhline(ms(scan[("state@t ALL", -1)])[0], ls="--", c="crimson", lw=1.3,
           label=f"all GDN states patched ({ms(scan[('state@t ALL', -1)])[0]:+.3f})")
ax.axhline(0, c="k", lw=.8)
ax.set_xticks(np.arange(len(layers))); ax.set_xticklabels(layers)
ax.set_xlabel("GDN layer"); ax.set_ylabel("recovery of logit difference")
ax.set_title("A. Writer scan: clean$\\to$corrupted GDN patches\n(8-pair dictionary, 100 pairs)",
             fontsize=11)
ax.set_ylim(-0.1, 1.30)
ax.legend(fontsize=8, loc="upper left")

# inset: same data, zoomed so the sub-percent effects are readable
ins = ax.inset_axes([0.30, 0.26, 0.67, 0.44])
for j, (k, lab) in enumerate(kinds):
    src = scan if (k, layers[0]) in scan else ctrl
    m = np.array([ms(src[(k, L)])[0] if (k, L) in src else np.nan for L in layers])
    s_ = np.array([ms(src[(k, L)])[1] if (k, L) in src else np.nan for L in layers])
    ins.bar(np.arange(len(layers)) + (j - 1.5) * w, m, w, yerr=s_, capsize=1.5, color=cols[j])
ins.axhline(0, c="k", lw=.6)
ins.set_ylim(-0.07, 0.13)
ins.set_xticks(np.arange(len(layers))); ins.set_xticklabels(layers, fontsize=6)
ins.tick_params(labelsize=6)
ins.set_title("zoom (note: all state patches $\\approx$ 0)", fontsize=7.5)

ax = axes[1]
names = [("emb@value", -2, "token embedding @ value"),
         ("resid_after_layer0@value", 0, "residual after layer 0 @ value"),
         ("out@value", 0, "GDN-0 output @ value  (sufficiency)"),
         ("REV out@value", 0, "GDN-0 output @ value  (necessity, reversed)"),
         ("out@value", 14, "GDN-14 output @ value"),
         ("out@final", 22, "GDN-22 output @ final token")]
vals = [ms(ctrl[(k, L)]) for k, L, _ in names]
y = np.arange(len(names))[::-1]
ax.barh(y, [v[0] for v in vals], xerr=[v[1] for v in vals],
        color=["0.75", "0.6", "#1f77b4", "#4aa3df", "#8fbc8f", "#d4a017"], capsize=3)
ax.set_yticks(y); ax.set_yticklabels([n[2] for n in names], fontsize=8.5)
ax.axvline(0, c="k", lw=.8); ax.axvline(1, c="crimson", ls=":", lw=1)
ax.set_xlim(0, 1.18)
ax.set_xlabel("recovery of logit difference")
ax.set_title("B. Controls at the source value position", fontsize=11)

fig.tight_layout(); fig.savefig(a.out, dpi=150)
print("wrote", a.out)
