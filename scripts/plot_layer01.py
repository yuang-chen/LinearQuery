"""Figure 8: layer 0 vs layer 1 -- geometry, content, redundancy, specialisation."""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default="results/exp12_layer01.json")
ap.add_argument("--noise", default="results/exp12b_matched_noise.json")
ap.add_argument("--out", default="figures/fig8_layer01.png")
a = ap.parse_args()

J = json.load(open(a.inp))
rows = J["rows"]
sel = lambda k: [r for r in rows if r["kind"] == k]
NZ = json.load(open(a.noise))["rows"]
C = {0: "#c44e52", 1: "#4c72b0", 2: "#8fbc8f", 4: "#b0b0b0"}

fig, axes = plt.subplots(2, 3, figsize=(17, 9.2))

# --- A. geometry -------------------------------------------------------------
ax = axes[0, 0]
g = sorted(sel("geometry"), key=lambda r: r["layer"])
L = [r["layer"] for r in g]
ax.bar(range(len(L)), [r["ratio"] for r in g],
       color=[C.get(x, "0.78") for x in L])
ax.set_xticks(range(len(L))); ax.set_xticklabels(L, fontsize=7)
ax.axhline(1, ls=":", c="k", lw=1)
ax.set_xlabel("GDN layer"); ax.set_ylabel("$\\|o_L\\| \\,/\\, \\|h_{in}\\|$")
ax.set_title("A. Layer 0 writes a vector ~2x larger than\nthe residual stream it reads", fontsize=10.5)
for i in (0, 1):
    ax.text(i, g[i]["ratio"] + .04, f"{g[i]['ratio']:.2f}", ha="center", fontsize=9)

# --- B. matched-absolute noise -----------------------------------------------
ax = axes[0, 1]
for Lx in (0, 1, 2, 4):
    v = sorted([r for r in NZ if r["mode"] == "abs" and r["layer"] == Lx], key=lambda r: r["amt"])
    ax.errorbar([r["amt"] for r in v], [r["dnll"] for r in v], yerr=[r["sem"] for r in v],
                fmt="o-", color=C[Lx], label=f"layer {Lx}", ms=4, capsize=2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("absolute perturbation $\\delta$  ($o_L \\to o_L + \\delta u$)")
ax.set_ylabel("$\\Delta$NLL (nats)")
ax.set_title("B. At MATCHED absolute noise, layers 0 and 1\nare equally sensitive", fontsize=10.5)
ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")

# --- C. structure index ------------------------------------------------------
ax = axes[0, 2]
lad = {(r["layer"], r["rung"]): r for r in sel("ladder")}
noise_o = {r["layer"]: r["dnll"] for r in sel("noise") if r["eps"] == 1.0}
lays = [0, 1, 2, 4]
zero = [lad[(Lx, "zero")]["dnll"] if (Lx, "zero") in lad
        else next(r["dnll"] for r in sel("joint") if r["name"] == f"zero mixers [{Lx}]")
        for Lx in lays]
idx = [z / noise_o[Lx] for z, Lx in zip(zero, lays)]
ax.bar(range(len(lays)), idx, .6, color=[C[x] for x in lays])
for i, v in enumerate(idx):
    ax.text(i, v + .2, f"{v:.1f}x", ha="center", fontsize=9)
ax.set_xticks(range(len(lays))); ax.set_xticklabels([f"layer {x}" for x in lays], fontsize=9)
ax.set_ylabel("$\\Delta$NLL(delete) / $\\Delta$NLL(random noise of equal size)")
ax.set_title("C. Structure index: only for layer 0 does the\nspecific direction matter much more than its size",
             fontsize=10.5)

# --- D. redundancy -----------------------------------------------------------
ax = axes[1, 0]
j = sel("joint")
names = [("[0]", "zero mixers [0]"), ("[1]", "zero mixers [1]"), ("[2]", "zero mixers [2]"),
         ("[1,2]", "zero mixers [1, 2]"), ("[1,2,4]", "zero mixers [1, 2, 4]"),
         ("[1,2,4,5]", "zero mixers [1, 2, 4, 5]")]
vals = [next(r["dnll"] for r in j if r["name"] == nm) for _, nm in names]
x = np.arange(len(names))
ax.bar(x, vals, .6, color=[C[0], C[1], C[2], "#dd8452", "#dd8452", "#dd8452"])
jr = {r["n"]: r["dnll"] for r in sel("joint_random_mean")}
for n, xi in ((2, 3), (3, 4), (4, 5)):
    if n in jr:
        ax.hlines(jr[n], xi - .4, xi + .4, color="k", lw=2, ls="--")
ax.plot([], [], "k--", lw=2, label="matched random sets, same size")
ax.set_xticks(x); ax.set_xticklabels([n for n, _ in names], fontsize=8.5)
ax.set_xlabel("GDN mixers zeroed"); ax.set_ylabel("$\\Delta$NLL (nats)")
ax.set_yscale("symlog", linthresh=.1)
ax.set_title("D. Layer 1 is cheap alone; layers 1+2\ntogether are 6x worse than either", fontsize=10.5)
ax.legend(fontsize=8, loc="upper left")
for i, v in enumerate(vals):
    ax.text(i, v * 1.15, f"{v:+.2f}", ha="center", fontsize=8)

# --- E. information ladder ---------------------------------------------------
ax = axes[1, 1]
RUNGS = [("zero", "zeros"), ("global_mean", "global mean"), ("token_mean", "token-type mean"),
         ("token_local", "token-local"), ("conv1tap", "conv 1 tap"), ("norecur", "no recurrence")]
w = 0.38
for j2, Lx in enumerate((0, 1)):
    v = [lad[(Lx, r)]["dnll"] for r, _ in RUNGS]
    ax.bar(np.arange(len(RUNGS)) + (j2 - .5) * w, v, w, color=C[Lx], label=f"layer {Lx}")
ax.set_yscale("log")
ax.set_xticks(np.arange(len(RUNGS)))
ax.set_xticklabels([l for _, l in RUNGS], fontsize=7.5, rotation=20, ha="right")
ax.set_ylabel("$\\Delta$NLL (nats)")
ax.set_title("E. The same ladder, ~30x smaller for layer 1\nat every rung", fontsize=10.5)
ax.legend(fontsize=8)

# --- F. specialisation -------------------------------------------------------
ax = axes[1, 2]
cc = {r["cond"]: r for r in sel("concentration01")}
fr = [1, 5, 10]
for nm, Lx in (("zero L0", 0), ("zero L1", 1)):
    ax.plot(fr, [cc[nm]["top1"], cc[nm]["top5"], cc[nm]["top10"]], "o-", color=C[Lx], ms=6,
            label=f"zero layer {Lx}  ({cc[nm]['frac_pos']:.0%} of positions hurt)")
ax.plot(fr, [f / 100 for f in fr], ":", c="k", lw=1, label="uniform damage")
ax.set_xlabel("worst $k$% of positions"); ax.set_ylabel("share of that condition's damage")
ax.set_title("F. Layer 0 hurts nearly every token;\nlayer 1 hurts a narrow set", fontsize=10.5)
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3)
ax.text(.03, .96, f"$r(\\Delta_{{L0}},\\Delta_{{L1}}) = {sel('corr01')[0]['pearson']:+.2f}$",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(fc="white", ec="0.7", alpha=.9))

fig.tight_layout(); fig.savefig(a.out, dpi=150)
print("wrote", a.out)
