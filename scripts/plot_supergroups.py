"""Figure 10: roles of the GDN super-groups G1+G2 (before the reader) and G4+G5 (after)."""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--inp", default="results/exp14_supergroups.json")
ap.add_argument("--out", default="figures/fig10_supergroups.png")
a = ap.parse_args()

J = json.load(open(a.inp))
rows = J["rows"]
sel = lambda k: [r for r in rows if r["kind"] == k]
BP, BA, BG = J["base_ppl"], J["base_acc"], J["base_gap"]
ATT = [3, 7, 11, 15, 19, 23]
CC = {"intact": "k", "remove G0": "#8c2d04", "remove G1+G2": "#4c72b0",
      "remove G3": "#dd8452", "remove G4+G5": "#c44e52"}

fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.6))

# --- A. ablation grid --------------------------------------------------------
ax = axes[0, 0]
sup = {r["name"]: r for r in sel("super")}
sing = {r["name"]: r for r in sel("single")}
items = [("G1+G2", sup["remove G1+G2"]), ("G4+G5", sup["remove G4+G5"]),
         ("G0", sing["remove G0"]), ("G3", sing["remove G3"]),
         ("G0+G3", sup["remove G0+G3"])]
x = np.arange(len(items))
ax.bar(x - .2, [r["ppl"] for _, r in items], .4, color="#4c72b0", label="perplexity")
ax.set_yscale("log"); ax.set_ylabel("WikiText-2 perplexity", color="#4c72b0")
ax.tick_params(axis="y", colors="#4c72b0")
ax.axhline(BP, ls="--", c="#4c72b0", lw=1.2, alpha=.7)
r6 = next(r for r in rows if r["kind"] == "random6_summary")
ax.axhline(np.exp(J["base_nll"] + r6["mean"]), ls=":", c="k", lw=1.4,
           label=f"6 random GDN layers ({np.exp(J['base_nll']+r6['mean']):.0f})")
ax2 = ax.twinx()
ax2.bar(x + .2, [r["acc"] for _, r in items], .4, color="#c44e52", label="retrieval accuracy")
ax2.axhline(BA, ls="--", c="#c44e52", lw=1.2, alpha=.7)
ax2.set_ylabel("retrieval accuracy", color="#c44e52"); ax2.set_ylim(0, 1.12)
ax2.tick_params(axis="y", colors="#c44e52")
ax.set_xticks(x); ax.set_xticklabels([f"remove\n{n}" for n, _ in items], fontsize=8.5)
ax.set_title("A. Both super-groups cost ~the same perplexity;\nonly G1+G2 destroys retrieval",
             fontsize=10.5)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
for i, (_, r) in enumerate(items):
    ax2.text(i + .2, r["acc"] + .03, f"{r['acc']:.2f}", ha="center", fontsize=8)

# --- B. logit lens -----------------------------------------------------------
ax = axes[0, 1]
for r in sel("logitlens"):
    ax.plot(range(len(r["traj"])), r["traj"], "o-", ms=3.5, color=CC[r["cond"]], label=r["cond"],
            lw=2 if r["cond"] == "intact" else 1.6)
for L in ATT:
    ax.axvline(L, color="0.85", lw=5, zorder=0)
ax.text(15, ax.get_ylim()[1] * .06, "softmax layers", fontsize=7, color="0.45", ha="center")
ax.set_xlabel("layer (residual stream read after this layer)")
ax.set_ylabel("$D$ = logit(clean ans) − logit(corrupt ans)")
ax.set_title("B. Logit lens: G1+G2 removal kills the jump AT layer 15;\n"
             "G4+G5 removal lets it happen, then loses it after 19", fontsize=10.5)
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.3)
ax.axhline(0, c="k", lw=.8)

# --- C. reader attention -----------------------------------------------------
ax = axes[1, 0]
at = sel("attn")
conds = [r["cond"] for r in at]
keys = [k for k in at[0] if k.startswith("L")]
w = 0.36
for j, k in enumerate(keys):
    ax.bar(np.arange(len(conds)) + (j - .5) * w, [r[k] for r in at], w, label=k)
ax.set_xticks(range(len(conds)))
ax.set_xticklabels([c.replace("remove ", "−") for c in conds], fontsize=8.5)
ax.set_ylabel("P(final token → source value token)")
ax.set_title("C. Reader attention: G1+G2 and G3 break the reader's\n"
             "aim; G4+G5 leaves it untouched", fontsize=10.5)
ax.legend(fontsize=8)

# --- D. recurrent horizon ----------------------------------------------------
ax = axes[1, 1]
hz = sel("horizon")
CB = {"G0": "#8c2d04", "G1+G2": "#4c72b0", "G3": "#dd8452", "G4+G5": "#c44e52"}
for r in hz:
    ax.plot(r["windows"], r["dnll"], "o-", color=CB[r["block"]], ms=5, label=r["block"])
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("recurrent memory window $w$ (tokens) inside that block only")
ax.set_ylabel("$\\Delta$NLL (nats)")
ax.set_title("D. Whose recurrence is long-range?\nG1+G2's value vanishes by w=64; G4+G5's persists",
             fontsize=10.5)
ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")

fig.tight_layout(); fig.savefig(a.out, dpi=150)
print("wrote", a.out)
