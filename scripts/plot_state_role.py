"""Figure 4: what the GDN recurrent state is for -- horizon, decodability, route isolation."""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--ppl", default="results/exp8_horizon_ppl.json")
ap.add_argument("--probe_end", default="results/exp7_state_probe_dictend.json")
ap.add_argument("--probe_fin", default="results/exp7_state_probe_final.json")
ap.add_argument("--role", default="results/exp6_state_role.json")
ap.add_argument("--out", default="figures/fig4_state_role.png")
a = ap.parse_args()

P = {r["name"]: r for r in json.load(open(a.ppl))["rows"]}
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))

# --- A. functional memory horizon -------------------------------------------
ax = axes[0]
ws, ppls = [], []
for k, r in P.items():
    if k.startswith("state reset every"):
        ws.append(r["window"]); ppls.append(r["ppl"])
o = np.argsort(ws); ws = np.array(ws)[o]; ppls = np.array(ppls)[o]
base = P["baseline"]["ppl"]
ax.plot(ws, ppls, "o-", color="#1f77b4", lw=2, ms=6, label="GDN state reset every $w$ tokens")
ax.axhline(base, ls="--", c="k", lw=1.2, label=f"intact model (PPL {base:.1f})")
ax.axhline(P["all GDN layers skipped"]["ppl"], ls=":", c="crimson", lw=1.3,
           label=f"all GDN layers removed ({P['all GDN layers skipped']['ppl']:.3g})")
ax.axhline(P["all softmax layers skipped"]["ppl"], ls=":", c="#d4a017", lw=1.3,
           label=f"all softmax layers removed ({P['all softmax layers skipped']['ppl']:.0f})")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("reset window $w$ (tokens of recurrent memory retained)")
ax.set_ylabel("WikiText-2 perplexity")
ax.set_title("A. Functional memory horizon of the state\n"
             "(convolution cache and softmax KV left intact)", fontsize=10.5)
ax.legend(fontsize=7.5, loc="upper right")
ax.grid(alpha=.3, which="both")

# --- B. decodability of the binding from the state ---------------------------
ax = axes[1]
for path, lab, c, mk in ((a.probe_end, "state before the question is read", "#c44e52", "o"),
                         (a.probe_fin, "state at the final token", "#4c72b0", "s")):
    try:
        D = json.load(open(path))
    except FileNotFoundError:
        continue
    rs = [r for r in D["rows"] if r["family"] == "query-key"]
    rs.sort(key=lambda r: r["layer"])
    ax.plot([r["layer"] for r in rs], [r["acc"] for r in rs], mk + "-", color=c, label=lab, ms=5)
    ceil = [r for r in D["rows"] if r["family"] == "ceiling"]
    if ceil:
        ax.axhline(ceil[0]["acc"], ls="--", c=c, lw=1, alpha=.5)
ax.axhline(0.125, ls=":", c="k", lw=1.2, label="'presence' floor (knows the 8 values, not the binding)")
ax.axhline(0.05, ls=":", c="0.6", lw=1, label="uninformed chance (20 words)")
for x in (15, 19):
    ax.axvline(x, color="#d4a017", lw=6, alpha=.18)
ax.text(15.2, 0.93, "softmax\nL15 / L19", fontsize=7.5, color="#8a6d1a")
ax.set_xlabel("GDN layer"); ax.set_ylabel("probe accuracy (20-way)")
ax.set_title("B. Can the key$\\to$value binding be decoded\nfrom the recurrent state?", fontsize=10.5)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=7.5, loc="center left")

# --- C. route isolation ------------------------------------------------------
ax = axes[2]
rows = json.load(open(a.role))["rows"]
names = ["identity ctrl (corrupted)", "state route only", "kv route only", "identity ctrl (clean)"]
labs = ["neither\nroute", "state\nroute only", "KV\nroute only", "both\nroutes"]
vals, errs = [], []
for nm in names:
    v = [x for r in rows if r["kind"] == "route" and r["name"] == nm for x in r["rec"]]
    vals.append(np.mean(v)); errs.append(np.std(v) / len(v) ** .5)
cols = ["0.7", "#c44e52", "#4c72b0", "0.5"]
ax.bar(range(4), vals, .62, yerr=errs, capsize=3, color=cols)
for i, v in enumerate(vals):
    ax.text(i, v + .04, f"{v:+.3f}", ha="center", fontsize=9)
ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=9)
ax.set_ylabel("recovery of answer logit difference")
ax.set_ylim(-0.05, 1.18)
ax.axhline(0, c="k", lw=.8)
ax.set_title("C. Cut one route, keep the other\n(8-pair dictionary, 100 pairs)", fontsize=10.5)

fig.tight_layout(); fig.savefig(a.out, dpi=150)
print("wrote", a.out)
