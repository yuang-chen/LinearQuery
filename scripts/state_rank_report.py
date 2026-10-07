#!/usr/bin/env python
"""Ranks of the memory states dumped by `scripts/state_rank.py`: per-head CSV, heatmap figure,
markdown tables (adapted from LinearSwap tools/state_rank_report.py).

    python scripts/state_rank_report.py 0.8B 9B G1b Gtiny
    python scripts/state_rank_report.py 0.8B 0.8B_wikitext          # same layout: Δ panels vs the first
    python scripts/state_rank_report.py 0.8B_dict G1b_dict --mark final

Numerical rank counts singular values above n·eps_fp32·σ_max (the fp32 SVD floor, n = the larger
state dimension); the significant rank counts those above 1 % of σ_max; the effective rank is
exp(entropy of σ / Σσ) (Roy & Vetterli, 2007). Full rank is the smaller state dimension (Gated
DeltaNet 128, Mamba-2 64). Each head is summarised by its median over the batch.

Adaptations for this project: models may differ in layout (layers, heads, state shape), so the
layer tables and Δ panels are produced per group of models that share a layout; every layer is
tagged with its Bind-Query circuit role, and a per-role summary compares the models.
"""
import argparse, csv, json
from pathlib import Path

import numpy as np

EPS32 = 2.0 ** -23
ROLES = ("bind", "feeder", "query", "post-reader")
TAG = {"bind": "B", "feeder": "f", "query": "Q", "post-reader": "·"}


def ranks(sv, tol):
    """an all-zero state (a dead head: Granite-Tiny L12 h24) has rank 0"""
    sv = np.asarray(sv)
    top = sv[..., :1]
    rel = np.divide(sv, top, out=np.zeros_like(sv), where=top > 0)
    return (rel > tol).sum(-1), (rel > 1e-2).sum(-1)                  # [B, H] each


def erank(sv):
    """Entropy effective rank exp(H(p)), p = σ / Σσ; also the share of the energy Σσ² in σ₁."""
    sv = np.asarray(sv)
    tot, e2 = sv.sum(-1, keepdims=True), (sv ** 2).sum(-1)
    p = np.divide(sv, tot, out=np.zeros_like(sv), where=tot > 0)
    er = np.where(tot[..., 0] > 0, np.exp(-(p * np.log(np.where(p > 0, p, 1))).sum(-1)), 0.0)   # dead head: 0
    return er, np.divide(sv[..., 0] ** 2, e2, out=np.full_like(e2, np.nan), where=e2 > 0)


p = argparse.ArgumentParser()
p.add_argument("models", nargs="+", help="names of results/state_rank/<name>.json")
p.add_argument("--dir", default="results/state_rank")
p.add_argument("--mark", default=None, help="prefix length (or dict_end / final) for the per-layer views; "
                                            "default: the last mark of each model")
p.add_argument("--fig", default="figures/state_rank.png")
p.add_argument("--csv", default="per_head_rank.csv", help="written into --dir")
a = p.parse_args()

D = {n: json.load(open(Path(a.dir) / f"{n}.json")) for n in a.models}
info = {}
for n, d in D.items():
    k1, k2 = d["state_shape"]
    mark = d["marks"][-1] if a.mark is None else (a.mark if a.mark in map(str, d["marks"]) and
                                                  not str(a.mark).isdigit() else int(a.mark))
    info[n] = dict(full=min(k1, k2), tol=max(k1, k2) * EPS32, mark=mark, layers=d["layers"],
                   roles=[d["roles"][str(i)] for i in d["layers"]])

num, sig, eff, top = {}, {}, {}, {}                                    # [layer, B, H] at the chosen mark
Path(a.dir).mkdir(parents=True, exist_ok=True)
with open(Path(a.dir) / a.csv, "w") as f:
    w = csv.writer(f)
    w.writerow(["model", "mark", "layer", "role", "head", "num_rank_median", "num_rank_min", "num_rank_max",
                "full_rank_seqs", "full_rank", "sig_rank_median", "eff_rank_median", "sigma1_energy_median"])
    for n, d in D.items():
        I = info[n]
        for m in d["marks"]:
            per = [ranks(d["sv"][f"{i}@{m}"], I["tol"]) for i in I["layers"]]
            ers = [erank(d["sv"][f"{i}@{m}"]) for i in I["layers"]]
            for i, rl, (r, r1), (e, t) in zip(I["layers"], I["roles"], per, ers):
                for h in range(r.shape[1]):
                    w.writerow([n, m, i, rl, h, int(np.median(r[:, h])), r[:, h].min(), r[:, h].max(),
                                int((r[:, h] == I["full"]).sum()), I["full"], int(np.median(r1[:, h])),
                                round(float(np.median(e[:, h])), 2), round(float(np.nanmedian(t[:, h])), 3)])
            if m == I["mark"]:
                num[n] = np.stack([r for r, _ in per]); sig[n] = np.stack([r1 for _, r1 in per])
                eff[n] = np.stack([e for e, _ in ers]); top[n] = np.stack([t for _, t in ers])

# groups of models with the same layout (layers, heads, state shape) share layer tables and Δ panels
groups = {}
for n in a.models:
    groups.setdefault((tuple(info[n]["layers"]), num[n].shape[2], info[n]["full"]), []).append(n)

# ---- markdown: per-role summary across all models
print("\n#### Per circuit role: median effective rank / median numerical rank (full rank) / median σ₁ energy share\n")
print("| model | mark | " + " | ".join(ROLES) + " | all |\n|---|---|" + "---|" * (len(ROLES) + 1))
for n in a.models:
    I = info[n]; cells = []
    for rl in ROLES + ("all",):
        k = [j for j, x in enumerate(I["roles"]) if rl == "all" or x == rl]
        cells.append("—" if not k else f"{np.median(eff[n][k]):.1f} / {int(np.median(num[n][k]))} ({I['full']}) / "
                     f"{np.nanmedian(top[n][k]):.2f}")
    print(f"| {n} | {I['mark']} | " + " | ".join(cells) + " |")

# ---- markdown: rank against prefix length, per model
print("\n#### Full-rank states (of layers × heads × batch) and median effective rank against prefix length\n")
for n, d in D.items():
    I = info[n]
    print(f"**{n}** (full rank {I['full']}): " + "  ".join(
        f"{m}: {sum(int((ranks(d['sv'][f'{i}@{m}'], I['tol'])[0] == I['full']).sum()) for i in I['layers'])}"
        f" / {np.median(np.stack([erank(d['sv'][f'{i}@{m}'])[0] for i in I['layers']])):.1f}"
        for m in d["marks"]))

# ---- markdown: per-layer and per-head tables, one block per layout group
for (layers, H, full), names in groups.items():
    roles = info[names[0]]["roles"]
    print(f"\n### Layout: {len(layers)} linear layers × {H} heads, full rank {full} — {', '.join(names)}")
    print(f"\n#### Per layer: full-rank states / median numerical rank / median significant rank\n")
    print("| layer | role | " + " | ".join(names) + " |\n|---|---|" + "---|" * len(names))
    for k, i in enumerate(layers):
        print(f"| {i} | {roles[k]} | " + " | ".join(
            f"{int((num[n][k] == full).sum())} / {int(np.median(num[n][k]))} / {int(np.median(sig[n][k]))}"
            for n in names) + " |")
    print(f"\n#### Per layer: median effective rank / median σ₁ share of the energy\n")
    print("| layer | role | " + " | ".join(names) + " |\n|---|---|" + "---|" * len(names))
    for k, i in enumerate(layers):
        print(f"| {i} | {roles[k]} | " + " | ".join(f"{np.median(eff[n][k]):.1f} / {np.nanmedian(top[n][k]):.2f}"
                                                   for n in names) + " |")
    print("| all | | " + " | ".join(f"**{np.median(eff[n]):.1f}** / **{np.nanmedian(top[n]):.2f}**" for n in names) + " |")
    print(f"\n#### Per-head median effective rank ({' / '.join(names)})\n")
    print("| L | " + " | ".join(f"h{h}" for h in range(H)) + " |\n|---|" + "---|" * H)
    for k, i in enumerate(layers):
        print(f"| {i}{TAG[roles[k]]} | " + " | ".join("/".join(f"{np.median(eff[n][k, :, h]):.0f}" for n in names)
                                                      for h in range(H)) + " |")
    print(f"\n#### Per-head median numerical rank ({' / '.join(names)}; bold: ≤ {full // 6})\n")
    print("| L | " + " | ".join(f"h{h}" for h in range(H)) + " |\n|---|" + "---|" * H)
    for k, i in enumerate(layers):
        cells = []
        for h in range(H):
            v = [int(np.median(num[n][k, :, h])) for n in names]
            cells.append(("**" + "/".join(map(str, v)) + "**") if max(v) <= full // 6 else "/".join(map(str, v)))
        print(f"| {i}{TAG[roles[k]]} | " + " | ".join(cells) + " |")
    ref = np.median(num[names[0]], 1).ravel()
    for n in names[1:]:
        x = np.median(num[n], 1).ravel()
        print(f"\n{n}: corr with {names[0]} per-head rank {np.corrcoef(ref, x)[0, 1]:.3f}, mean change {np.mean(x - ref):+.1f}")
        e0, e1 = np.median(eff[names[0]], 1).ravel(), np.median(eff[n], 1).ravel()
        print(f"{n}: corr with {names[0]} per-head effective rank {np.corrcoef(e0, e1)[0, 1]:.3f}, "
              f"ratio of medians {np.median(e1) / np.median(e0):.2f}, heads lower {np.mean(e1 < e0):.0%}")

# ---- figure: rows = numerical / significant / effective rank; columns = models, then Δ vs the first
#      model of each layout group (only for groups with more than one model)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

INK, MUTED, SURFACE = "#262624", "#6b6a66", "#fcfcfb"
ROLE_C = {"bind": "#b3302f", "feeder": MUTED, "query": "#1c5cab", "post-reader": "#b0aea8"}
seq = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
div = LinearSegmentedColormap.from_list("div", ["#b3302f", "#e34948", "#f0efec", "#3987e5", "#1c5cab"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "none"})
panels = [(n, None) for n in a.models]
# Δ panels only between runs of the *same* model (e.g. DCLM vs LongBench), as in LinearSwap where
# all rows share one backbone; two different models with the same layout are not compared cell by cell
panels += [(n, g[0]) for g in groups.values() for n in g[1:] if D[n].get("tag") == D[g[0]].get("tag")]
metrics = [("numerical rank", num, 1.0), ("significant rank (σ > 1% σ_max)", sig, 0.625),
           ("effective rank", eff, 0.3125)]
fig, ax = plt.subplots(len(metrics), len(panels), figsize=(2.4 * len(panels) + 1.2, 4.2 * len(metrics)),
                       facecolor=SURFACE, squeeze=False,
                       gridspec_kw=dict(wspace=0.35, hspace=0.25))
for r, (label, M, frac) in enumerate(metrics):
    for c, (n, refn) in enumerate(panels):
        A = ax[r, c]; I = info[n]; vmax = I["full"] * frac
        med = np.median(M[n], 1)                                       # [layer, head]
        if refn is None:
            im = A.imshow(med, cmap=seq, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
            title = n
        else:
            im = A.imshow(med - np.median(M[refn], 1), cmap=div, norm=TwoSlopeNorm(0, -vmax * 0.75, vmax * 0.25),
                          aspect="auto", interpolation="nearest")
            title = f"{n} − {refn}"
        if r == 0:
            A.set_title(title, fontsize=10, color=INK, pad=6)
        H = med.shape[1]
        A.set_facecolor(SURFACE)
        A.set_xticks(range(0, H, max(1, H // 8)))
        A.set_yticks(range(len(I["layers"])))
        A.set_yticklabels(I["layers"], fontsize=7)
        for t, rl in zip(A.get_yticklabels(), I["roles"]):
            t.set_color(ROLE_C[rl])
        A.tick_params(which="both", length=0)
        if r == len(metrics) - 1:
            A.set_xlabel("head")
        if c == 0:
            A.set_ylabel(f"{label}\n\nlayer", color=INK)
        cb = fig.colorbar(im, ax=A, fraction=0.06, pad=0.02)
        cb.outline.set_visible(False); cb.ax.tick_params(length=0, colors=MUTED, labelsize=7)
marks = sorted({str(info[n]["mark"]) for n in a.models})
fig.suptitle(f"Rank of each head's memory state at {' / '.join(marks)} (median over the batch)\n"
             "layer labels: Bind (red), Query (blue), feeder (grey), after the reader (light)",
             fontsize=11, color=INK, y=0.93)
Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.fig, dpi=150, bbox_inches="tight", facecolor=SURFACE)
print(f"\n[state_rank_report] figure -> {a.fig}; per-head CSV -> {Path(a.dir) / a.csv}")
