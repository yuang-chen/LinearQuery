"""Section 4 figure for Qwen3.5-9B (chat8): per linear-attention block, (a) removal -- accuracy, logit gap
(relative to the intact model) and the reader's attention on the queried entry (exp31); (b) final-token
transplant -- rate of answering the donor's question and the reader's attention on the donor's entry
(exp22). -> figures/circuit_9B.{png,pdf}"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.chdir("/mnt/yuang/LinearQuery")
pr = json.load(open("results/exp31_profile_9B_chat8.json"))
tr = json.load(open("results/exp22_transplant_9B.json"))
blocks = pr["blocks"]; n = len(blocks)
I = pr["intact"]
rem = {"Accuracy": [b["acc"] for b in blocks],
       "Logit gap (÷ intact)": [b["gap"] / I["gap"] for b in blocks],
       "Reader attention on target": [b["attn"] for b in blocks]}
fin = {r["group"]: r for r in tr["transplant"] if r["positions"] == "final"}
tra = {"Answers the donor's question": [fin[g]["ansA"] for g in range(n)],
       "Reader attention on donor's entry": [fin[g]["attA"] for g in range(n)]}

INK, INK2, GRID, SURF, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#efeeea"
C = ["#2a78d6", "#eb6834", "#1baf7a"]              # categorical slots 1-3 of the reference palette
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 2, figsize=(10.4, 2.35), dpi=200, gridspec_kw=dict(wspace=0.12))
fig.patch.set_facecolor(SURF)
labels = [f"G{b['block']}\nL{b['layers'][0]}–{b['layers'][-1]}" for b in blocks]
ROLE = {0: "Encode", 4: "Address"}


def panel(ax, series, title, w, colors):
    ax.set_facecolor(SURF)
    x = np.arange(n)
    for g, lab in ROLE.items():
        ax.axvspan(g - 0.5, g + 0.5, color=BAND, zorder=0)
        ax.text(g, 1.03, lab, ha="center", va="bottom", fontsize=8, color=INK, weight="bold")
    k = len(series)
    for i, (lab, v) in enumerate(series.items()):
        ax.bar(x + (i - (k - 1) / 2) * w, v, w * 0.92, color=colors[i], label=lab, zorder=2,
               edgecolor=SURF, linewidth=0.6)
    ax.axvline(4.5, color=INK2, lw=0.9, ls=(0, (1, 2)), zorder=1)     # softmax layer 19 (the reader)
    ax.text(4.95, 1.03, "reader L19", fontsize=7.5, color=INK2, va="bottom", ha="left")
    ax.set_xticks(x); ax.set_xticklabels([f"G{i}" for i in range(n)], fontsize=8)
    ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.5, 1.0])
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=k, frameon=False, fontsize=7.3,
              handlelength=0.9, columnspacing=0.9, borderaxespad=0)


panel(axes[0], rem, "(a) Removal: is the block necessary?", 0.26, C)
panel(axes[1], tra, "(b) Transplant: does the block decide what is retrieved?", 0.34, [C[0], C[2]])

for ext in ("png", "pdf"):
    fig.savefig(f"figures/circuit_9B.{ext}", facecolor=SURF, bbox_inches="tight", pad_inches=0.1)
print({k: [round(x, 2) for x in v] for k, v in {**rem, **tra}.items()})
