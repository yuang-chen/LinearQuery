"""Compact figures for paper Sections 4-5 (Qwen3.5-9B, chat8):
  attention_example_9B  reader's attention over the 8 entries: intact / Address removed / Address transplanted
  construction_9B       (a) query vs keys, (b) what carries the address, (c) upstream leave-one-out
Also re-exports nothing else; copies both to paper/images/."""
import json, os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.chdir("/mnt/yuang/LinearQuery")
INK, INK2, GRID, SURF, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#efeeea"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a8a7a2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})


def style(ax, ylim=None):
    ax.set_facecolor(SURF)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/{name}.{ext}", facecolor=SURF, bbox_inches="tight", pad_inches=0.05)
    shutil.copy(f"figures/{name}.png", f"paper/images/{name}.png")


# ------------------------------------------------------------------ attention example
ex = json.load(open("results/exp32_example_9B_chat8.json"))
ent = [e.strip().rstrip(",.") for e in ex["entries"]]
A, Bq = 2, 5                                     # donor asks entry 2 (banana), host asks entry 5 (ginger)
fig, ax = plt.subplots(figsize=(6.6, 1.75), dpi=200); fig.patch.set_facecolor(SURF)
x = np.arange(8); w = 0.27
for i, (c, col, lab) in enumerate((("intact", BLUE, "host intact (asks ginger)"),
                                   ("Address removed", GRAY, "Address block removed"),
                                   ("Address transplanted", ORANGE, "Address block from donor (asks banana)"))):
    ax.bar(x + (i - 1) * w, ex["example"][c]["att"], w * 0.92, color=col, label=lab, zorder=2,
           edgecolor=SURF, linewidth=0.5)
labels = [e.split("=")[0] + ("\n(host's key)" if j == Bq else "\n(donor's key)" if j == A else "") for j, e in enumerate(ent)]
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
for j in (A, Bq):
    ax.get_xticklabels()[j].set_color(INK); ax.get_xticklabels()[j].set_fontweight("bold")
style(ax, (0, 1.0)); ax.set_yticks([0, 0.5, 1.0])
ax.set_ylabel("Reader attention", fontsize=8)
ax.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=3, bbox_to_anchor=(0, 1.16), handlelength=1.0,
          columnspacing=1.0)
save(fig, "attention_example_9B")

# ------------------------------------------------------------------ construction figure
qk = {v: json.load(open(f"results/exp21/9B_{v}.json"))["selectivity"] for v in ("chat8",)}
ql8 = json.load(open("results/exp21/9B-L19H15_list8.json"))["selectivity"]
def sel(rows, grp, q, k):
    r = next(r for r in rows if r["group"] == grp and r["q"].startswith(q) and r["k"].startswith(k))
    return r["selectivity"], r["top1"]
pa = [("intact", sel(qk["chat8"], "Gr-1", "in", "in"), sel(ql8, "G4", "in", "in")),
      ("keys\nw/o $G_4$", sel(qk["chat8"], "Gr-1", "in", "Gr"), sel(ql8, "G4", "in", "G4")),
      ("query\nw/o $G_4$", sel(qk["chat8"], "Gr-1", "Gr", "in"), sel(ql8, "G4", "G4", "in")),
      ("query\nw/o $G_3$", sel(qk["chat8"], "Gr-2", "Gr", "in"), sel(ql8, "G3", "G3", "in"))]
st = {r["name"]: r for r in json.load(open("results/exp32_address_9B_chat8.json"))["rows"]}
stl = {r["name"]: r for r in json.load(open("results/exp32_address_9B_list8.json"))["rows"]}
pb = [("output, final token", "A  output, final token", BLUE),
      ("output, earlier tokens", "A  output, question span excl. final", BLUE),
      ("state of $G_4$", "B  state G4 [16, 17, 18]", AQUA),
      ("state of $G_3$", "B  state G3 [12, 13, 14]", AQUA),
      ("state of $G_0$", "B  state G0 [0, 1, 2]", AQUA),
      ("input, final token", "D  input to each Address mixer, final token", ORANGE)]
pc = [("w/o\n$G_1$", "C  output from donor without G1"), ("w/o\n$G_2$", "C  output from donor without G2"),
      ("w/o\n$G_3$", "C  output from donor without G3"), ("w/o\n$G_1$–$G_3$", "C  output from donor without G1-G3")]

fig, axes = plt.subplots(1, 3, figsize=(10.4, 2.25), dpi=200, gridspec_kw=dict(width_ratios=[1.0, 1.1, 0.8], wspace=0.55))
fig.patch.set_facecolor(SURF)
ax = axes[0]; x = np.arange(len(pa)); w = 0.36
for i in range(2):
    v = [p[1 + i][0] for p in pa]
    ax.bar(x + (i - 0.5) * w, v, w * 0.92, color=BLUE, alpha=1.0 if i == 0 else 0.45, zorder=2,
           edgecolor=SURF, linewidth=0.5)
    for xi, p in zip(x, pa):
        ax.text(xi + (i - 0.5) * w, p[1 + i][0] + 0.12, f"{p[1 + i][1]:.2f}", ha="center", fontsize=6.3, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pa], fontsize=7.5)
style(ax, (0, 6.6)); ax.set_ylabel("Reader selectivity", fontsize=8)
ax.set_title("(a) Address writes the query, not the keys", loc="left", fontsize=9, color=INK)
ax.text(1.0, 0.97, "numbers: top-1", transform=ax.transAxes, fontsize=6.5, color=INK2, ha="right", va="top")

ax = axes[1]; y = np.arange(len(pb))[::-1]
for yi, (lab, key, col) in zip(y, pb):
    ax.barh(yi + 0.17, st[key]["ansA"], 0.32, color=col, zorder=2)
    ax.barh(yi - 0.17, stl[key]["ansA"], 0.32, color=col, alpha=0.45, zorder=2)
    if st[key]["ansA"] < 0.05:
        ax.text(0.015, yi, "0.00", va="center", fontsize=6.5, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([p[0] for p in pb], fontsize=7.5)
ax.set_xlim(0, 1.05); ax.set_xticks([0, 0.5, 1.0])
ax.set_facecolor(SURF); ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_xlabel("Answers the donor's question", fontsize=8)
ax.set_title("(b) What carries the address?", loc="left", fontsize=9, color=INK)
ax.text(1.0, -0.36, "dark bars: chat8 · light bars: list8 (all panels)", transform=ax.transAxes, fontsize=6.8,
        color=INK2, ha="right")

ax = axes[2]; x = np.arange(len(pc))
ax.bar(x - 0.18, [st[k]["ansA"] for _, k in pc], 0.34, color=BLUE, zorder=2)
ax.bar(x + 0.18, [stl[k]["ansA"] for _, k in pc], 0.34, color=BLUE, alpha=0.45, zorder=2)
ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pc], fontsize=7.5)
style(ax, (0, 1.05)); ax.set_yticks([0, 0.5, 1.0])
ax.set_ylabel("Address still redirects", fontsize=8)
ax.text(3, 0.03, "0.00", ha="center", fontsize=6.5, color=INK2)
ax.set_title("(c) Address without upstream", loc="left", fontsize=9, color=INK)
save(fig, "construction_9B")
print("ok")

# ------------------------------------------------------------------ Section 7: two reader layers (Granite-Tiny)
fig, ax = plt.subplots(figsize=(4.6, 1.75), dpi=200); fig.patch.set_facecolor(SURF)
d = json.load(open("results/exp22c_combined_Gtiny_chat8.json"))
rows = {d["meta"]["names"][d["meta"]["groups"].index(r["layers"])]: r for r in d["transplant"] if r["positions"] == "final"}
order = [("G2", "$G_2$ (before L25)"), ("G3", "$G_3$ (before L35)"), ("G2+G3", "$G_2$ + $G_3$")]
x = np.arange(3); w = 0.26
series = [("answers donor's question", [rows[k]["ansA"] for k, _ in order], BLUE),
          ("L25 reader on donor's entry", [rows[k]["attA"] for k, _ in order], AQUA),
          ("L35 reader on donor's entry", [rows[k]["L35H1_A"] for k, _ in order], ORANGE)]
for i, (lab, v, col) in enumerate(series):
    ax.bar(x + (i - 1) * w, v, w * 0.92, color=col, label=lab, zorder=2, edgecolor=SURF, linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels([o[1] for o in order], fontsize=7.8)
style(ax, (0, 1.05)); ax.set_yticks([0, 0.5, 1.0])
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=6.8, handlelength=0.9,
          columnspacing=0.8, borderaxespad=0)
save(fig, "two_readers")
print("ok s7")
