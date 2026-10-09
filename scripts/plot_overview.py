"""Overview figure (paper Fig. 1): (a) the Encode--Address circuit on one prompt, (b) the query swap,
(c) transfer across families. Rebuilt from the original hand-made figure with the current names.
-> figures/overview.{png,pdf} and paper/images/intro.png"""
import os, shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

os.chdir("/mnt/yuang/LinearQuery")
INK, INK2, GRID = "#222222", "#666666", "#c9c8c4"
SOFT_E, SOFT_F = "#5a4a8c", "#e9e5f3"              # softmax
ADDR_E, ADDR_F = "#2a5fa8", "#dde9f8"              # Address
FEED_E, FEED_F = "#c4c3be", "#f1f0ec"              # feeders
ENC_E, ENC_F = "#b0302f", "#f6e1e1"                # Encode
HIL = "#c2410c"                                    # the swapped answer

fig = plt.figure(figsize=(15.6, 4.7), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 156); ax.set_ylim(0, 47); ax.axis("off")
Y = dict(soft=31.0, addr=25.5, feed=20.3, enc=15.6)  # row centres
H = 3.4


def box(x0, x1, row, label=None, dashed=False, italic=False, fs=12):
    e, f = {"soft": (SOFT_E, SOFT_F), "addr": (ADDR_E, ADDR_F), "feed": (FEED_E, FEED_F),
            "enc": (ENC_E, ENC_F)}[row]
    ax.add_patch(FancyBboxPatch((x0, Y[row] - H / 2), x1 - x0, H, boxstyle="round,pad=0,rounding_size=0.6",
                                ec=e, fc=f, lw=1.8, ls=(0, (4, 3)) if dashed else "-", zorder=2))
    if label:
        ax.text((x0 + x1) / 2, Y[row], label, ha="center", va="center", fontsize=fs, color=e,
                weight="bold" if not italic else "normal", style="italic" if italic else "normal", zorder=3)


def dot(x, row, r=0.85, c=None, txt=None, fs=13):
    c = c or {"soft": SOFT_E, "addr": ADDR_E, "enc": ENC_E}[row]
    ax.add_patch(Circle((x, Y[row]), r, fc=c, ec="white", lw=1.2, zorder=5))
    if txt:
        ax.text(x, Y[row] - 0.05, txt, ha="center", va="center", fontsize=fs, color="white", weight="bold", zorder=6)


def arrow(p, q, c, lw=1.8, ls="-", rad=0.0, head=True):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=4,head_width=2.4" if head else "-",
                                 connectionstyle=f"arc3,rad={rad}", color=c, lw=lw, ls=ls, zorder=4,
                                 shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ (a) the circuit
ax.text(2, 44.5, "(a) The Encode–Address circuit", fontsize=17, weight="bold", color=INK, va="center")
for row, name, sub in (("soft", "softmax", "retrieves"), ("addr", "Address", "builds the query"),
                       ("feed", "feeders", None), ("enc", "Encode", "token identity")):
    e = {"soft": SOFT_E, "addr": ADDR_E, "feed": INK2, "enc": ENC_E}[row]
    ax.text(2.6, Y[row] + (0.9 if sub else 0), name, fontsize=15, weight="bold", color=e, va="center")
    if sub:
        ax.text(2.6, Y[row] - 1.4, sub, fontsize=12, color=INK2, va="center")
cols = [32.4, 49.5, 65.7]
for x in cols:
    ax.plot([x, x], [12.5, 37.5], color=GRID, lw=1.6, zorder=1)
for row in ("soft", "addr", "feed", "enc"):
    box(22.5, 72.8, row, dashed=(row == "feed"))
dot(cols[0], "enc"); dot(cols[1], "enc")
dot(cols[1], "soft", r=1.5, txt="k")
dot(cols[2], "addr", r=1.5, txt="q")
dot(cols[2], "soft", r=0.75)
arrow((cols[1], Y["enc"] + 0.9), (cols[1], Y["soft"] - 1.5), ENC_E, lw=1.8, ls=(0, (2, 1.5)), head=False)
arrow((cols[2], Y["addr"] + 1.5), (cols[2], Y["soft"] - 0.75), ADDR_E, lw=2.0)
arrow((cols[2] - 0.6, Y["soft"] + 0.6), (cols[1] + 1.4, Y["soft"] + 0.6), SOFT_E, lw=2.0, rad=0.55, head=False)
ax.text((cols[1] + cols[2]) / 2 + 0.5, 35.7, "q · k", fontsize=13, color=SOFT_E, ha="center")
arrow((cols[2], Y["soft"] + 0.75), (cols[2], 38.4), INK, lw=1.4)
ax.text(cols[2], 39.6, "wood", fontsize=16, weight="bold", color=INK, ha="center", va="center")
ax.text(cols[0], 10.9, "berry=amber", fontsize=14, color=INK, ha="center", va="center")
ax.text(cols[1], 10.9, "ginger=wood", fontsize=14, color=INK, ha="center", va="center", weight="bold")
ax.text(cols[1], 8.8, "asked entry", fontsize=12, color=INK2, ha="center", va="center")
ax.text(cols[2] + 0.4, 10.9, "…of ginger is", fontsize=14, color=INK, ha="center", va="center")
ax.text(cols[2], 8.8, "final token", fontsize=12, color=INK2, ha="center", va="center")

# ------------------------------------------------------------------ (b) swap the query
ax.text(75.8, 44.5, "(b) Swap the query", fontsize=17, weight="bold", color=INK, va="center")
bx = [85.6, 101.8]
for i, x in enumerate(bx):
    ax.plot([x, x], [12.5, 37.5], color=GRID, lw=1.6, zorder=1)
    for row in ("soft", "addr", "feed", "enc"):
        box(x - 4.5, x + 4.5, row, dashed=(row == "feed"))
    dot(x, "soft", r=0.75)
    dot(x, "addr", r=1.75, txt="$q_A$", fs=13)
    arrow((x, Y["addr"] + 1.75), (x, Y["soft"] - 0.75), ADDR_E, lw=2.0)
    arrow((x, Y["soft"] + 0.75), (x, 38.4), INK, lw=1.4)
    ax.text(x, 39.6, "lime", fontsize=16, weight="bold", color=INK if i == 0 else HIL, ha="center", va="center")
arrow((bx[0] + 1.6, Y["addr"] + 1.2), (bx[1] - 1.6, Y["addr"] + 1.2), ADDR_E, lw=2.2, rad=-0.45, head=False)
for x, a, b in ((bx[0], "prompt A", "…of banana is"), (bx[1], "prompt B", "…of ginger is")):
    ax.text(x, 10.9, a, fontsize=14, color=INK, ha="center", va="center")
    ax.text(x, 8.9, b, fontsize=14, color=INK, ha="center", va="center")
for j, t in enumerate(("B answers A's question", "0.97–0.99 (Qwen3.5-9B)", "any other block: ≈ 0")):
    ax.text(93.7, 6.2 - 2.4 * j, t, fontsize=13, color=INK, ha="center", va="center")

# ------------------------------------------------------------------ (c) across families
ax.text(113.2, 44.5, "(c) Across families", fontsize=17, weight="bold", color=INK, va="center")
cx = [121.3, 146.5]
for i, x in enumerate(cx):
    box(x - 6.1, x + 6.1, "soft", "softmax", fs=14)
    if i == 0:
        box(x - 6.1, x + 6.1, "addr", "Address", fs=14)
    else:
        box(x - 6.1, x + 6.1, "addr", "from Qwen", italic=True, fs=14)
    box(x - 6.1, x + 6.1, "feed", "…", dashed=True, fs=13)
    box(x - 6.1, x + 6.1, "enc", "Encode", fs=14)
arrow((cx[0] + 6.1, Y["addr"]), (cx[1] - 6.1, Y["addr"]), ADDR_E, lw=2.2)
ax.text((cx[0] + cx[1]) / 2, Y["addr"] + 1.8, "alignment maps", fontsize=11, color=ADDR_E, ha="center")
ax.text((cx[0] + cx[1]) / 2, Y["addr"] - 1.8, "no fine-tuning", fontsize=11, color=ADDR_E, ha="center", va="top")
for x, a, b in ((cx[0], "Qwen3.5", "Gated DeltaNet"), (cx[1], "Granite4.0", "Mamba-2")):
    ax.text(x, 10.9, a, fontsize=14, color=INK, ha="center", va="center")
    ax.text(x, 8.9, b, fontsize=14, color=INK, ha="center", va="center")
for j, t in enumerate(("address, 9B → 7B: 0.00 → 0.99", "block, 1B → 0.8B: 0.30 → 0.95")):
    ax.text((cx[0] + cx[1]) / 2, 6.2 - 2.4 * j, t, fontsize=13, color=INK, ha="center", va="center")

os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/overview.{ext}", facecolor="white")
shutil.copy("figures/overview.png", "paper/images/intro.png")
print("wrote figures/overview.{png,pdf} and paper/images/intro.png")
