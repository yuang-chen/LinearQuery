"""Overview figure, redesigned (candidate for paper Fig. 1).

One visual language for all three panels: x = prompt positions, y = depth. Linear-attention
blocks are horizontal bands; softmax layers are thin dark bands. Inside the Address band the
recurrent state is drawn as a ribbon that picks up the queried key while the question is read and
writes it out at the final token, where the reader turns it into its attention query.
(a) the circuit on one prompt, (b) removal vs. transplant of the Address block, (c) the address
crossing to another architecture through an alignment map.
-> figures/overview_v2.{png,pdf}  (does not touch paper/images/intro.png)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D

os.chdir("/mnt/yuang/LinearQuery")

INK, INK2, INK3 = "#222222", "#5c5c5c", "#9a9a9a"
LIN, SOFT = "#e6e5e1", "#4b4b4b"                 # linear layer / softmax layer
ENC_F, ENC_E = "#e3f3ea", "#1f8f5f"              # Encode band
ADR_F, ADR_E = "#e2edfb", "#2f6fd1"              # Address band
HOST, DONOR = "#2f7de1", "#e8672b"               # host question / donor question
OK = "#1fa36b"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig = plt.figure(figsize=(17, 6.4), dpi=110)


# ------------------------------------------------------------------ helpers
def band(ax, y0, y1, x0, x1, fc, ec, lw=1.2, ls="-", z=1, hatch=None):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z, hatch=hatch))


def arrow(ax, p, q, c, lw=1.8, ls="-", rad=0.0, head=True, z=6, hl=5, hw=3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=f"-|>,head_length={hl},head_width={hw}" if head else "-",
                                 connectionstyle=f"arc3,rad={rad}", color=c, lw=lw, ls=ls, zorder=z,
                                 shrinkA=0, shrinkB=0))


def dot(ax, x, y, c, r=0.16, txt=None, fs=10, z=8, tc="white"):
    ax.add_patch(Circle((x, y), r, fc=c, ec="white", lw=1.2, zorder=z))
    if txt:
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, color=tc, weight="bold", zorder=z + 1)


def ribbon(ax, xs, y, c, lw=5, alpha=1.0, z=5):
    ax.plot(xs, [y] * len(xs), color=c, lw=lw, alpha=alpha, solid_capstyle="round", zorder=z)


# ------------------------------------------------------------------ shared stack geometry
# rows (y): tokens 0.0 | Encode 0.6-1.4 | feeders 1.7-2.4 | Address 2.7-3.6 | reader 3.85-4.2 | later 4.45-5.0 | out 5.5
Y_TOK, ENC, FEED, ADR, RDR, LATER, Y_OUT = 0.0, (0.65, 1.45), (1.75, 2.45), (2.75, 3.65), (4.0, 4.35), (4.95, 5.4), 5.9
YA = (ADR[0] + ADR[1]) / 2
YR = (RDR[0] + RDR[1]) / 2
YL = (LATER[0] + LATER[1]) / 2


def stack(ax, x0, x1, show_feed=True, show_later=True, addr_fc=ADR_F, addr_ec=ADR_E, addr_hatch=None,
          labels=False, lx=None):
    """Draw the layer bands between x0 and x1."""
    band(ax, *ENC, x0, x1, ENC_F, ENC_E)
    if show_feed:
        band(ax, *FEED, x0, x1, "#f4f4f2", "#c8c7c2", ls=(0, (3, 2)))
    band(ax, *ADR, x0, x1, addr_fc, addr_ec, hatch=addr_hatch)
    band(ax, *RDR, x0, x1, SOFT, SOFT)
    if show_later:
        band(ax, *LATER, x0, x1, "#f4f4f2", "#c8c7c2", ls=(0, (3, 2)))
    if labels:
        lx = x0 - 0.25 if lx is None else lx
        ax.text(lx, (ENC[0] + ENC[1]) / 2 + 0.17, "Encode  $G_0$", ha="right", va="center", fontsize=12,
                weight="bold", color=ENC_E)
        ax.text(lx, (ENC[0] + ENC[1]) / 2 - 0.2, "layers 0–2\nre-encodes each token", ha="right", va="top",
                fontsize=9, color=INK2, linespacing=1.1)
        ax.text(lx, (FEED[0] + FEED[1]) / 2 + 0.12, "$G_1$–$G_3$", ha="right", va="center", fontsize=11,
                color=INK2, weight="bold")
        ax.text(lx, (FEED[0] + FEED[1]) / 2 - 0.2, "feed the address,\nredundantly", ha="right", va="top",
                fontsize=9, color=INK2, linespacing=1.1)
        ax.text(lx, YA + 0.17, "Address  $G_4$", ha="right", va="center", fontsize=12, weight="bold", color=ADR_E)
        ax.text(lx, YA - 0.2, "layers 16–18\nrecurrent state", ha="right", va="top", fontsize=9, color=INK2,
                linespacing=1.1)
        ax.text(lx, YR + 0.1, "reader  L19", ha="right", va="center", fontsize=11, weight="bold", color=SOFT)
        ax.text(lx, YR - 0.2, "softmax head L19H11", ha="right", va="center", fontsize=9, color=INK2)
        ax.text(lx, YL + 0.12, "later layers", ha="right", va="center", fontsize=11, color=INK2, weight="bold")
        ax.text(lx, YL - 0.2, "incl. readers L23, L27", ha="right", va="center", fontsize=9, color=INK2)


def tokens(ax, cols, texts, subs=None, bold=(), col=None, fs=10):
    for i, (x, t) in enumerate(zip(cols, texts)):
        c = INK if col is None else col[i]
        ax.text(x, Y_TOK, t, ha="center", va="center", fontsize=fs + 0.5 if i in bold else fs,
                weight="bold" if i in bold else "normal", color=c)
        if subs and subs[i]:
            ax.text(x, Y_TOK - 0.36, subs[i], ha="center", va="center", fontsize=9, color=INK3)


def encode_ticks(ax, cols, c=ENC_E):
    for x in cols:
        arrow(ax, (x, Y_TOK + 0.22), (x, ENC[0] + 0.02), c, lw=1.1, hl=3.5, hw=2.2, z=3)
        ax.add_patch(Rectangle((x - 0.1, ENC[0] + 0.18), 0.2, 0.42, fc=c, ec="none", alpha=0.9, zorder=4))


def address_ribbon(ax, x_start, x_key, x_end, c, label_key=None, out=True, z=5):
    """Recurrent-state ribbon in the Address band: empty before the key, filled from the key on."""
    ribbon(ax, [x_start, x_key], YA, "#c9d6ea", lw=4, z=z)
    ribbon(ax, [x_key, x_end], YA, c, lw=5, z=z + 1)
    dot(ax, x_key, YA, c, r=0.17, z=z + 2)
    if label_key:
        ax.text(x_key, YA + 0.33, label_key, ha="center", va="center", fontsize=9.5, color=c, style="italic")
    if out:
        arrow(ax, (x_end, YA + 0.1), (x_end, RDR[0] - 0.02), c, lw=2.4, z=z + 2)


# ================================================================== (a) the circuit
axa = fig.add_axes([0.0, 0.0, 0.485, 1.0]); axa.set_xlim(-0.2, 10.7); axa.set_ylim(-1.5, 6.9); axa.axis("off")
axa.text(0.0, 6.6, "(a)  The Encode–Address circuit in Qwen3.5-9B", fontsize=14, weight="bold", color=INK,
         va="center")
axa.text(0.0, 6.25, "the Address block's recurrent state carries the queried key to the final token, "
         "where the reader forms its query from it", fontsize=9.5, color=INK2, va="center")

X0, X1 = 3.15, 10.5
cols = [3.7, 5.05, 6.4, 7.25, 8.55, 9.95]
XKEY = 8.75
texts = ["berry=amber", "banana=lime", "ginger=wood", "…", "", "**"]
subs = [None, None, "queried entry", None, "question", "final token"]
stack(axa, X0, X1, labels=True, lx=2.95)
axa.plot([cols[0] - 0.5, cols[3] + 0.3], [Y_TOK - 0.6] * 2, color=INK3, lw=1)
axa.text((cols[0] - 0.5 + cols[3] + 0.3) / 2, Y_TOK - 0.8, "dictionary (8 entries)", ha="center", va="center", fontsize=9.5,
         color=INK3)
tokens(axa, cols, texts, subs, bold=(2,), fs=9.5)
axa.text(cols[4] - 0.02, Y_TOK, "the value of", ha="right", va="center", fontsize=9.5, color=INK)
axa.text(cols[4] + 0.02, Y_TOK, "ginger", ha="left", va="center", fontsize=10, weight="bold", color=HOST,
         bbox=dict(boxstyle="round,pad=0.12", fc="#eaf1fc", ec="none"), zorder=4)
axa.text(cols[4] + 0.7, Y_TOK, "?", ha="left", va="center", fontsize=9.5, color=INK, zorder=5)
encode_ticks(axa, cols[:3] + [cols[4] - 0.6, XKEY, cols[5]])
# recurrent state ribbon: empty over the dictionary, picks up "ginger" in the question, exits at the final token
address_ribbon(axa, X0 + 0.3, XKEY, cols[5], HOST, label_key="picks up the key")
axa.text((X0 + 0.3 + XKEY - 0.9) / 2, YA + 0.33, "recurrent state, read along the prompt", ha="center",
         va="center", fontsize=9.5, color=INK3, style="italic")
axa.text(cols[5] - 0.15, ADR[1] + 0.17, "address, written at the final token only", ha="right", va="center",
         fontsize=9.5, color=HOST)
# feeders -> address: faint upward arrows from G1-G3 into the Address band
for x in (cols[4] - 0.6, XKEY, cols[5]):
    arrow(axa, (x, FEED[1] - 0.03), (x, ADR[0] - 0.02), "#b5b4ae", lw=1.0, hl=3.5, hw=2.2, z=3)
# reader: q at the final position, k at the queried entry, attention arc
dot(axa, cols[5], YR, HOST, r=0.2, txt="q", fs=10)
dot(axa, cols[2], YR, SOFT, r=0.2, txt="k", fs=10)
dot(axa, cols[1], YR, SOFT, r=0.11)
dot(axa, cols[0], YR, SOFT, r=0.11)
arrow(axa, (cols[5] - 0.18, YR + 0.2), (cols[2] + 0.18, YR + 0.2), HOST, lw=2.2, rad=0.25, hl=5, hw=3)
axa.text(X0 + 0.25, RDR[1] + 0.3, "the reader attends to the entry\nwhose key matches q  (0.83)",
         ha="left", va="center", fontsize=9.5, color=INK2, linespacing=1.1)
# value flows up to the output
arrow(axa, (cols[5], RDR[1] + 0.02), (cols[5], Y_OUT - 0.25), INK, lw=1.6, z=4)
axa.text(cols[5], Y_OUT, "wood", ha="center", va="center", fontsize=13, weight="bold", color=INK)
axa.text(cols[5] + 0.42, Y_OUT, "✓", ha="left", va="center", fontsize=12, color=OK, weight="bold")
axa.text(cols[5] - 0.45, Y_OUT, "answer", ha="right", va="center", fontsize=9.5, color=INK3)
# legend strip for layer kinds (bottom left, under the row labels)
lx0, ly = 0.0, -0.55
axa.add_patch(Rectangle((lx0, ly - 0.1), 0.32, 0.2, fc=LIN, ec="#c8c7c2", lw=0.8))
axa.text(lx0 + 0.4, ly, "linear attention (Gated DeltaNet)", va="center", fontsize=9, color=INK2)
axa.add_patch(Rectangle((lx0, ly - 0.45), 0.32, 0.2, fc=SOFT, ec=SOFT, lw=0.8))
axa.text(lx0 + 0.4, ly - 0.35, "softmax attention: one layer in four", va="center", fontsize=9, color=INK2)

# ================================================================== (b) removal vs transplant
axb = fig.add_axes([0.49, 0.0, 0.325, 1.0]); axb.set_xlim(-0.1, 7.95); axb.set_ylim(-1.5, 6.9); axb.axis("off")
axb.text(0.0, 6.6, "(b)  Two interventions on the Address block", fontsize=14, weight="bold", color=INK, va="center")
axb.text(0.0, 6.25, "removal says dispensable; transplant says decisive",
         fontsize=9.5, color=INK2, va="center")
bcols = [0.6, 1.9, 3.2]          # banana=lime, ginger=wood, final token
btexts = ["banana=lime", "ginger=wood", "**"]


def mini(ax, dx, title, sub):
    c = [x + dx for x in bcols]
    ax.text(c[0] - 0.5, Y_OUT, title, fontsize=11, weight="bold", color=INK, va="center", ha="left")
    tokens(ax, c, btexts, [None, "host asks ginger", "final"], bold=(), fs=9.3)
    return c


# --- left: removal
stack(axb, 0.1, 3.65, show_feed=False, addr_fc="white", addr_ec="#b5b4ae", addr_hatch="////")
c = mini(axb, 0.0, "Remove it", "")
encode_ticks(axb, c)
axb.text((c[0] + c[2]) / 2, YA, "output zeroed, all positions", ha="center", va="center", fontsize=9, color=INK2,
         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=6)
dot(axb, c[2], YR, "#b5b4ae", r=0.2, txt="q", fs=10)
dot(axb, c[1], YR, SOFT, r=0.11); dot(axb, c[0], YR, SOFT, r=0.11)
arrow(axb, (c[2] - 0.18, YR + 0.2), (c[1] + 0.18, YR + 0.2), "#b5b4ae", lw=1.6, ls=(0, (2, 2)), rad=0.45, hl=4, hw=2.5)
axb.text(0.2, RDR[1] + 0.3, "reader loses ginger\n0.83 → 0.06", ha="left", va="center", fontsize=8.5,
         color=INK2, linespacing=1.1)
# later reader L27 compensates
dot(axb, c[2], YL, INK, r=0.15); dot(axb, c[1], YL, SOFT, r=0.11)
arrow(axb, (c[2] - 0.14, YL + 0.16), (c[1] + 0.14, YL + 0.16), INK, lw=1.8, rad=0.5, hl=4.5, hw=2.8)
axb.text(0.22, YL, "a later reader\n(L27) finds it", ha="left", va="center", fontsize=8.5, color=INK2, linespacing=1.1)
arrow(axb, (c[2], LATER[1] + 0.02), (c[2], Y_OUT - 0.25), INK, lw=1.6, z=4)
axb.text(c[2], Y_OUT, "wood", ha="center", va="center", fontsize=13, weight="bold", color=INK)
axb.text(c[2] + 0.48, Y_OUT, "✓", ha="left", va="center", fontsize=12, color=OK, weight="bold")
axb.text((c[0] + c[2]) / 2, -0.55, "accuracy\nstays 1.00", ha="center", va="top", fontsize=10, color=INK,
         weight="bold", linespacing=1.1)
axb.text((c[0] + c[2]) / 2, -1.12, "→ removal alone would\ncall it dispensable", ha="center", va="top", fontsize=9.5,
         color=INK2, style="italic", linespacing=1.1)

# --- right: transplant
dx = 4.2
stack(axb, 0.1 + dx, 3.65 + dx, show_feed=False)
c = mini(axb, dx, "Transplant it", "")
encode_ticks(axb, c)
# host ribbon (asks ginger) faint; the donor's output replaces it at the final token
ribbon(axb, [0.4 + dx, c[2] - 0.35], YA, HOST, lw=4, alpha=0.35)
axb.text((0.4 + dx + c[2] - 0.35) / 2, YA + 0.3, "host state (asks ginger)", ha="center", va="center", fontsize=8.5,
         color=HOST, alpha=0.85)
# donor chip in the space of the feeder band
yc0, yc1 = FEED[0] + 0.02, FEED[1] - 0.02
axb.add_patch(FancyBboxPatch((0.25 + dx, yc0), 3.35, yc1 - yc0, boxstyle="round,pad=0.03,rounding_size=0.1",
                             fc="#fdece4", ec=DONOR, lw=1.2, zorder=7))
axb.text(0.4 + dx, (yc0 + yc1) / 2, "donor run: same dictionary,\nasks banana", ha="left", va="center",
         fontsize=8.5, color=DONOR, zorder=8, linespacing=1.1)
dot(axb, c[2], (yc0 + yc1) / 2, DONOR, r=0.14, z=9)
arrow(axb, (c[2], (yc0 + yc1) / 2 + 0.14), (c[2], RDR[0] - 0.02), DONOR, lw=2.4, z=8)
axb.text(c[2] - 0.25, YA - 0.27, "its Address output\nreplaces the host's here", ha="right", va="center",
         fontsize=8, color=DONOR, zorder=9, linespacing=1.05)
dot(axb, c[2], YA, DONOR, r=0.17, z=9)
dot(axb, c[2], YR, DONOR, r=0.2, txt="q", fs=10)
dot(axb, c[1], YR, SOFT, r=0.11); dot(axb, c[0], YR, SOFT, r=0.2, txt="k", fs=10)
arrow(axb, (c[2] - 0.18, YR + 0.2), (c[0] + 0.18, YR + 0.2), DONOR, lw=2.2, rad=0.3, hl=5, hw=3)
axb.text((c[0] + c[2]) / 2, YL, "reader moves to banana:  0.00 → 0.78", ha="center", va="center", fontsize=8.5,
         color=INK2, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=6)
arrow(axb, (c[2], RDR[1] + 0.02), (c[2], Y_OUT - 0.25), DONOR, lw=1.6, z=4)
axb.text(c[2], Y_OUT, "lime", ha="center", va="center", fontsize=13, weight="bold", color=DONOR)
axb.text((c[0] + c[2]) / 2, -0.55, "answers the donor's\nquestion: 0.99", ha="center", va="top", fontsize=10,
         color=INK, weight="bold", linespacing=1.1)
axb.text((c[0] + c[2]) / 2, -1.12, "→ its output alone decides\nwhat is retrieved", ha="center", va="top",
         fontsize=9.5, color=INK2, style="italic", linespacing=1.1)
# divider
axb.plot([3.95, 3.95], [-0.4, 6.0], color="#dcdbd7", lw=1)

# ================================================================== (c) across families
axc = fig.add_axes([0.815, 0.0, 0.185, 1.0]); axc.set_xlim(-0.3, 4.0); axc.set_ylim(-1.5, 6.9); axc.axis("off")
axc.text(-0.2, 6.6, "(c)  Transfer across architectures", fontsize=14, weight="bold", color=INK, va="center")
axc.text(-0.2, 6.25, "Qwen's address, linearly aligned,\nsteers Granite's readers", fontsize=9.5, color=INK2, va="top", linespacing=1.1)

QWEN_SOFT = {3, 7, 11, 15, 19, 23, 27, 31}
GRAN_SOFT = {5, 15, 25, 35}
Y_BOT, Y_TOP = 0.05, 5.0


def strip(ax, x, n, soft, w=0.55, highlight=(), readers=()):
    h = (Y_TOP - Y_BOT) / n
    for i in range(n):
        y = Y_BOT + i * h
        if i in soft:
            fc, ec = SOFT, SOFT
        elif any(a <= i <= b for a, b in highlight):
            fc, ec = ADR_F, ADR_E
        else:
            fc, ec = LIN, "#d2d1cc"
        ax.add_patch(Rectangle((x, y), w, h - 0.012, fc=fc, ec=ec, lw=0.6, zorder=2))
    for a, b in highlight:
        ax.add_patch(Rectangle((x - 0.03, Y_BOT + a * h - 0.01), w + 0.06, (b - a + 1) * h, fc="none", ec=ADR_E,
                               lw=1.6, zorder=3))
    for r in readers:
        ax.add_patch(Rectangle((x - 0.03, Y_BOT + r * h - 0.01), w + 0.06, h, fc="none", ec=INK, lw=1.4, zorder=3))
    return h


xq, xg = 0.25, 2.55
hq = strip(axc, xq, 32, QWEN_SOFT, highlight=[(16, 18)], readers=[19])
hg = strip(axc, xg, 40, GRAN_SOFT, highlight=[(16, 24), (26, 34)], readers=[25, 35])
axc.text(xq + 0.275, Y_TOP + 0.5, "Qwen3.5-9B", ha="center", va="center", fontsize=11, weight="bold", color=INK)
axc.text(xq + 0.275, Y_TOP + 0.22, "Gated DeltaNet\n32 layers", ha="center", va="center", fontsize=8.5, color=INK2, linespacing=1.1)
axc.text(xg + 0.275, Y_TOP + 0.5, "Granite4.0-7B", ha="center", va="center", fontsize=11, weight="bold", color=INK)
axc.text(xg + 0.275, Y_TOP + 0.22, "Mamba-2\n40 layers", ha="center", va="center", fontsize=8.5, color=INK2, linespacing=1.1)
# layer labels
axc.text(xq - 0.08, Y_BOT + 19.5 * hq, "L19", ha="right", va="center", fontsize=8.5, color=INK)
axc.text(xq - 0.08, Y_BOT + 17.5 * hq, "$G_4$", ha="right", va="center", fontsize=8.5, color=ADR_E)
axc.text(xq - 0.08, Y_BOT + 0.5 * hq, "0", ha="right", va="center", fontsize=7.5, color=INK3)
axc.text(xq - 0.08, Y_BOT + 31.5 * hq, "31", ha="right", va="center", fontsize=7.5, color=INK3)
axc.text(xg + 0.63, Y_BOT + 25.5 * hg, "L25", ha="left", va="center", fontsize=8.5, color=INK)
axc.text(xg + 0.63, Y_BOT + 35.5 * hg, "L35", ha="left", va="center", fontsize=8.5, color=INK)
axc.text(xg + 0.63, Y_BOT + 20.5 * hg, "$G_2$", ha="left", va="center", fontsize=8.5, color=ADR_E)
axc.text(xg + 0.63, Y_BOT + 30.5 * hg, "$G_3$", ha="left", va="center", fontsize=8.5, color=ADR_E)
axc.text(xg + 0.63, Y_BOT + 0.5 * hg, "0", ha="left", va="center", fontsize=7.5, color=INK3)
axc.text(xg + 0.63, Y_BOT + 39.5 * hg, "39", ha="left", va="center", fontsize=7.5, color=INK3)
# the address vector leaves Qwen's G4 at the final position, passes W, lands in Granite's G2 and G3
ya = Y_BOT + 17.5 * hq
dot(axc, xq + 0.275, ya, HOST, r=0.14, z=9)
wx = 1.55
axc.add_patch(FancyBboxPatch((wx - 0.33, ya - 0.26), 0.66, 0.52, boxstyle="round,pad=0.03,rounding_size=0.1",
                             fc="white", ec=HOST, lw=1.4, zorder=7))
axc.text(wx, ya + 0.02, "$W$", ha="center", va="center", fontsize=12, color=HOST, weight="bold", zorder=8)
arrow(axc, (xq + 0.58, ya), (wx - 0.33, ya), HOST, lw=2.0, head=False, z=6)
for yt in (Y_BOT + 20.5 * hg, Y_BOT + 30.5 * hg):
    arrow(axc, (wx + 0.33, ya), (xg - 0.03, yt), HOST, lw=2.0, rad=-0.15 if yt > ya else 0.15, z=6)
axc.text(wx, ya - 0.42, "alignment map\n(ridge regression)", ha="center", va="top", fontsize=8.5, color=INK2)
axc.text(wx, ya - 0.95, "no fine-tuning", ha="center", va="top", fontsize=8.5, color=INK2)
# outcome
axc.text(1.7, -0.55, "Granite answers Qwen's\nquestion: 0.99", ha="center", va="top", fontsize=10, color=INK,
         weight="bold", linespacing=1.1)
axc.text(1.7, -1.12, "→ into one of its two Address\nblocks alone: fails", ha="center", va="top", fontsize=9,
         color=INK2, style="italic", linespacing=1.1)

os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/overview_v2.{ext}", facecolor="white", bbox_inches="tight", pad_inches=0.08)
print("wrote figures/overview_v2.{png,pdf}")
