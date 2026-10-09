"""GnA-style pruning curve (cf. GnA Fig. 2) from exp30: the model keeps layers 0..L (embeddings ->
layers 0..L -> final norm -> LM head); every later layer is removed. One point per L.
usage: plot_prune.py [0.8B|9B|G1b] [knowledge|lm]

Both curves are GnA-style normalised accuracy (their Fig. 2): (score - chance) / (full - chance), so 0 =
random guessing and 1 = the full model; values below chance are shown as 0, as in GnA's figure.
  retrieval  dictionary task, accuracy choosing among the 8 dictionary values (GnA-style answer
             scoring), mean of chat8 / chat16 / list8 / rev8 / long512; chance 1/8 (exp30)
  knowledge  GnA's knowledge set and metric (their notebook's eval_knowledge: ARC-C, ARC-E, PIQA,
             Winogrande, OpenBookQA, HellaSwag; lm-eval zero-shot, acc, mean of the six), the mean
             normalised as a mean against the mean chance level 1/3 (per-task normalisation is unstable: the
             full 0.8B model is within a few points of chance on ARC-C/Winogrande and below it on OBQA) (exp30b)
  lm         (appendix variant) WikiText-2 loss: (ln V - NLL) / (ln V - NLL_full), 0 = uniform guess
Below the plot, an architecture strip: the layer each step adds (dark = softmax attention, light =
linear), with the circuit's Encode and Address blocks and the reader (exp21).
-> figures/prune_curve_<tag>[_lm].{png,pdf}
"""
import json, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

os.chdir("/mnt/yuang/LinearQuery")
TAG = sys.argv[1] if len(sys.argv) > 1 else "0.8B"
SECOND = sys.argv[2] if len(sys.argv) > 2 else "knowledge"
CFG = {"0.8B": dict(name="Qwen3.5-0.8B", V=248320, bind=(0, 2), address=(12, 14), reader=15),
       "9B":   dict(name="Qwen3.5-9B", V=248320, bind=(0, 2), address=(16, 18), reader=19),
       "G1b":  dict(name="Granite4.0-1B", V=100352, bind=(0, 4), address=(16, 24), reader=25)}[TAG]
CHANCE = {"arc_challenge": .25, "arc_easy": .25, "piqa": .5, "winogrande": .5, "openbookqa": .25, "hellaswag": .25}
INK, INK2, INK3, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SOFT, LIN, BAND = "#52514e", "#dcdbd6", "#efeeea"
BLUE, ORANGE = "#2a78d6", "#eb6834"                # categorical slots 1-2 of the reference palette
r = json.load(open(f"results/exp30_prune_{TAG}.json"))
N, ATTN = r["meta"]["n_layers"], set(r["meta"]["attn"])
rows = sorted(r["rows"], key=lambda e: e["kept"])
k = [e["kept"] - 1 for e in rows]                  # x = index of the last layer kept (0-based)
norm = lambda x, full, ch: max(0.0, (x - ch) / (full - ch))     # GnA: 0 = chance, 1 = full, clipped at 0
retr = [norm(e["dict_cmean"], rows[-1]["dict_cmean"], 1 / 8) for e in rows]
if SECOND == "knowledge":
    kr = {e["kept"]: e for e in json.load(open(f"results/exp30b_knowledge_{TAG}.json"))["rows"]}
    score = lambda e, t: e["acc"][t]                # GnA's eval_knowledge uses acc
    full = kr[N]
    mean = lambda e: sum(score(e, t) for t in CHANCE) / len(CHANCE)
    ch_mean = sum(CHANCE.values()) / len(CHANCE)
    sec = [norm(mean(kr[kk + 1]), mean(full), ch_mean) for kk in k]
    SEC_LABEL = "Knowledge: ARC-C/E, PIQA, Winogrande, OBQA, HellaSwag (mean)"
else:
    nll = [math.log(e["ppl"]) for e in rows]
    lnV = math.log(CFG["V"])
    sec = [max(0.0, (lnV - x) / (lnV - nll[-1])) for x in nll]
    SEC_LABEL = "Language modelling: WikiText-2 (% of full-model loss reduction)"
lm = sec

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
fig = plt.figure(figsize=(7.8, 3.25), dpi=200)
fig.patch.set_facecolor(SURF)
gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 0.72], hspace=0.13, left=0.09, right=0.98, top=0.98, bottom=0.02)
ax, st = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for a_ in (ax, st):
    a_.set_facecolor(SURF)

# ---- main panel
R0 = CFG["reader"]
ax.axvspan(R0 - 0.5, R0 + 0.5, color=BAND, zorder=0)   # the reader layer
ax.plot(k, retr, color=BLUE, lw=2, marker="o", ms=4.5, mec=SURF, mew=0.8, zorder=3,
        label="Retrieval: dictionary lookup (choice among 8 values)")
ax.plot(k, lm, color=ORANGE, lw=2, marker="o", ms=4.5, mec=SURF, mew=0.8, zorder=3, label=SEC_LABEL)
i = k.index(R0)
ax.annotate(f"reader L{R0} kept\n→ retrieval appears", xy=(R0, retr[i]),
            xytext=(R0 - 10.5, 0.48),
            fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
ax.set_xlim(-0.6, N - 0.4); ax.set_ylim(-0.02, 1.05)
ax.text(N - 0.6, 0.015, "chance (below-chance values shown as 0)" if SECOND == "knowledge"
        else "uniform guess (worse shown as 0)", color=INK2, fontsize=8, ha="right", va="bottom")
ax.set_xticks(range(N)); ax.tick_params(axis="x", labelbottom=False, length=0)
ax.set_ylabel("Normalized accuracy\n(0 = chance, 1 = full model)" if SECOND == "knowledge"
              else "Normalized LM score\n(0 = uniform guess, 1 = full model)")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "bottom"):
    ax.spines[s].set_visible(False)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), frameon=False, fontsize=8.3)

# ---- architecture strip: box at x = k is the layer that step adds (layer k-1)
for L in range(N):
    st.add_patch(Rectangle((L - 0.42, 0.55), 0.84, 0.9, color=SOFT if L in ATTN else LIN, lw=0))
    st.text(L, 0.25, str(L), ha="center", va="center", fontsize=6.5 if N > 32 else 7, color=INK2)
def bracket(lo, hi, text):                         # under layers lo..hi
    st.plot([lo - 0.4, lo - 0.4, hi + 0.4, hi + 0.4], [-0.05, -0.2, -0.2, -0.05], color=INK2, lw=0.9)
    st.text((lo + hi) / 2, -0.32, text, ha="center", va="top", fontsize=8, color=INK)
bracket(*CFG["bind"], "Encode")
bracket(*CFG["address"], "Address")
st.plot([R0 - 0.4, R0 - 0.4, R0 + 0.4, R0 + 0.4], [-0.05, -0.2, -0.2, -0.05], color=INK2, lw=0.9)
st.text(R0 + 0.6, -0.32, f"Reader L{R0}", ha="left", va="top", fontsize=8, color=INK)
st.set_xlim(ax.get_xlim()); st.set_ylim(-0.7, 1.5); st.axis("off")
st.text(-0.6, 1.58, "Layer index L  ·  the point above layer L is the model keeping layers 0…L",
        fontsize=8, color=INK, va="bottom")
st.legend(handles=[Patch(color=SOFT, label="softmax attention"), Patch(color=LIN, label="linear (Gated DeltaNet)"
                   if TAG != "G1b" else "linear (Mamba-2)")],
          loc="lower right", bbox_to_anchor=(1.0, 0.93), ncol=2, frameon=False, fontsize=7.5, handlelength=1.0,
          bbox_transform=st.transAxes, borderaxespad=0)

os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/prune_curve_{TAG}{'' if SECOND == 'knowledge' else '_lm'}.{ext}", facecolor=SURF,
                bbox_inches="tight", pad_inches=0.12)
for kk, a_, b_ in zip(k, retr, lm):
    print(f"k {kk:2d}  retrieval {a_:5.2f}  second {b_:5.2f}")
