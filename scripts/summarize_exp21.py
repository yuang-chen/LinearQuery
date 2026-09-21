"""Aggregate results/exp21/*.json into markdown tables (results/exp21/summary.md)."""
import json, glob, os

VARIANTS = ["chat8", "chat4", "chat16", "list8", "rev8", "perm8", "long512", "long2048", "mmlu",
            "mmlu_hint"]
MODELS = ["0.8B", "9B"]
R = {}
for f in glob.glob("results/exp21/*.json"):
    d = json.load(open(f))
    R[(d["meta"]["tag"], d["meta"]["variant"])] = d
out = []
P = out.append


def rows(m):
    return [(v, R[(m, v)]) for v in VARIANTS if (m, v) in R]


for m in MODELS:
    if not rows(m):
        continue
    P(f"\n## {m}\n")
    ng = len(rows(m)[0][1]["meta"]["groups"])
    P("### Baseline and group ablations (acc / gap)\n")
    P("| variant | n | tokens | baseline | " + " | ".join(f"G{i}" for i in range(ng)) + " |")
    P("|---|---|---|---|" + "---|" * ng)
    for v, d in rows(m):
        b = d["baseline"]
        g = " | ".join(f"{r['acc']:.2f} / {r['gap']:+.1f}" for r in d["groups"])
        P(f"| {v} | {d['meta']['n']} | {d['meta']['tokens']} | {b['acc']:.2f} / {b['gap']:+.1f} | {g} |")

    P("\n### Critical single GDN layers (gap < 60 % of baseline), and layer-0 heads\n")
    P("| variant | critical layers (acc / gap) | worst L0 head (acc / gap) | 2nd worst L0 head |")
    P("|---|---|---|---|")
    for v, d in rows(m):
        bg = d["baseline"]["gap"]
        crit = [r for r in d["layers"] if r["gap"] < 0.6 * bg]
        cs = ", ".join(f"L{r['layer']} ({r['acc']:.2f} / {r['gap']:+.1f})" for r in crit) or "—"
        w = sorted(d["l0_heads"], key=lambda r: r["gap"])
        P(f"| {v} | {cs} | h{w[0]['head']} ({w[0]['acc']:.2f} / {w[0]['gap']:+.1f}) | "
          f"h{w[1]['head']} ({w[1]['acc']:.2f} / {w[1]['gap']:+.1f}) |")

    P("\n### Reader head (top-8 by attention, chosen by largest gap drop when zeroed)\n")
    P("| variant | reader | attn on target entry | gap when zeroed | top-3 by attention |")
    P("|---|---|---|---|---|")
    for v, d in rows(m):
        RL, RH = d["meta"]["reader"]
        r = next(x for x in d["reader_heads"] if x["layer"] == RL and x["head"] == RH)
        top = ", ".join(f"L{x['layer']}H{x['head']} {x['target']:.2f}" for x in d["reader_heads"][:3])
        P(f"| {v} | L{RL}H{RH} | {r['target']:.2f} | {r['zero_gap']:+.1f} (base {d['baseline']['gap']:+.1f}) | {top} |")

    P("\n### q·k selectivity at the reader (s_target − mean s_other, top-1 among entries)\n")
    P("Gr-1 = GDN group right before the reader layer (G3 analogue); Gr-2 = the one before (G2 analogue). "
      "Loss = drop from intact when only K or only Q comes from the ablated run.\n")
    P("| variant | group | intact | ablated K only | ablated Q only | both | K-side loss | Q-side loss |")
    P("|---|---|---|---|---|---|---|---|")
    for v, d in rows(m):
        for gname in ("Gr-1", "Gr-2"):
            s = {(r["q"], r["k"]): r for r in d["selectivity"] if r["group"] == gname}
            if not s:
                continue
            ii = s[("intact", "intact")]; ik = s[("intact", gname)]
            qi = s[(gname, "intact")]; qq = s[(gname, gname)]
            f = lambda r: f"{r['selectivity']:.2f} ({r['top1']:.2f})"
            lay = next(r["layers"] for r in d["selectivity"] if r["group"] == gname)
            P(f"| {v} | {gname} {lay} | {f(ii)} | {f(ik)} | {f(qi)} | {f(qq)} | "
              f"{ii['selectivity'] - ik['selectivity']:+.2f} | {ii['selectivity'] - qi['selectivity']:+.2f} |")

    P("\n### Reader attention split (target / other entries / first token)\n")
    P("| variant | intact | Gr-1 removed | Gr-2 removed |")
    P("|---|---|---|---|")
    for v, d in rows(m):
        sp = d["reader_split"]
        f = lambda x: f"{x['target']:.2f} / {x['others']:.2f} / {x['first']:.2f}"
        P(f"| {v} | {f(sp['intact'])} | {f(sp['Gr-1'])} | {f(sp['Gr-2'])} |")

txt = "\n".join(out)
open("results/exp21/summary.md", "w").write(txt)
print(txt)
