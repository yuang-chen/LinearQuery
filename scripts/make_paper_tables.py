"""Emit the appendix LaTeX tables straight from results/, so the paper cannot drift from data."""
import json, glob, os
V = ["chat8", "chat4", "chat16", "list8", "rev8", "perm8", "long512", "mmlu", "mmlu_hint"]
MODELS = [("0.8B", r"\qsmall{}"), ("9B", r"\qlarge{}"), ("G1b", r"\gsmall{}"), ("Gtiny", r"\glarge{}")]
out = []
P = out.append

P(r"\section{Full block-ablation tables}" + "\n" + r"\label{app:blocks}" + "\n")
for tag, name in MODELS:
    rows = []
    for v in V:
        f = f"results/exp21/{tag}_{v}.json"
        if v.startswith("mmlu"):          # the standard 5-shot runs supersede the bare-prompt ones
            f5 = f"results/exp21/{tag}-5shot_{v}.json"
            f = f5 if os.path.exists(f5) else f
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        g = " & ".join(f"{r['acc']:.2f}" for r in d["groups"])
        rows.append((v, d["meta"]["n"], d["baseline"]["acc"], d["baseline"]["gap"], g,
                     len(d["groups"])))
    if not rows:
        continue
    ng = rows[0][5]
    P(r"\begin{table}[h]\centering\small")
    P(rf"\caption{{Block ablation, {name}: accuracy per task variant.}}")
    P(r"\begin{tabular}{lcc" + "c" * ng + "}")
    P(r"\toprule")
    P("variant & $n$ & baseline & " + " & ".join(f"$G_{i}$" for i in range(ng)) + r" \\")
    P(r"\midrule")
    for v, n, acc, gap, g, _ in rows:
        P(rf"\ttt{{{v.replace('_', chr(92) + '_')}}} & {n} & {acc:.2f} / ${gap:+.1f}$ & {g} \\")
    P(r"\bottomrule\end{tabular}\end{table}" + "\n")

P(r"\section{Single-layer ablation}" + "\n" + r"\label{app:layers}" + "\n")
P(r"\begin{table}[h]\centering\small")
P(r"\caption{Layers whose removal leaves the gap below 60\% of baseline (\ttt{chat8}), "
  r"and the worst single linear head of layer 0.}")
P(r"\begin{tabular}{llc}")
P(r"\toprule")
P(r"Model & critical single layers (accuracy) & worst layer-0 head \\")
P(r"\midrule")
for tag, name in MODELS:
    f = f"results/exp21/{tag}_chat8.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f)); b = d["baseline"]["gap"]
    crit = [r for r in d["layers"] if r["gap"] < 0.6 * b]
    cs = ", ".join(f"L{r['layer']} ({r['acc']:.2f})" for r in crit) or "---"
    w = sorted(d["l0_heads"], key=lambda r: r["gap"])[0]
    P(rf"{name} & {cs} & h{w['head']} ({w['acc']:.2f}) \\")
P(r"\bottomrule\end{tabular}\end{table}" + "\n")

P(r"\section{Transplant, all blocks and position sets}" + "\n" + r"\label{app:transplant}" + "\n")
P(r"\begin{table}[h]\centering\small")
P(r"\caption{Query transplant (\ttt{chat8}), $\acc_A$ for every block and position set. "
  r"\emph{dict} is the control.}")
P(r"\begin{tabular}{llcccc}")
P(r"\toprule")
P(r"Model & block & final & question & dict & all \\")
P(r"\midrule")
for tag, name in MODELS:
    f = f"results/exp22_transplant_{tag}_chat8.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    byg = {}
    for r in d["transplant"]:
        byg.setdefault(r["group"], {})[r["positions"]] = r["ansA"]
    for gi, rs in sorted(byg.items()):
        P(rf"{name if gi == 0 else ''} & $G_{gi}$ & " +
          " & ".join(f"{rs.get(p, float('nan')):.2f}" for p in ("final", "question", "dict", "all")) + r" \\")
    P(r"\midrule")
out[-1] = r"\bottomrule\end{tabular}\end{table}" + "\n"

open("paper/tex/generated_tables.tex", "w").write("\n".join(out))
print("\n".join(out[:24]))
print(f"... wrote paper/tex/generated_tables.tex ({len(out)} lines)")
