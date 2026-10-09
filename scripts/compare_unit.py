"""Mixer-only vs whole-layer (mixer + MLP) interventions, Qwen3.5-0.8B/9B/27B -> results/unit_layer/compare.md"""
import json, os
os.chdir("/mnt/yuang/LinearQuery")
L = []
P = lambda *s: L.append(" ".join(str(x) for x in s))
ld = lambda f: json.load(open(f)) if os.path.exists(f) else None
T22 = {("0.8B", "chat8"): "exp22_transplant_0.8B", ("9B", "chat8"): "exp22_transplant_9B"}
f22 = lambda t, v: f"results/{T22.get((t, v), f'exp22_transplant_{t}_{v}')}.json"

P("# Mixer-only vs whole-layer (mixer + MLP) interventions\n")
for tag in ("0.8B", "9B", "27B"):
    P(f"## {tag}\n")
    P("### Block ablation: acc / gap\n")
    for v in ("chat8", "list8", "rev8"):
        m, l = ld(f"results/exp21/{tag}_{v}.json"), ld(f"results/exp21/{tag}-layer_{v}.json")
        if m is None or l is None:
            P(f"{v}: missing"); continue
        n = len(m["groups"])
        if v == "chat8":
            P("| variant | unit | baseline | " + " | ".join(f"G{i}" for i in range(n)) + " |")
            P("|---|---|---|" + "---|" * n)
        for unit, d in (("mixer", m), ("layer", l)):
            P(f"| {v} | {unit} | {d['baseline']['acc']:.2f} / {d['baseline']['gap']:+.1f} | "
              + " | ".join(f"{g['acc']:.2f} / {g['gap']:+.1f}" for g in d["groups"]) + " |")
    P("\n### Single-block transplant, final token: ansA / D(A-B) / attA (question span ansA in brackets)\n")
    P("| variant | block | mixer | layer |")
    P("|---|---|---|---|")
    for v in ("chat8", "list8", "rev8"):
        m, l = ld(f22(tag, v)), ld(f"results/unit_layer/exp22_{tag}_{v}.json")
        if m is None or l is None:
            P(f"| {v} | missing | | |"); continue
        def tab(d):
            o = {}
            for r in d["transplant"]:
                o.setdefault(r["group"], {})[r["positions"]] = r
            return o
        tm, tl = tab(m), tab(l)
        for g in sorted(tm):
            fm = lambda x: (f"{x['final']['ansA']:.2f} / {x['final']['dAB']:+.1f} / {x['final']['attA']:.2f} "
                            f"[{x['question']['ansA']:.2f}]") if x else "—"
            lay = m["meta"]["groups"][g]
            P(f"| {v} | G{g} [{lay[0]}–{lay[-1]}] | {fm(tm[g])} | {fm(tl.get(g))} |")
    if tag == "27B":
        P("\n### Combined transplants: ansA / D(A-B) / attA\n")
        P("| variant | blocks | pos | mixer | layer |")
        P("|---|---|---|---|---|")
        for v in ("chat8", "list8", "rev8"):
            m, l = ld(f"results/exp22c_combined_27B_{v}.json"), ld(f"results/unit_layer/exp22c_27B_{v}.json")
            if m is None or l is None:
                P(f"| {v} | missing | | | |"); continue
            key = lambda r: (tuple(r["layers"]), r["positions"])
            rl = {key(r): r for r in l["transplant"]}
            names = dict(zip(map(tuple, m["meta"]["groups"]), m["meta"]["names"]))
            for r in m["transplant"]:
                y = rl.get(key(r))
                fmt = lambda x: f"{x['ansA']:.2f} / {x['dAB']:+.1f} / {x['attA']:.2f}" if x else "—"
                nm = names[tuple(r["layers"])]
                nm = f"{nm.split('+')[0]}–{nm.split('+')[-1]}" if nm.count("+") > 2 else nm
                P(f"| {v} | {nm} | {r['positions']} | {fmt(r)} | {fmt(y)} |")
    P("")
open("results/unit_layer/compare.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
