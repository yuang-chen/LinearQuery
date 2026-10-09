"""fp32 vs bf16 for the headline measurements (scripts/run_bf16_check.sh) -> results/bf16/compare.md"""
import json, os
os.chdir("/mnt/yuang/LinearQuery")
L = []
P = lambda *s: L.append(" ".join(str(x) for x in s))
ld = lambda f: json.load(open(f)) if os.path.exists(f) else None

P("# fp32 vs bf16 (chat8)\n")
P("## Baseline and block ablations (exp21 B1/B2): acc / gap\n")
P("| model | | baseline | " + " | ".join(f"G{i}" for i in range(16)) + " |")
P("|---|---|---|" + "---|" * 16)
for tag in ("0.8B", "9B", "G1b", "Gtiny", "N8B", "27B"):
    for prec, f in (("fp32", f"results/exp21/{tag}_chat8.json"), ("bf16", f"results/exp21/{tag}-bf16_chat8.json")):
        d = ld(f)
        if d is None:
            P(f"| {tag} | {prec} | missing |"); continue
        b = d["baseline"]
        cells = [f"{g['acc']:.2f} / {g['gap']:+.1f}" for g in d["groups"]]
        P(f"| {tag} | {prec} | {b['acc']:.2f} / {b['gap']:+.1f} | " + " | ".join(cells) + " |")


def rows(d, positions=("final",)):
    return {(r.get("group"), tuple(r["layers"]), r["positions"]): r for r in d["transplant"]
            if r["positions"] in positions}


P("\n## Single-block query transplant (exp22, final token): ansA / D(A-B) / attA\n")
P("| model | block | fp32 | bf16 |")
P("|---|---|---|---|")
for tag, f32 in (("0.8B", "exp22_transplant_0.8B"), ("9B", "exp22_transplant_9B"), ("G1b", "exp22_transplant_G1b_chat8"),
                 ("Gtiny", "exp22_transplant_Gtiny_chat8"), ("N8B", "exp22_transplant_N8B_chat8")):
    a, b = ld(f"results/{f32}.json"), ld(f"results/bf16/exp22_{tag}_chat8.json")
    if a is None or b is None:
        P(f"| {tag} | | {'ok' if a else 'missing'} | {'ok' if b else 'missing'} |"); continue
    ra, rb = rows(a), rows(b)
    for k in ra:
        x, y = ra[k], rb.get(k)
        fmt = lambda r: f"{r['ansA']:.2f} / {r['dAB']:+.1f} / {r['attA']:.2f}" if r else "—"
        P(f"| {tag} | G{k[0]} [{k[1][0]}–{k[1][-1]}] | {fmt(x)} | {fmt(y)} |")

P("\n## Combined transplants (exp22c): ansA / D(A-B) / attA, final and question\n")
P("| model | blocks | pos | fp32 | bf16 |")
P("|---|---|---|---|---|")
for tag in ("Gtiny", "N8B", "27B", "Q38-27B"):
    a, b = ld(f"results/exp22c_combined_{tag}_chat8.json"), ld(f"results/bf16/exp22c_{tag}_chat8.json")
    if a is None or b is None:
        P(f"| {tag} | | | {'ok' if a else 'missing'} | {'ok' if b else 'missing'} |"); continue
    names = dict(zip(map(tuple, a["meta"]["groups"]), a["meta"]["names"]))
    ra, rb = rows(a, ("final", "question")), rows(b, ("final", "question"))
    for k, x in ra.items():
        y = rb.get(k)
        fmt = lambda r: f"{r['ansA']:.2f} / {r['dAB']:+.1f} / {r['attA']:.2f}" if r else "—"
        nm = names.get(k[1], "?").replace("+G", "+")
        P(f"| {tag} | {nm} | {k[2]} | {fmt(x)} | {fmt(y)} |")

open("results/bf16/compare.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
