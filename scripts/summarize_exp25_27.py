"""Markdown tables for exp25 (minimal hybrid), exp26 (cross-family query transfer) and exp27
(cross-family block transplant) -> results/exp25_27_summary.md"""
import json, glob, os, sys
import numpy as np

os.chdir("/mnt/yuang/LinearQuery")
L = []
P = lambda *s: L.append(" ".join(str(x) for x in s))
f2 = lambda x: "—" if x is None else f"{x:.2f}"

# ------------------------------------------------------------------ exp25
P("# exp25 — minimal hybrid (training-free, GnA-style)\n")
for tag in ("0.8B", "9B", "G1b", "Gtiny"):
    fn = f"results/exp25_minimal_{tag}.json"
    if not os.path.exists(fn):
        continue
    r = json.load(open(fn)); m = r["meta"]
    P(f"## {tag}  (softmax {m['attn']}; circuit bind {m['circuit']['bind']} query {m['circuit']['query']}; "
      f"cut* = {m.get('cut_star')})\n")
    P("### P1 truncation (keep layers 0..cut)\n")
    P("| cut | dict acc | dict cacc | chat8 | list8 | rev8 | kv20 | PPL |")
    P("|---|---|---|---|---|---|---|---|")
    for x in r.get("P1", []):
        d = x["dict"]
        P(f"| {x['cut']} | {x['dict_mean']:.3f} | {x['dict_cmean']:.3f} | "
          + " | ".join(f"{d[v]['acc']:.2f}/{d[v]['cacc']:.2f}" for v in ("chat8", "list8", "rev8"))
          + f" | {x['kv20']:.3f} | {x['ppl']:.1f} |")
    if "P2" in r:
        P("\n### P2 single linear layer removed (mean): layers that cost > 5 points of dev accuracy\n")
        for dn, v in r["P2"].items():
            crit = [f"L{x['layer']}:{x['dev']:.2f}" for x in v["rows"] if x["dev"] < v["base_dev"] - 0.05]
            P(f"- {dn} ({v['key']}, base {v['base_dev']:.3f}): " + (", ".join(crit) or "none"))
    if "P3" in r:
        P("\n### P3 greedy minimal set (mean removal)\n")
        for dn, v in r["P3"].items():
            P(f"- {dn}: {len(v['minimal'])} linear layers kept {v['minimal']} (ref {v['ref']:.3f}, "
              f"{len(v['path'])} removed)")
    if "P4" in r:
        P("\n### P4 configurations (test suite)\n")
        P("| depth | removal | set | k | dict acc | dict cacc | chat8 | list8 | rev8 | long512 | kv20 | PPL |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|")
        rows = r["P4"]
        # collapse random seeds into mean ± sd
        groups = {}
        for x in rows:
            key = (x["depth"], x["removal"], x["name"].split("_s")[0] if "random" in x["name"] else x["name"])
            groups.setdefault(key, []).append(x)
        for (dp, rm, nm), xs in groups.items():
            mean = lambda f: np.mean([f(x) for x in xs])
            sd = lambda f: np.std([f(x) for x in xs])
            cell = lambda f: (f"{mean(f):.2f}±{sd(f):.2f}" if len(xs) > 1 else f"{mean(f):.2f}")
            P(f"| {dp} | {rm} | {nm}{' (x%d)' % len(xs) if len(xs) > 1 else ''} | {xs[0]['n_linear']} | "
              f"{cell(lambda x: x['dict_mean'])} | {cell(lambda x: x['dict_cmean'])} | "
              + " | ".join(cell(lambda x, v=v: x['dict'][v]['acc']) for v in ("chat8", "list8", "rev8", "long512"))
              + f" | {cell(lambda x: x['kv20'])} | {cell(lambda x: x['ppl'])} |")
    if r.get("P5"):
        P("\n### P5 lm-eval (mean removal)\n")
        ks = list(r["P5"][0]["knowledge"])
        P("| depth | set | " + " | ".join(ks) + " | MMLU-5 | FDA | SWDE |")
        P("|---|---|" + "---|" * (len(ks) + 3))
        for x in r["P5"]:
            P(f"| {x['depth']} | {x['name']} | " + " | ".join(f2(x['knowledge'].get(k)) for k in ks)
              + f" | {f2(x['mmlu'])} | {f2(x['recall'].get('fda'))} | {f2(x['recall'].get('swde'))} |")
    P("")

# ------------------------------------------------------------------ exp26
P("# exp26 — query-vector transfer across families (ridge map, unseen keys)\n")
P("| donor → receiver | variant | host ansB | in-family ansA | **xfam ansA** | attn A | xfam-self ansB | shuffled ansA | mean ansA | other donor blocks ansA | emb ansA |")
P("|---|---|---|---|---|---|---|---|---|---|---|")
curves = []
for fn in sorted(f for f in glob.glob("results/exp26_xquery_*.json") if "_seed" not in f):
    r = json.load(open(fn)); m = r["meta"]; C = r["conditions"]
    oth = "  ".join(f"{k[4:]}:{v['ansA']:.2f}" for k, v in C.items() if k.startswith("src=G"))
    P(f"| {m['donor']} → {m['recv']} | {m['variant']} | {C['host (no transplant)']['ansB']:.2f} | "
      f"{C['in-family']['ansA']:.2f} | **{C['xfam']['ansA']:.2f}** | {C['xfam']['attA']:.2f} | "
      f"{C['xfam-self']['ansB']:.2f} | {C['xfam-shuffled']['ansA']:.2f} | {C['mean']['ansA']:.2f} | "
      f"{oth} | {C.get('src=emb', {}).get('ansA', float('nan')):.2f} |")
    if "curve" in r:
        curves.append((m, r["curve"]))
P("\n## Learning curves: ansA vs number of paired fit prompts (query block marked *)\n")
for m, cv in curves:
    ns = [x["n"] for x in next(iter(cv.values()))]
    P(f"**{m['donor']} → {m['recv']} ({m['variant']})**\n")
    P("| source | " + " | ".join(map(str, ns)) + " |")
    P("|---|" + "---|" * len(ns))
    for s, xs in cv.items():
        P(f"| {s}{'*' if s == m['query_source'] else ''} | " + " | ".join(f"{x['ansA']:.2f}" for x in xs) + " |")
    P("")

# ------------------------------------------------------------------ exp27
P("# exp27 — stitched query-block transplant across families (GnA §5.4 style, no training)\n")
for fn in sorted(f for f in glob.glob("results/exp27_xblock_*.json") if "_seed" not in f):
    r = json.load(open(fn)); m = r["meta"]; C = r["conditions"]
    P(f"**{m['donor']} blocks into {m['recv']} (replace layers {m['recv_block'][0]}–{m['recv_block'][1]}), "
      f"{m['variant']}**, n_test {m['n_test']}\n")
    P("| condition | acc | cacc | reader attn | WikiText PPL |")
    P("|---|---|---|---|---|")
    for k, c in C.items():
        b = "**" if "*" in k else ""
        P(f"| {b}{k}{b} | {c['acc']:.3f} | {c['cacc']:.3f} | {c['att']:.2f} | {c['ppl']:.1f} |")
    P("")

# ------------------------------------------------------------------ seed replicates
P("# Seed replicates (seed 0 = main run; seeds 1, 2 = new key split and dictionaries)\n")
P("| experiment | seed 0 | seed 1 | seed 2 | mean ± sd |")
P("|---|---|---|---|---|")
for kind, pat, cond in (("exp26 xfam ansA", "results/exp26_xquery_{}_to_{}_chat8{}.json", "xfam"),
                        ("exp27 query-block acc", "results/exp27_xblock_{}_into_{}_chat8{}.json", None)):
    for d_, r_ in (("0.8B", "G1b"), ("G1b", "0.8B")):
        vals = []
        for sfx in ("", "_seed1", "_seed2"):
            fn = pat.format(d_, r_, sfx)
            if not os.path.exists(fn):
                vals.append(None); continue
            C = json.load(open(fn))["conditions"]
            k = cond or next(k for k in C if k.endswith("*") and "nomix" not in k)
            vals.append(C[k]["ansA" if cond else "acc"])
        got = [v for v in vals if v is not None]
        P(f"| {kind} {d_} → {r_} | " + " | ".join(f2(v) for v in vals)
          + (f" | {np.mean(got):.3f} ± {np.std(got):.3f} |" if got else " | — |"))
open("results/exp25_27_summary.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
