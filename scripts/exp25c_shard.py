"""Step 25c: finish exp25 P4/P5 (and the exp25b matched lm-eval) for one model in parallel shards.

Reads the sequential phases already saved in results/exp25_minimal_<tag>.json (cut*, the P2 PPL
ranking, the P3 greedy sets, and any P4 rows done), rebuilds exactly the configuration lists of
exp25_minimal.py / exp25b_lmeval_matched.py, skips what is done, and evaluates every n-th of the
rest. `--merge` folds the shard files back into the main result files."""
import sys, json, argparse, random, time, glob, os
import numpy as np
import torch
sys.path.insert(0, "/mnt/yuang/LinearQuery")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/mnt/yuang/models/Qwen3.5-9B")
ap.add_argument("--tag", default="9B")
ap.add_argument("--shard", default="1/1")
ap.add_argument("--n_random", type=int, default=5)
ap.add_argument("--n_random_b", type=int, default=3)
ap.add_argument("--lm_limit", type=int, default=500)
ap.add_argument("--mmlu_bs", type=int, default=4, help="MMLU 5-shot batch (fp32 9B: 16 runs out of memory)")
ap.add_argument("--merge", action="store_true")
a = ap.parse_args()
MAIN = f"results/exp25_minimal_{a.tag}.json"
src = json.load(open(MAIN))
m = src["meta"]; C = m["circuit"]; LIN = m["linear"]; cut_star = m["cut_star"]
DEPTHS = {"full": None, "trunc": cut_star}
greedy = {d: src["P3"][d]["minimal"] for d in DEPTHS}
ppl_rank = {d: [r["layer"] for r in sorted(src["P2"][d]["rows"], key=lambda r: -r["ppl"])] for d in DEPTHS}
lin_upto = lambda cut: [L for L in LIN if cut is None or L <= cut]


def named_sets(dname, cut):                      # identical to exp25_minimal.named_sets
    lin = lin_upto(cut)
    bq = sorted(set(C["bind"]) | set(C["query"])); k = len(bq)
    pre = [L for L in lin if L < C["reader"]]
    S = {"all": None, "none": [], "bind": C["bind"], "query": C["query"], "bind+query": bq,
         "pre-reader": pre, "greedy": greedy[dname], f"ppl-top{k}": sorted(ppl_rank[dname][:k]),
         f"first{k}": lin[:k], f"last{k}": lin[-k:]}
    rng = random.Random(0)
    for s in range(a.n_random):
        S[f"random{k}_s{s}"] = sorted(rng.sample(lin, k))
    kg = len(greedy[dname])
    S[f"ppl-top{kg}(greedy-size)"] = sorted(ppl_rank[dname][:kg])
    for s in range(a.n_random):
        S[f"random{kg}(greedy-size)_s{s}"] = sorted(rng.sample(lin, kg))
    return S


# ---- the full job list, in a fixed order
jobs = []
for dname, cut in DEPTHS.items():
    for name, keep in named_sets(dname, cut).items():
        for removal in (("mean", "zero") if keep is not None else ("-",)):
            jobs.append(("P4", dname, name, keep, removal))
bq = sorted(set(C["bind"]) | set(C["query"])); k = len(bq)
for dname, name, keep in [("full", "all", None), ("trunc", "all", None), ("full", "bind+query", bq),
                          ("trunc", "bind+query", bq), ("trunc", "greedy", greedy["trunc"]),
                          ("full", "greedy", greedy["full"]),
                          ("trunc", f"ppl-top{k}", sorted(ppl_rank["trunc"][:k])),
                          ("full", f"ppl-top{k}", sorted(ppl_rank["full"][:k])),
                          ("trunc", f"random{k}_s0", sorted(random.Random(0).sample(lin_upto(cut_star), k))),
                          ("full", "none", [])]:
    jobs.append(("P5", dname, name, keep, "mean"))
for name, keep in {"greedy": greedy["full"], "pre-reader": [L for L in LIN if L < C["reader"]]}.items():
    kk = len(keep)
    jobs.append(("B", "full", name, sorted(keep), "mean"))
    jobs.append(("B", "full", f"ppl-top{kk}", sorted(ppl_rank["full"][:kk]), "mean"))
    rng = random.Random(kk)
    for s in range(a.n_random_b):
        jobs.append(("B", "full", f"random{kk}_s{s}", sorted(rng.sample(LIN, kk)), "mean"))

done = {("P4", r["depth"], r["name"], r["removal"]) for r in src.get("P4", [])}
done |= {("P5", r["depth"], r["name"], "mean") for r in src.get("P5", [])}
if os.path.exists(f"results/exp25b_lmeval_{a.tag}.json"):
    done |= {("B", "full", r["name"], "mean") for r in json.load(open(f"results/exp25b_lmeval_{a.tag}.json"))["rows"]}

if a.merge:
    shards = sorted(glob.glob(f"results/exp25c_{a.tag}_shard*.json"))
    got = {}
    for fn in shards:
        for r in json.load(open(fn)):
            got[(r["phase"], r["depth"], r["name"], r["removal"])] = r
    P4, P5, B = list(src.get("P4", [])), list(src.get("P5", [])), []
    bfile = f"results/exp25b_lmeval_{a.tag}.json"
    bres = json.load(open(bfile)) if os.path.exists(bfile) else dict(meta=dict(tag=a.tag), rows=[])
    for j in jobs:
        key = (j[0], j[1], j[2], j[4])
        if key in done or key not in got:
            continue
        r = dict(got[key]); ph = r.pop("phase")
        (P4 if ph == "P4" else P5 if ph == "P5" else bres["rows"]).append(r)
    order = {(j[0], j[1], j[2], j[4]): i for i, j in enumerate(jobs)}
    P4.sort(key=lambda r: order.get(("P4", r["depth"], r["name"], r["removal"]), 1e9))
    P5.sort(key=lambda r: order.get(("P5", r["depth"], r["name"], "mean"), 1e9))
    src["P4"], src["P5"] = P4, P5
    json.dump(src, open(MAIN, "w"), indent=1)
    json.dump(bres, open(bfile, "w"), indent=1)
    missing = [j[:3] for j in jobs if (j[0], j[1], j[2], j[4]) not in done and (j[0], j[1], j[2], j[4]) not in got]
    print(f"merged {len(got)} rows from {len(shards)} shards; missing {len(missing)}: {missing}")
    sys.exit()

todo = [j for j in jobs if (j[0], j[1], j[2], j[4]) not in done]
i, n = map(int, a.shard.split("/"))
mine = todo[i - 1::n]
OUT = f"results/exp25c_{a.tag}_shard{i}.json"
print(f"[{a.tag}] {len(jobs)} jobs, {len(done)} done, {len(todo)} to do; shard {a.shard}: {len(mine)}", flush=True)

from src.runner import Runner
from src.minimal import config_hooks, mixer_means, DictSuite, KVRetrieval, WikiPPL, lm_eval_tasks, KNOWLEDGE
t0 = time.time()
R = Runner(model_path=a.model, dtype=torch.float32)
means = mixer_means(R)
need_p4 = any(j[0] == "P4" for j in mine)
if need_p4:
    TEST, KV, PPL = DictSuite(R, seed=0), KVRetrieval(R, n=m["args"]["kv_n"], seed=0), WikiPPL(R)
rows = []
for ph, dname, name, keep, removal in mine:
    cut = DEPTHS[dname]
    h = config_hooks(R, cut=cut, keep=keep, removal=removal, means=means)
    if ph == "P4":
        r = dict(name=name, depth=dname, cut=cut, keep=keep, removal=removal,
                 n_linear=len(lin_upto(cut)) if keep is None else len(keep))
        r["dict"] = TEST.run(h)
        r["dict_mean"] = float(np.mean([v["acc"] for v in r["dict"].values()]))
        r["dict_cmean"] = float(np.mean([v["cacc"] for v in r["dict"].values()]))
        r["kv20"] = KV.run(h); r["ppl"] = PPL.run(h)
        msg = f"dict {r['dict_mean']:.3f}/c{r['dict_cmean']:.3f} kv20 {r['kv20']:.3f} ppl {r['ppl']:.1f}"
    else:
        r = dict(name=name, depth=dname, keep=keep, removal=removal)
        if ph == "B":
            r["k"] = len(keep)
        r["knowledge"] = lm_eval_tasks(R, h, KNOWLEDGE, limit=a.lm_limit)
        r["mmlu"] = lm_eval_tasks(R, h, ["mmlu"], limit=10, num_fewshot=5, batch_size=a.mmlu_bs).get("mmlu")
        r["recall"] = lm_eval_tasks(R, h, ["fda", "swde"], limit=200, batch_size=4)
        msg = f"knowledge {r['knowledge']} mmlu {r['mmlu']} recall {r['recall']}"
    r["phase"] = ph
    rows.append(r)
    json.dump(rows, open(OUT, "w"), indent=1)
    print(f"[{(time.time() - t0) / 60:6.1f}m] {ph} {dname:5s} {removal:4s} {name:28s} {msg}", flush=True)
print(f"wrote {OUT}", flush=True)
