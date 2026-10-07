"""Step 25 (paper exp 2): a minimal hybrid designed from the Bind-Query circuit.

GnA (arXiv:2504.18574, Sec. 3.2-3.4) builds a training-free "minimal model": truncate the network
after the layer where retrieval is read out, then remove everything inside it that retrieval does
not need, and show that retrieval survives while knowledge-heavy scores fall. We do the same for
the *linear* layers of a hybrid, with the circuit (Parts VII-XVIII) telling us which to keep:

  Bind   the first linear block (token identity / key-value binding)       e.g. 0.8B: G0 [0,1,2]
  Query  the linear block right before the first strong reader (the query) e.g. 0.8B: G3 [12,13,14]

Phases (selection always on a *dev* dictionary sample, seed 1; everything reported on seed 0):
  P1  truncation curve: keep layers 0..c for c = every softmax layer (and full depth);
      cut* = the smallest c whose dev *candidate* accuracy (GnA choice scoring) is >= --trunc_tol
      of full depth. Everything else is selected on full-vocabulary accuracy.
  P2  single linear-layer removal (mean) inside cut* and at full depth: dev retrieval + PPL
  P3  greedy backward elimination (mean removal) on dev retrieval, stop below --greedy_tol of
      the reference: a data-driven minimal set, to compare with the circuit's
  P4  named configurations x {full depth, cut*} x {mean, zero}: circuit sets and matched-size
      baselines (PPL-ranked, first-k, last-k, random-k x5) on the test suite, KV-retrieval, PPL
  P5  lm-eval (knowledge 0-shot, MMLU 5-shot, FDA/SWDE recall) on the main configurations
"""
import sys, json, argparse, time, random, os
import numpy as np
import torch
sys.path.insert(0, "/mnt/yuang/LinearQuery")
from src.runner import Runner
from src.minimal import (config_hooks, mixer_means, DictSuite, KVRetrieval, WikiPPL,
                         lm_eval_tasks, KNOWLEDGE)

# circuit per model, from Parts XVI-XVIII (reader = first strong reader on chat8)
CIRCUIT = {
    "0.8B":  dict(bind=[0, 1, 2], query=[12, 13, 14], reader=15),
    "9B":    dict(bind=[0, 1, 2], query=[16, 17, 18], reader=19),
    "G1b":   dict(bind=[0, 1, 2, 3, 4], query=list(range(16, 25)), reader=25),
    "Gtiny": dict(bind=[0, 1, 2, 3, 4], query=list(range(16, 25)), reader=25),
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/mnt/yuang/gdn2-in-place/models/Qwen3.5-0.8B")
ap.add_argument("--tag", default="0.8B")
ap.add_argument("--trunc_tol", type=float, default=0.95)
ap.add_argument("--greedy_tol", type=float, default=0.90)
ap.add_argument("--n_random", type=int, default=5)
ap.add_argument("--kv_n", type=int, default=200)
ap.add_argument("--lm_limit", type=int, default=500)
ap.add_argument("--skip_lmeval", action="store_true")
ap.add_argument("--dtype", default="fp32")
ap.add_argument("--out", default="")
a = ap.parse_args()
OUT = a.out or f"results/exp25_minimal_{a.tag}.json"
C = CIRCUIT[a.tag]

t0 = time.time()
R = Runner(model_path=a.model, dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16)
LIN = R.gdn_layers
print(f"[{a.tag}] {R.n_layers} layers, softmax {R.attn_layers}, {len(LIN)} linear; circuit {C}",
      flush=True)


def log(*s):
    print(f"[{(time.time() - t0) / 60:6.1f}m]", *s, flush=True)


means = mixer_means(R)
DEV = DictSuite(R, variants=("chat8", "list8", "rev8"), seed=1)
TEST = DictSuite(R, seed=0)
KV = KVRetrieval(R, n=a.kv_n, seed=0)
PPL = WikiPPL(R)
res = dict(meta=dict(model=a.model, tag=a.tag, circuit=C, attn=R.attn_layers, linear=LIN,
                     n_layers=R.n_layers, args=vars(a)))
json_dump = lambda: json.dump(res, open(OUT, "w"), indent=1)


def dev_score(hooks, key="acc"):
    """mean dev accuracy over chat8/list8/rev8 (key: acc = full vocabulary, cacc = among values)"""
    d = DEV.run(hooks)
    return float(np.mean([v[key] for v in d.values()])), d


def full_eval(hooks, name, extra=None):
    r = dict(name=name, **(extra or {}))
    r["dict"] = TEST.run(hooks)
    r["dict_mean"] = float(np.mean([v["acc"] for v in r["dict"].values()]))
    r["dict_cmean"] = float(np.mean([v["cacc"] for v in r["dict"].values()]))
    r["kv20"] = KV.run(hooks)
    r["ppl"] = PPL.run(hooks)
    return r


def fmt(r):
    d = r["dict"]
    return (f"dict {r['dict_mean']:.3f}/c{r['dict_cmean']:.3f} ["
            + " ".join(f"{k}:{v['acc']:.2f}/{v['cacc']:.2f}" for k, v in d.items())
            + f"]  kv20 {r['kv20']:.3f}  ppl {r['ppl']:.1f}")


# ------------------------------------------------------------------ P1 truncation curve
res["P1"] = []
full_dev, _ = dev_score([])
full_cdev, _ = dev_score([], "cacc")
cuts = [c for c in R.attn_layers if c < R.n_layers - 1] + [None]
cut_star = None
for c in cuts:
    h = config_hooks(R, cut=c)
    s, _ = dev_score(h, "cacc")
    r = full_eval(h, f"cut{c if c is not None else 'full'}", dict(cut=c, cdev=s))
    res["P1"].append(r)
    log(f"P1 cut {str(c):>4s}  cdev {s:.3f}  " + fmt(r))
    if cut_star is None and c is not None and s >= a.trunc_tol * full_cdev:
        cut_star = c
if cut_star is None:
    cut_star = R.n_layers - 1
res["meta"]["cut_star"] = cut_star
res["meta"]["full_dev"] = full_dev
log(f"cut* = {cut_star} (full-depth dev {full_dev:.3f})")
json_dump()

DEPTHS = {"full": None, "trunc": cut_star}
# a truncated Qwen model has the answer on top among the values but not over the vocabulary
# (Part VIII logit lens), so selection inside the truncated model uses candidate accuracy
KEY = {"full": "acc", "trunc": "cacc"}


def lin_upto(cut):
    return [L for L in LIN if cut is None or L <= cut]


# ------------------------------------------------------------------ P2 single-layer removal
res["P2"] = {}
ppl_rank = {}
for dname, cut in DEPTHS.items():
    base_h = config_hooks(R, cut=cut)
    base_dev, _ = dev_score(base_h, KEY[dname])
    base_ppl = PPL.run(base_h)
    rows = []
    for L in lin_upto(cut):
        keep = [x for x in lin_upto(cut) if x != L]
        h = config_hooks(R, cut=cut, keep=keep, removal="mean", means=means)
        s, d = dev_score(h, KEY[dname])
        p = PPL.run(h)
        rows.append(dict(layer=L, dev=s, dev_by=d, ppl=p))
        log(f"P2 {dname} -L{L:<3d} dev {s:.3f} (base {base_dev:.3f})  ppl {p:.1f} (base {base_ppl:.1f})")
    res["P2"][dname] = dict(key=KEY[dname], base_dev=base_dev, base_ppl=base_ppl, rows=rows)
    ppl_rank[dname] = [r["layer"] for r in sorted(rows, key=lambda r: -r["ppl"])]
json_dump()


# ------------------------------------------------------------------ P3 greedy backward elimination
res["P3"] = {}
greedy = {}
for dname, cut in DEPTHS.items():
    keep = list(lin_upto(cut))
    ref = res["P2"][dname]["base_dev"]
    path = []
    while keep:
        best = None
        for L in keep:
            k2 = [x for x in keep if x != L]
            s, _ = dev_score(config_hooks(R, cut=cut, keep=k2, removal="mean", means=means), KEY[dname])
            if best is None or s > best[1]:
                best = (L, s)
        if best[1] < a.greedy_tol * ref:
            break
        keep.remove(best[0])
        path.append(dict(removed=best[0], dev=best[1], kept=list(keep)))
        log(f"P3 {dname} greedy: drop L{best[0]} -> dev {best[1]:.3f}  kept {keep}")
    greedy[dname] = keep
    res["P3"][dname] = dict(ref=ref, path=path, minimal=keep)
    log(f"P3 {dname} minimal set ({len(keep)}): {keep}")
json_dump()


# ------------------------------------------------------------------ P4 named configurations
def named_sets(dname, cut):
    lin = lin_upto(cut)
    bq = sorted(set(C["bind"]) | set(C["query"]))
    k = len(bq)
    pre = [L for L in lin if L < C["reader"]]
    S = {"all": None, "none": [], "bind": C["bind"], "query": C["query"], "bind+query": bq,
         "pre-reader": pre, "greedy": greedy[dname],
         f"ppl-top{k}": sorted(ppl_rank[dname][:k]),
         f"first{k}": lin[:k], f"last{k}": lin[-k:]}
    rng = random.Random(0)
    for s in range(a.n_random):
        S[f"random{k}_s{s}"] = sorted(rng.sample(lin, k))
    kg = len(greedy[dname])
    S[f"ppl-top{kg}(greedy-size)"] = sorted(ppl_rank[dname][:kg])
    for s in range(a.n_random):
        S[f"random{kg}(greedy-size)_s{s}"] = sorted(rng.sample(lin, kg))
    return S


res["P4"] = []
for dname, cut in DEPTHS.items():
    for name, keep in named_sets(dname, cut).items():
        for removal in (("mean", "zero") if keep is not None else ("-",)):
            h = config_hooks(R, cut=cut, keep=keep, removal=removal, means=means)
            r = full_eval(h, name, dict(depth=dname, cut=cut, keep=keep, removal=removal,
                                        n_linear=len(lin_upto(cut)) if keep is None else len(keep)))
            res["P4"].append(r)
            log(f"P4 {dname:5s} {removal:4s} {name:28s} k={r['n_linear']:2d}  " + fmt(r))
        json_dump()

# ------------------------------------------------------------------ P5 lm-eval on main configurations
if not a.skip_lmeval:
    bq = sorted(set(C["bind"]) | set(C["query"]))
    k = len(bq)
    main = [("full", "all", None), ("trunc", "all", None),
            ("full", "bind+query", bq), ("trunc", "bind+query", bq),
            ("trunc", "greedy", greedy["trunc"]), ("full", "greedy", greedy["full"]),
            ("trunc", f"ppl-top{k}", sorted(ppl_rank["trunc"][:k])),
            ("full", f"ppl-top{k}", sorted(ppl_rank["full"][:k])),
            ("trunc", f"random{k}_s0", sorted(random.Random(0).sample(lin_upto(cut_star), k))),
            ("full", "none", [])]
    res["P5"] = []
    for dname, name, keep in main:
        h = config_hooks(R, cut=DEPTHS[dname], keep=keep, removal="mean", means=means)
        r = dict(depth=dname, name=name, keep=keep)
        r["knowledge"] = lm_eval_tasks(R, h, KNOWLEDGE, limit=a.lm_limit)
        r["mmlu"] = lm_eval_tasks(R, h, ["mmlu"], limit=10, num_fewshot=5, batch_size=4).get("mmlu")  # bs 16 OOMs at 9B fp32
        r["recall"] = lm_eval_tasks(R, h, ["fda", "swde"], limit=200, batch_size=4)
        res["P5"].append(r)
        log(f"P5 {dname:5s} {name:14s} knowledge {r['knowledge']}  mmlu {r['mmlu']}  recall {r['recall']}")
        json_dump()

res["meta"]["seconds"] = time.time() - t0
json_dump()
log(f"wrote {OUT}")
