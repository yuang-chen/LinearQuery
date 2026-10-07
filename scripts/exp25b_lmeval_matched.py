"""Step 25b: lm-eval for exp25 configurations at *matched* size (full depth, mean removal).

exp25 P5 compares the greedy set with the k = |bind+query| PPL-ranked set; here every circuit set
is paired with a PPL-ranked set and random sets of the same size, reading the sets and the PPL
ranking from the exp25 result file."""
import sys, json, argparse, random, time
import torch
sys.path.insert(0, "/mnt/yuang/LinearQuery")
from src.runner import Runner
from src.minimal import config_hooks, mixer_means, lm_eval_tasks, KNOWLEDGE

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/mnt/yuang/gdn2-in-place/models/Qwen3.5-0.8B")
ap.add_argument("--tag", default="0.8B")
ap.add_argument("--n_random", type=int, default=3)
ap.add_argument("--lm_limit", type=int, default=500)
a = ap.parse_args()
src = json.load(open(f"results/exp25_minimal_{a.tag}.json"))
OUT = f"results/exp25b_lmeval_{a.tag}.json"
t0 = time.time()
R = Runner(model_path=a.model, dtype=torch.float32)
means = mixer_means(R)
C = src["meta"]["circuit"]
lin = src["meta"]["linear"]
rank = [r["layer"] for r in sorted(src["P2"]["full"]["rows"], key=lambda r: -r["ppl"])]
sets = {"greedy": src["P3"]["full"]["minimal"],
        "pre-reader": [L for L in lin if L < C["reader"]]}
configs = []
for name, keep in sets.items():
    k = len(keep)
    configs.append((name, sorted(keep)))
    configs.append((f"ppl-top{k}", sorted(rank[:k])))
    rng = random.Random(k)
    for s in range(a.n_random):
        configs.append((f"random{k}_s{s}", sorted(rng.sample(lin, k))))
res = dict(meta=dict(tag=a.tag, model=a.model, sets=sets, ppl_rank=rank), rows=[])
for name, keep in configs:
    h = config_hooks(R, cut=None, keep=keep, removal="mean", means=means)
    r = dict(name=name, keep=keep, k=len(keep))
    r["knowledge"] = lm_eval_tasks(R, h, KNOWLEDGE, limit=a.lm_limit)
    r["mmlu"] = lm_eval_tasks(R, h, ["mmlu"], limit=10, num_fewshot=5, batch_size=4).get("mmlu")  # bs 16 OOMs at 9B fp32
    r["recall"] = lm_eval_tasks(R, h, ["fda", "swde"], limit=200, batch_size=4)
    res["rows"].append(r)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"[{(time.time() - t0) / 60:6.1f}m] {name:16s} k={len(keep):2d} {keep}  knowledge {r['knowledge']}  "
          f"mmlu {r['mmlu']:.3f}  recall {r['recall']}", flush=True)
print(f"wrote {OUT}", flush=True)
