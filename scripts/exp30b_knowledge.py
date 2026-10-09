"""Step 30b: knowledge tasks along the layerwise truncation curve of exp30 (GnA Fig. 2): keep layers 0..cut,
remove every later layer (mixer and MLP). GnA's knowledge set (arXiv:2504.18574 Sec. 3.1): ARC-Challenge,
ARC-Easy, PIQA, Winogrande, OpenBookQA, HellaSwag; lm-evaluation-harness, zero-shot, log-likelihood
multiple choice; acc and acc_norm both stored. Sharded over cuts; `--merge` joins the shard files."""
import sys, json, argparse, time, glob
import torch
sys.path.insert(0, "/mnt/yuang/LinearQuery")
from src.runner import Runner, hook_ctx
from src.minimal import config_hooks

TASKS = ["arc_challenge", "arc_easy", "piqa", "winogrande", "openbookqa", "hellaswag"]
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/mnt/yuang/models/Qwen3.5-9B")
ap.add_argument("--tag", default="9B")
ap.add_argument("--shard", default="1/1")
ap.add_argument("--limit", type=int, default=500, help="items per task (lm-eval --limit)")
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
ap.add_argument("--merge", action="store_true")
a = ap.parse_args()
OUT = f"results/exp30b_knowledge_{a.tag}.json"

if a.merge:
    fs = sorted(glob.glob(f"results/exp30b_knowledge_{a.tag}_shard*.json"))
    rows = sorted((r for f in fs for r in json.load(open(f))["rows"]), key=lambda r: r["kept"])
    json.dump(dict(meta=json.load(open(fs[0]))["meta"], rows=rows), open(OUT, "w"), indent=1)
    print(f"wrote {OUT}: {len(rows)} configurations"); sys.exit()

import lm_eval
from lm_eval.models.huggingface import HFLM
t0 = time.time()
R = Runner(model_path=a.model, dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16)
lm = HFLM(pretrained=R.model, tokenizer=R.tok, batch_size=a.batch_size)
k, n = map(int, a.shard.split("/"))
cuts = (list(range(R.n_layers - 1)) + [None])[k - 1::n]
res = dict(meta=dict(model=a.model, tag=a.tag, n_layers=R.n_layers, attn=R.attn_layers, tasks=TASKS,
                     limit=a.limit, num_fewshot=0, dtype=a.dtype), rows=[])
for c in cuts:
    with hook_ctx(config_hooks(R, cut=c)):
        out = lm_eval.simple_evaluate(model=lm, tasks=TASKS, num_fewshot=0, limit=a.limit,
                                      log_samples=False, verbosity="ERROR")["results"]
    r = dict(cut=c, kept=R.n_layers if c is None else c + 1,
             acc={t: float(out[t]["acc,none"]) for t in TASKS},
             acc_norm={t: float(out[t]["acc_norm,none"]) for t in TASKS if "acc_norm,none" in out[t]})
    res["rows"].append(r)
    print(f"[{(time.time() - t0) / 60:5.1f}m] kept {r['kept']:2d}  "
          + " ".join(f"{t}:{r['acc_norm'].get(t, r['acc'][t]):.3f}" for t in TASKS), flush=True)
    json.dump(res, open(f"results/exp30b_knowledge_{a.tag}_shard{k}of{n}.json", "w"), indent=1)
