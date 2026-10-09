"""Step 30: layerwise pruning curve (GnA Fig. 2 style): keep layers 0..cut, remove every later layer
(mixer and MLP, residual passes straight to the final norm), for every cut -- not only after each
softmax layer as in exp25 P1. Same metrics as exp25 P1 (so its 8 points are reproduced):
  dictionary: GnA-style choice accuracy among the 8 values (cacc), chat8/chat16/list8/rev8/long512
  WikiText-2 perplexity (16 x 512 tokens)
Sharded over cuts; `--merge` joins the shard files."""
import sys, json, argparse, time, glob
import numpy as np
import torch
sys.path.insert(0, "/mnt/yuang/LinearQuery")
from src.runner import Runner
from src.minimal import config_hooks, DictSuite, WikiPPL

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/mnt/yuang/models/Qwen3.5-9B")
ap.add_argument("--tag", default="9B")
ap.add_argument("--shard", default="1/1")
ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"],
                help="fp32 by default here, to reproduce the exp25 P1 points exactly")
ap.add_argument("--merge", action="store_true")
a = ap.parse_args()
OUT = f"results/exp30_prune_{a.tag}.json"

if a.merge:
    rows = [r for f in sorted(glob.glob(f"results/exp30_prune_{a.tag}_shard*.json"))
            for r in json.load(open(f))["rows"]]
    meta = json.load(open(glob.glob(f"results/exp30_prune_{a.tag}_shard*.json")[0]))["meta"]
    rows.sort(key=lambda r: r["kept"])
    json.dump(dict(meta=meta, rows=rows), open(OUT, "w"), indent=1)
    print(f"wrote {OUT}: {len(rows)} configurations"); sys.exit()

t0 = time.time()
R = Runner(model_path=a.model, dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16)
TEST = DictSuite(R, seed=0)
PPL = WikiPPL(R)
k, n = map(int, a.shard.split("/"))
cuts = list(range(R.n_layers - 1)) + [None]
mine = cuts[k - 1::n]
res = dict(meta=dict(model=a.model, tag=a.tag, n_layers=R.n_layers, attn=R.attn_layers,
                     linear=R.gdn_layers, dtype=a.dtype), rows=[])
for c in mine:
    h = config_hooks(R, cut=c)
    d = TEST.run(h)
    r = dict(cut=c, kept=R.n_layers if c is None else c + 1, dict=d,
             dict_cmean=float(np.mean([v["cacc"] for v in d.values()])),
             dict_mean=float(np.mean([v["acc"] for v in d.values()])), ppl=PPL.run(h))
    res["rows"].append(r)
    print(f"[{(time.time() - t0) / 60:5.1f}m] kept {r['kept']:2d}  cacc {r['dict_cmean']:.3f}  "
          f"ppl {r['ppl']:.1f}", flush=True)
    json.dump(res, open(f"results/exp30_prune_{a.tag}_shard{k}of{n}.json", "w"), indent=1)
