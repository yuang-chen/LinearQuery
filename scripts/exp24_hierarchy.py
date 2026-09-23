"""Step 24: the top of the ablation hierarchy -- whole components, before groups and layers.

Parts XVI-XVIII ablate groups, then layers, then heads. This adds the coarsest level, so the
paper can present a strict top-down narrative:

  all linear layers            -- do the linear mixers matter at all?
  all softmax layers           -- contrast: do the attention layers matter at all?
  all linear except the first block   -- is the early block sufficient on its own?
  all linear except the query block   -- is the query block sufficient on its own?
  first block only removed / query block only removed   (repeat of Part XVI, as a check)

Same task, metrics and zero-ablation as Part XVI (chat8, 100 items, acc and gap).
"""
import sys, json, argparse, time, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from src.runner import Runner, hook_ctx
from src.gen import make_items, batches

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/user/yac/LinearSwap/models/Qwen3.5-0.8B")
ap.add_argument("--tag", default="0.8B")
ap.add_argument("--variant", default="chat8")
ap.add_argument("--group_size", type=int, default=3)
ap.add_argument("--query_group", type=int, default=-1, help="index of the query block (default: "
                "the group before the reader found in the exp21 run)")
a = ap.parse_args()
OUT = f"results/exp24_hierarchy_{a.tag}.json"

t0 = time.time()
R = Runner(model_path=a.model, dtype=torch.float32)
tok = R.tok
GROUPS = R.groups(a.group_size)
QG = a.query_group
if QG < 0:
    d = json.load(open(f"results/exp21/{a.tag}_{a.variant}.json"))
    RL = d["meta"]["reader"][0]
    QG = max(i for i, s in enumerate(R.attn_layers) if s <= RL)
print(f"[{a.tag}] linear {len(R.gdn_layers)} layers, softmax {R.attn_layers}, "
      f"groups {GROUPS}, query block G{QG} {GROUPS[QG]}", flush=True)


class Zero:
    """Zero a mixer's output. Softmax layers return (hidden, attn_weights); linear ones a tensor."""
    def __init__(self, L): self.L = L
    def register(self):
        def fn(m, a_, o):
            if isinstance(o, tuple):
                return (torch.zeros_like(o[0]),) + tuple(o[1:])
            return torch.zeros_like(o)
        return R.mixer(self.L).register_forward_hook(fn)


items = make_items(a.variant, tok, seed=0)
B = batches(tok, items, R.device, 25)
N = sum(len(b.items) for b in B)


@torch.no_grad()
def evaluate(layers):
    ok = tot = 0; gaps = []
    for b in B:
        ds = []
        for ids, want in ((b.clean_ids, b.aid), (b.corr_ids, b.cid)):
            with hook_ctx([Zero(L) for L in layers]):
                lg = R.model(ids, use_cache=False).logits[:, -1].float()
            ok += int((lg.argmax(-1) == want).sum()); tot += len(want)
            ds.append(b.D(lg))
        gaps += (ds[0] - ds[1]).tolist()
    return dict(acc=ok / tot, gap=float(np.mean(gaps)), n_layers=len(layers))


LIN = list(R.gdn_layers)
first, query = GROUPS[0], GROUPS[QG]
CONDS = [
    ("intact", []),
    ("all linear removed", LIN),
    ("all softmax removed", list(R.attn_layers)),
    ("all linear but first block", [L for L in LIN if L not in first]),
    ("all linear but query block", [L for L in LIN if L not in query]),
    ("first block removed", first),
    ("query block removed", query),
]
res = dict(meta=dict(model=a.model, tag=a.tag, variant=a.variant, n=N, groups=GROUPS,
                     query_group=QG, linear=LIN, softmax=list(R.attn_layers)), conditions=[])
for name, layers in CONDS:
    r = evaluate(layers); r.update(name=name, layers=layers)
    res["conditions"].append(r)
    print(f"  {name:28s} acc {r['acc']:.3f}  gap {r['gap']:+6.2f}  ({r['n_layers']} layers)",
          flush=True)

res["meta"]["seconds"] = time.time() - t0
json.dump(res, open(OUT, "w"), indent=2)
print(f"wrote {OUT}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
