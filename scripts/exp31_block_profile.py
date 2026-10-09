"""Step 31: per-block removal profile for the paper's Section 4 figure. For every linear-attention block
(mixer outputs zeroed at all positions, MLPs untouched): accuracy, paired logit gap, and the reader's
attention from the final token on the queried entry (clean prompts). Same items and metrics as exp21."""
import sys, json, argparse, time, torch
import numpy as np
sys.path.insert(0, "/mnt/yuang/LinearQuery")
from src.runner import Runner, hook_ctx
from src.qkv import AttnWeights
from src.gen import make_items, batches

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/mnt/yuang/models/Qwen3.5-9B")
ap.add_argument("--tag", default="9B")
ap.add_argument("--variant", default="chat8")
ap.add_argument("--reader", default="19,11")
ap.add_argument("--group_size", type=int, default=3)
ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
a = ap.parse_args()
t0 = time.time()
R = Runner(model_path=a.model, dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16)
R.model.config.get_text_config()._attn_implementation = "eager"
RL, RH = map(int, a.reader.split(","))
GROUPS = R.groups(a.group_size)
B = batches(R.tok, make_items(a.variant, R.tok, seed=0), R.device, 25)


class Zero:
    def __init__(self, L): self.L = L
    def register(self):
        return R.mixer(self.L).register_forward_hook(lambda m, a_, o: torch.zeros_like(o))


@torch.no_grad()
def evaluate(hooks):
    ok = tot = 0; gaps = []; att = []
    for b in B:
        aw = AttnWeights(R.mixer(RL))
        with hook_ctx(hooks + [aw]):
            lc = R.model(b.clean_ids, use_cache=False).logits[:, -1]
        at = aw.value[:, RH, b.pos["final"]].float()
        att += at[:, b.pos["entry_pos"][b.ti]].sum(-1).tolist()
        with hook_ctx(hooks):
            lx = R.model(b.corr_ids, use_cache=False).logits[:, -1]
        ok += (lc.argmax(-1) == b.aid).sum().item() + (lx.argmax(-1) == b.cid).sum().item()
        tot += 2 * len(b.aid)
        gaps += (b.D(lc) - b.D(lx)).tolist()
    return dict(acc=ok / tot, gap=float(np.mean(gaps)), attn=float(np.mean(att)))


res = dict(meta=dict(model=a.model, tag=a.tag, variant=a.variant, reader=[RL, RH], groups=GROUPS,
                     dtype=a.dtype), intact=evaluate([]), blocks=[])
print("intact", res["intact"], flush=True)
for gi, G in enumerate(GROUPS):
    r = evaluate([Zero(L) for L in G]); r.update(block=gi, layers=G)
    res["blocks"].append(r)
    print(f"G{gi} {G}: acc {r['acc']:.2f} gap {r['gap']:+.1f} attn {r['attn']:.2f}", flush=True)
res["meta"]["seconds"] = time.time() - t0
json.dump(res, open(f"results/exp31_profile_{a.tag}_{a.variant}.json", "w"), indent=1)
