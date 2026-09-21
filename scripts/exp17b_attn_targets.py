"""Step 17b: when G1/G2/G3 are removed, where does the reader's attention go instead?

exp17 showed the key identity is linearly decodable at every value token from layer 3 on, with
or without G2 -- so G2 does not create the binding. Here: split L15H5's attention (final token,
clean prompt) into target value, the neighbouring entries' values, other values, key words,
the query key in the question, <|im_start|> (BOS sink), and everything else.
"""
import sys, json, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from src.runner import Runner, hook_ctx
from src.qkv import AttnWeights
from src.task import make_examples
from src.harness import group_examples

R = Runner(dtype=torch.float32)
R.model.config.get_text_config()._attn_implementation = "eager"
exs = []
for i, (ti, di) in enumerate([(2, 5), (5, 2), (1, 6), (6, 1)]):
    exs += make_examples(25, 8, seed=11 + i, target_pos=ti, distract_pos=di)
groups = group_examples(R, exs)
CONDS = {"intact": [], "remove G1": [4, 5, 6], "remove G2": [8, 9, 10], "remove L10": [10],
         "remove G3": [12, 13, 14], "remove L14": [14], "remove G1+G2": [4, 5, 6, 8, 9, 10]}


class Zero:
    def __init__(self, L): self.L = L
    def register(self):
        return R.mixer(self.L).register_forward_hook(lambda m, a_, o: torch.zeros_like(o))


CATS = ["target value", "prev entry value", "next entry value", "other values", "first entry value",
        "key words", "query key (question)", "<|im_start|>", "rest"]
rows = []
print(f"{'condition':14s} " + " ".join(f"{c[:12]:>12s}" for c in CATS), flush=True)
for cn, zl in CONDS.items():
    tot = np.zeros(len(CATS)); n = 0
    for g in groups:
        aw = AttnWeights(R.mixer(15))
        with torch.no_grad(), hook_ctx([aw] + [Zero(L) for L in zl]):
            R.model(g.clean_ids, use_cache=False)
        a = aw.value[:, 5, g.pos["final"]].float().cpu().numpy()      # [B, T]
        ti = g.exs[0].target_idx
        vp = g.pos["value_pos"]
        cat = {"target value": [vp[ti]],
               "prev entry value": [vp[ti - 1]] if ti > 0 else [],
               "next entry value": [vp[ti + 1]] if ti < 7 else [],
               "first entry value": [vp[0]] if ti not in (0, 1) else []}
        used = {p for v in cat.values() for p in v}
        cat["other values"] = [p for p in vp.values() if p not in used]
        cat["key words"] = list(g.pos["key_pos"].values())
        cat["query key (question)"] = [g.pos["query_key_question"]]
        cat["<|im_start|>"] = [0]
        used = {p for v in cat.values() for p in v}
        cat["rest"] = [p for p in range(a.shape[1]) if p not in used]
        tot += np.array([a[:, cat[c]].sum(1).mean() if cat[c] else 0 for c in CATS]); n += 1
    tot /= n
    rows.append(dict(cond=cn, **{c: float(v) for c, v in zip(CATS, tot)}))
    print(f"{cn:14s} " + " ".join(f"{v:12.3f}" for v in tot), flush=True)
json.dump(rows, open("results/exp17b_attn_targets.json", "w"), indent=2)
