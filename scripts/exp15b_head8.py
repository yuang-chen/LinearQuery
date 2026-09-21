"""Step 15b: is GDN-0 head 8 the writer from Part I, and what do the other heads do?

Part I: patching GDN layer 0's whole mixer output at the source value token, clean -> corrupted,
recovers +1.04 of the answer logit gap (sufficient AND necessary).
Step 15: zeroing head 8 of layer 0 alone takes retrieval from 1.000 to 0.000.

This resolves the Part I writer patch to individual heads, and asks whether head 8 is needed
everywhere or only at the source value token -- i.e. whether it is a general token encoder that
retrieval happens to depend on, or something retrieval-specific.
"""
import sys, json, argparse, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from src.runner import Runner, Capture, hook_ctx
from src.qkv import AttnWeights
from src.task import make_examples
from src.harness import group_examples, recovery

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=0)
ap.add_argument("--out", default="results/exp15b_head8.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
R.model.config.get_text_config()._attn_implementation = "eager"
L0 = a.layer
NH, HD = R.cfg.linear_num_value_heads, R.cfg.linear_value_head_dim
exs = []
for i, (ti, di) in enumerate([(2, 5), (5, 2), (1, 6), (6, 1)]):
    exs += make_examples(25, 8, seed=11 + i, target_pos=ti, distract_pos=di)
groups = group_examples(R, exs)
rows = []


class HeadPatchIn:
    """Patch heads' slices of the out_proj input at given positions with a donor (or zeros)."""

    def __init__(self, L, heads, positions, donor=None):
        self.L, self.heads, self.positions, self.donor = L, list(heads), positions, donor

    def register(self):
        def fn(mod, args, kwargs):
            t = (args[0] if args else kwargs["input"]).clone()
            for h in self.heads:
                sl = slice(h * HD, (h + 1) * HD)
                if self.positions is None:
                    t[..., sl] = 0 if self.donor is None else self.donor[..., sl]
                else:
                    for p in self.positions:
                        t[:, p, sl] = 0 if self.donor is None else self.donor[:, p, sl]
            return (t,) + tuple(args[1:]), kwargs
        return R.mixer(self.L).out_proj.register_forward_pre_hook(fn, with_kwargs=True)


class CapIn:
    def __init__(self, L): self.L, self.value = L, None
    def register(self):
        def fn(mod, args, kwargs):
            self.value = (args[0] if args else kwargs["input"]).detach()
        return R.mixer(self.L).out_proj.register_forward_pre_hook(fn, with_kwargs=True)


print("== A. head-resolved writer patch (clean donor -> corrupted run, @ source value) ==",
      flush=True)
per_head, whole = {h: [] for h in range(NH)}, []
attn_base, attn_h8 = [], []
for g in groups:
    tv = g.pos["target_value"]
    with torch.no_grad():
        d_clean = g.D(R.forward(g.clean_ids).logits[:, -1])
        d_corr = g.D(R.forward(g.corr_ids).logits[:, -1])
    cap = CapIn(L0)
    with torch.no_grad(), hook_ctx([cap]):
        R.model(g.clean_ids, use_cache=False)
    donor = cap.value
    for h in range(NH):
        with torch.no_grad(), hook_ctx([HeadPatchIn(L0, [h], [tv], donor)]):
            dp = g.D(R.model(g.corr_ids, use_cache=False).logits[:, -1])
        per_head[h] += recovery(dp, d_corr, d_clean).tolist()
    with torch.no_grad(), hook_ctx([HeadPatchIn(L0, range(NH), [tv], donor)]):
        dp = g.D(R.model(g.corr_ids, use_cache=False).logits[:, -1])
    whole += recovery(dp, d_corr, d_clean).tolist()

print(f"  all 16 heads patched: recovery {np.mean(whole):+.3f}")
order = sorted(range(NH), key=lambda h: -abs(np.mean(per_head[h])))
for h in order:
    m, s = np.mean(per_head[h]), np.std(per_head[h]) / len(per_head[h]) ** .5
    rows.append(dict(kind="writer_head", head=h, rec=float(m), sem=float(s)))
    if abs(m) > 0.02:
        print(f"  head {h:2d} alone: recovery {m:+.3f} ± {s:.3f}")

print("\n== B. where is head 8 needed?  (zero it at selected positions only) ==", flush=True)


def evaluate(hooks):
    ok = tot = 0
    gaps, atts = [], []
    for g in groups:
        ds_ = []
        for idsb, want in ((g.clean_ids, g.aid), (g.corr_ids, g.cid)):
            hk = [h(g) for h in hooks]
            with torch.no_grad(), hook_ctx(hk):
                lg = R.model(idsb, use_cache=False).logits[:, -1]
            ok += (lg.argmax(-1) == want).sum().item(); tot += idsb.shape[0]
            ds_.append(g.D(lg))
        gaps += (ds_[0] - ds_[1]).tolist()
        aw = AttnWeights(R.mixer(15))
        with torch.no_grad(), hook_ctx([aw] + [h(g) for h in hooks]):
            R.model(g.corr_ids, use_cache=False)
        atts += aw.value[:, 5, g.pos["final"], g.pos["target_value"]].float().cpu().tolist()
    return ok / tot, float(np.mean(gaps)), float(np.mean(atts))


SITES = {
    "intact": [],
    "head 8 zeroed everywhere": [lambda g: HeadPatchIn(L0, [8], None)],
    "head 8 zeroed @ source value only": [lambda g: HeadPatchIn(L0, [8], [g.pos["target_value"]])],
    "head 8 zeroed @ all value tokens": [
        lambda g: HeadPatchIn(L0, [8], list(g.pos["value_pos"].values()))],
    "head 8 zeroed @ query key only": [lambda g: HeadPatchIn(L0, [8], [g.pos["query_key_question"]])],
    "head 8 zeroed @ final token only": [lambda g: HeadPatchIn(L0, [8], [g.pos["final"]])],
    "head 8 zeroed except at value tokens": [
        lambda g: HeadPatchIn(L0, [8], [p for p in range(g.pos["n_tokens"])
                                        if p not in set(g.pos["value_pos"].values())])],
    "heads 3+4 zeroed everywhere (PPL-heavy control)": [lambda g: HeadPatchIn(L0, [3, 4], None)],
}
for nm, hooks in SITES.items():
    acc, gap, att = evaluate(hooks)
    rows.append(dict(kind="site", name=nm, acc=acc, gap=gap, attn=att))
    print(f"  {nm:48s} acc {acc:.3f}  gap {gap:+6.2f}  L15H5 attn {att:.3f}", flush=True)

json.dump(dict(meta=vars(a), whole=float(np.mean(whole)), rows=rows), open(a.out, "w"), indent=2)
print("\nwrote", a.out)
