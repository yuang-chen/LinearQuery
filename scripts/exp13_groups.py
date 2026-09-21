"""Step 13: GDN layer groups, and what survives if only layer 0 is kept.

The 18 GDN layers fall into exactly 6 blocks of 3 consecutive layers, each block sitting
between two softmax layers:
    G0 [0,1,2] | attn 3 | G1 [4,5,6] | attn 7 | G2 [8,9,10] | attn 11
    G3 [12,13,14] | attn 15 | G4 [16,17,18] | attn 19 | G5 [20,21,22] | attn 23

Measured on WikiText-2 (perplexity) and on the Part I dictionary task (retrieval accuracy and
the clean-vs-corrupt answer logit gap), because the two can dissociate sharply.

  A  ablate each group (zero its 3 GDN mixers), vs matched random triples
  B  keep only one group (zero the other 15 mixers)
  C  keep only one layer -- including "keep only GDN layer 0"
  D  keep the first k GDN layers
  E  cumulative removal from the front and from the back
  F  references: all GDN mixers zeroed; whole-layer skip variants

Softmax layers and all MLPs are left intact throughout unless stated; ablation zeroes the GDN
token mixer's output, which isolates the mixer from the layer-local feed-forward.
"""
import sys, json, argparse, random, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from datasets import load_dataset
from src.runner import Runner, hook_ctx
from src.task import make_examples
from src.harness import group_examples

ap = argparse.ArgumentParser()
ap.add_argument("--n_eval", type=int, default=16)
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--n_random", type=int, default=6)
ap.add_argument("--out", default="results/exp13_groups.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
GDN = R.gdn_layers
GROUPS = [GDN[i:i + 3] for i in range(0, len(GDN), 3)]
assert all(len(g) == 3 for g in GROUPS) and len(GROUPS) == 6, GROUPS
print("GDN groups:", GROUPS, flush=True)

d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
ids = R.tok("\n\n".join(t for t in d["text"] if t.strip()), return_tensors="pt").input_ids[0]
seqs = [ids[i * a.seq_len:(i + 1) * a.seq_len].unsqueeze(0).to(R.device) for i in range(a.n_eval)]

exs = []
for i, (ti, di) in enumerate([(2, 5), (5, 2), (1, 6), (6, 1)]):
    exs += make_examples(25, 8, seed=11 + i, target_pos=ti, distract_pos=di)
groups_ex = group_examples(R, exs)


class ZeroMixer:
    def __init__(self, L): self.L = L
    def register(self):
        def fn(mod, args, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + tuple(output[1:])
            return torch.zeros_like(output)
        return R.mixer(self.L).register_forward_hook(fn)


class SkipLayer:
    def __init__(self, L): self.L = L
    def register(self):
        def fn(mod, args, kwargs, output):
            return kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        return R.layers[self.L].register_forward_hook(fn, with_kwargs=True)


@torch.no_grad()
def ppl_per_seq(hooks=()):
    v = []
    for s in seqs:
        with hook_ctx(hooks):
            lg = R.model(s, use_cache=False).logits
        v.append(torch.nn.functional.cross_entropy(lg[0, :-1].float(), s[0, 1:]).item())
    return np.array(v)


@torch.no_grad()
def retrieval(hooks=()):
    ok = tot = 0
    gaps = []
    for g in groups_ex:
        ds_ = []
        for idsb, want in ((g.clean_ids, g.aid), (g.corr_ids, g.cid)):
            with hook_ctx(hooks):
                lg = R.model(idsb, use_cache=False).logits[:, -1]
            ok += (lg.argmax(-1) == want).sum().item(); tot += idsb.shape[0]
            ds_.append(g.D(lg))
        gaps += (ds_[0] - ds_[1]).tolist()
    return ok / tot, float(np.mean(gaps))


base_v = ppl_per_seq()
base = base_v.mean()
base_acc, base_gap = retrieval()
print(f"\nintact: PPL {np.exp(base):.2f}   retrieval acc {base_acc:.3f}  gap {base_gap:+.2f}\n",
      flush=True)
rows = []


def run(kind, name, zero_layers, do_retr=True, skip=False, **kw):
    hooks = [(SkipLayer if skip else ZeroMixer)(L) for L in zero_layers]
    v = ppl_per_seq(hooks)
    acc, gap = retrieval(hooks) if do_retr else (None, None)
    r = dict(kind=kind, name=name, layers=list(zero_layers), n=len(zero_layers),
             nll=float(v.mean()), ppl=float(np.exp(v.mean())), dnll=float(v.mean() - base),
             sem=float((v - base_v).std(ddof=1) / len(v) ** .5), acc=acc, gap=gap, **kw)
    rows.append(r)
    ar = f"  acc {acc:.3f}  gap {gap:+6.2f}" if acc is not None else ""
    print(f"  {name:46s} PPL {r['ppl']:12.2f}  dNLL {r['dnll']:+7.3f}{ar}", flush=True)
    return r


ALL = list(GDN)
print("== A. ablate each group of 3 consecutive GDN layers ==", flush=True)
for gi, g in enumerate(GROUPS):
    run("group_ablate", f"G{gi} {g} removed", g, group=gi)
print("  -- matched random triples --", flush=True)
rng = random.Random(0)
dr = []
for t in range(a.n_random):
    S = sorted(rng.sample(ALL, 3))
    dr.append(run("random3", f"random triple {S}", S, do_retr=False)["dnll"])
rows.append(dict(kind="random3_summary", mean=float(np.mean(dr)), std=float(np.std(dr)),
                 n_trials=a.n_random))
print(f"  random triple mean dNLL {np.mean(dr):+.3f} (sd {np.std(dr):.3f})", flush=True)

print("\n== B. keep only one group (zero the other 15 mixers) ==", flush=True)
for gi, g in enumerate(GROUPS):
    run("group_keep", f"keep only G{gi} {g}", [L for L in ALL if L not in g], group=gi)

print("\n== C. keep only one GDN layer ==", flush=True)
for L in [0, 1, 2, 4, 8, 12, 16, 20, 22]:
    run("keep_one", f"keep only GDN layer {L}", [x for x in ALL if x != L], keep=L)

print("\n== D. keep the first k GDN layers ==", flush=True)
for k in [1, 2, 3, 4, 6, 9, 12, 15, 18]:
    keep = ALL[:k]
    run("keep_first_k", f"keep first {k} GDN layers {keep}", [x for x in ALL if x not in keep], k=k)

print("\n== E. cumulative removal ==", flush=True)
for k in range(1, 7):
    rm = [L for g in GROUPS[:k] for L in g]
    run("cum_front", f"remove groups G0..G{k-1} ({len(rm)} layers)", rm, k=k)
for k in range(1, 7):
    rm = [L for g in GROUPS[-k:] for L in g]
    run("cum_back", f"remove groups G{6-k}..G5 ({len(rm)} layers)", rm, k=k)

print("\n== F. references ==", flush=True)
run("ref", "all 18 GDN mixers zeroed", ALL)
run("ref", "keep only GDN layer 0 (whole-layer skip elsewhere)",
    [x for x in ALL if x != 0], skip=True)
run("ref", "GDN layer 0 whole layer skipped", [0], skip=True)

json.dump(dict(meta=vars(a), base_nll=float(base), base_ppl=float(np.exp(base)),
               base_acc=base_acc, base_gap=base_gap, groups=GROUPS, rows=rows),
          open(a.out, "w"), indent=2)
print("\nwrote", a.out)
