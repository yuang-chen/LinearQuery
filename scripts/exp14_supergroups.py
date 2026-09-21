"""Step 14: the roles of the GDN super-groups [G1+G2] and [G4+G5].

Part VII found G0 [0,1,2] critical for perplexity and G3 [12,13,14] critical for retrieval
(it feeds the reader head at softmax layer 15).  That leaves two blocks either side of G3:

    G1+G2 = [4,5,6,8,9,10]      between softmax 3 and softmax 11, i.e. BEFORE the reader
    G4+G5 = [16,17,18,20,21,22] between softmax 15 and the output, i.e. AFTER the reader

Ablation size alone cannot distinguish their roles, so four probes are run alongside:

  A  ablation grid: each group, each adjacent pair, the two super-groups, keep-only-[G0+G3],
     and matched random 6-layer controls.  Metrics: perplexity AND retrieval.
  B  logit-lens trajectory: D = logit(clean answer) - logit(corrupt answer) read off the
     residual stream at the final position after every layer.  Says WHERE the answer is
     formed and where each ablation breaks the trajectory.
  C  reader attention: P(final -> source value token) for heads L15H5 and L19H1.  Separates
     "the reader can no longer find the source" from "the reader finds it but the answer is
     lost afterwards".
  D  sequence-position and recurrent-horizon dependence on WikiText: whether a block's value
     is local or long-range.
"""
import sys, json, argparse, random, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from datasets import load_dataset
from transformers import DynamicCache
from src.runner import Runner, hook_ctx
from src.qkv import AttnWeights
from src.task import make_examples
from src.harness import group_examples

ap = argparse.ArgumentParser()
ap.add_argument("--n_eval", type=int, default=16)
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--n_random", type=int, default=5)
ap.add_argument("--out", default="results/exp14_supergroups.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
R.model.config.get_text_config()._attn_implementation = "eager"
GDN = R.gdn_layers
G = [GDN[i:i + 3] for i in range(0, len(GDN), 3)]
SUPER = {"G1+G2": G[1] + G[2], "G4+G5": G[4] + G[5], "G0+G3": G[0] + G[3]}
print("groups:", G, flush=True)

d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
ids = R.tok("\n\n".join(t for t in d["text"] if t.strip()), return_tensors="pt").input_ids[0]
seqs = [ids[i * a.seq_len:(i + 1) * a.seq_len].unsqueeze(0).to(R.device) for i in range(a.n_eval)]

exs = []
for i, (ti, di) in enumerate([(2, 5), (5, 2), (1, 6), (6, 1)]):
    exs += make_examples(25, 8, seed=11 + i, target_pos=ti, distract_pos=di)
groups_ex = group_examples(R, exs)
NORM = R.model.model.norm if hasattr(R.model.model, "norm") else R.model.model.language_model.norm
HEAD = R.model.lm_head


class Zero:
    def __init__(self, L): self.L = L
    def register(self):
        def fn(mod, args, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + tuple(output[1:])
            return torch.zeros_like(output)
        return R.mixer(self.L).register_forward_hook(fn)


class CapResid:
    def __init__(self, L): self.L, self.value = L, None
    def register(self):
        def fn(mod, args, output):
            self.value = (output[0] if isinstance(output, tuple) else output).detach()
        return R.layers[self.L].register_forward_hook(fn)


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


base_v = ppl_per_seq(); base = base_v.mean()
base_acc, base_gap = retrieval()
print(f"intact PPL {np.exp(base):.2f}  acc {base_acc:.3f}  gap {base_gap:+.2f}\n", flush=True)
rows = []


def run(kind, name, zl, **kw):
    h = [Zero(L) for L in zl]
    v = ppl_per_seq(h)
    acc, gap = retrieval(h)
    r = dict(kind=kind, name=name, layers=list(zl), n=len(zl), ppl=float(np.exp(v.mean())),
             dnll=float(v.mean() - base), acc=acc, gap=gap, **kw)
    rows.append(r)
    print(f"  {name:36s} n={len(zl):2d}  PPL {r['ppl']:11.2f}  dNLL {r['dnll']:+7.3f}  "
          f"acc {acc:.3f}  gap {gap:+6.2f}", flush=True)
    return r


print("== A. ablation grid ==", flush=True)
for i in range(6):
    run("single", f"remove G{i}", G[i], g=i)
for i in range(5):
    run("adjacent", f"remove G{i}+G{i+1}", G[i] + G[i + 1], g=i)
for nm, ls in SUPER.items():
    run("super", f"remove {nm}", ls)
run("super", "keep only G0+G3", [L for L in GDN if L not in SUPER["G0+G3"]])
run("super", "keep only G1+G2", [L for L in GDN if L not in SUPER["G1+G2"]])
run("super", "keep only G4+G5", [L for L in GDN if L not in SUPER["G4+G5"]])
rng = random.Random(0)
dr = []
for t in range(a.n_random):
    S = sorted(rng.sample(GDN, 6))
    dr.append(run("random6", f"random 6 {S}", S)["dnll"])
rows.append(dict(kind="random6_summary", mean=float(np.mean(dr)), std=float(np.std(dr))))
print(f"  random-6 mean dNLL {np.mean(dr):+.3f} (sd {np.std(dr):.3f})", flush=True)

# =============================================================== B. logit-lens trajectory
print("\n== B. logit-lens: where is the answer formed? ==", flush=True)
CONDS = {"intact": [], "remove G0": G[0], "remove G1+G2": SUPER["G1+G2"],
         "remove G3": G[3], "remove G4+G5": SUPER["G4+G5"]}
for cname, zl in CONDS.items():
    traj = np.zeros(R.n_layers)
    nb = 0
    for g in groups_ex:
        caps = [CapResid(L) for L in range(R.n_layers)]
        with torch.no_grad(), hook_ctx(caps + [Zero(L) for L in zl]):
            R.model(g.clean_ids, use_cache=False)
        cl = [c.value[:, -1] for c in caps]
        caps = [CapResid(L) for L in range(R.n_layers)]
        with torch.no_grad(), hook_ctx(caps + [Zero(L) for L in zl]):
            R.model(g.corr_ids, use_cache=False)
        co = [c.value[:, -1] for c in caps]
        for L in range(R.n_layers):
            with torch.no_grad():
                dc = g.D(HEAD(NORM(cl[L])))
                dx = g.D(HEAD(NORM(co[L])))
            traj[L] += (dc - dx).mean().item()
        nb += 1
    traj /= nb
    rows.append(dict(kind="logitlens", cond=cname, traj=traj.tolist()))
    print(f"  {cname:14s} " + " ".join(f"{traj[L]:+5.1f}" for L in (3, 7, 11, 14, 15, 18, 19, 22, 23)),
          flush=True)
print("   (layers shown: 3 7 11 14 | 15 | 18 19 22 23 ; softmax layers are 3,7,11,15,19,23)")

# =============================================================== C. reader attention
print("\n== C. reader attention P(final -> source value token) ==", flush=True)
for cname, zl in CONDS.items():
    out = {}
    for (L, h) in ((15, 5), (19, 1)):
        vals = []
        for g in groups_ex:
            aw = AttnWeights(R.mixer(L))
            with torch.no_grad(), hook_ctx([aw] + [Zero(x) for x in zl]):
                R.model(g.corr_ids, use_cache=False)
            vals += aw.value[:, h, g.pos["final"], g.pos["target_value"]].float().cpu().tolist()
        out[f"L{L}H{h}"] = float(np.mean(vals))
    rows.append(dict(kind="attn", cond=cname, **out))
    print(f"  {cname:14s} " + "  ".join(f"{k} {v:.3f}" for k, v in out.items()), flush=True)

# =============================================================== D. locality
print("\n== D. sequence-position dependence (WikiText) ==", flush=True)


@torch.no_grad()
def per_token(hooks=(), reset=None):
    out = []
    for s in seqs:
        if reset is None:
            with hook_ctx(hooks):
                lg = R.model(s, use_cache=False).logits
        else:
            Ls, w = reset
            cache = DynamicCache(config=R.model.config.get_text_config())
            o = []
            with hook_ctx(hooks):
                for i in range(0, s.shape[1], w):
                    o.append(R.model(s[:, i:i + w], past_key_values=cache, use_cache=True).logits)
                    for L in Ls:
                        rs = cache.layers[L].recurrent_states[0]
                        if rs is not None:
                            rs.zero_()
            lg = torch.cat(o, 1)
        out.append(torch.nn.functional.cross_entropy(
            lg[0, :-1].float(), s[0, 1:], reduction="none").cpu())
    return torch.cat(out).numpy()


bt = per_token()
pos = np.concatenate([np.arange(1, s.shape[1]) for s in seqs])
BUCK = [(1, 8), (8, 32), (32, 128), (128, 512)]
print(f"  {'condition':16s}" + "".join(f"{f'pos {lo}-{hi}':>14s}" for lo, hi in BUCK))
for cname in ("remove G0", "remove G1+G2", "remove G3", "remove G4+G5"):
    dd = per_token([Zero(L) for L in CONDS[cname]]) - bt
    vals = [float(dd[(pos >= lo) & (pos < hi)].mean()) for lo, hi in BUCK]
    rows.append(dict(kind="position", cond=cname, buckets=[list(b) for b in BUCK], vals=vals))
    print(f"  {cname:16s}" + "".join(f"{v:+14.4f}" for v in vals), flush=True)

print("\n== D2. recurrent-horizon restricted to each block ==", flush=True)
print(f"  {'block':10s}" + "".join(f"{f'w={w}':>12s}" for w in (1, 8, 64)))
for nm, ls in (("G0", G[0]), ("G1+G2", SUPER["G1+G2"]), ("G3", G[3]), ("G4+G5", SUPER["G4+G5"])):
    vals = []
    for w in (1, 8, 64):
        dd = per_token(reset=(ls, w)) - bt
        vals.append(float(dd.mean()))
    rows.append(dict(kind="horizon", block=nm, windows=[1, 8, 64], dnll=vals))
    print(f"  {nm:10s}" + "".join(f"{v:+12.4f}" for v in vals), flush=True)

json.dump(dict(meta=vars(a), base_nll=float(base), base_ppl=float(np.exp(base)),
               base_acc=base_acc, base_gap=base_gap, groups=G, rows=rows),
          open(a.out, "w"), indent=2)
print("\nwrote", a.out)
