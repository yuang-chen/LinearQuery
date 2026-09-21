"""Step 15: which layer inside G3 carries retrieval, and what do individual GDN heads do?

Part VII showed G3 [12,13,14] is near-free for perplexity but near-fatal for retrieval, while
Part I found each of layers 12/13/14 individually harmless -- a redundancy.  Part A resolves
the group into layers by ablating every subset.  Parts B-D then resolve the two key layers
(GDN 0, and whichever layer of G3 dominates) into their 16 delta-rule heads.

A GDN layer has 16 value heads of width 128; head h owns slice [128h : 128(h+1)] of the
out_proj input (after the gated RMSNorm), so zeroing that slice removes exactly that head's
contribution to the residual stream.

  A  subsets of {12,13,14}: retrieval accuracy/gap, perplexity, and reader attention
  B  per-head ablation in layer 0 and layer x: perplexity and retrieval
  C  per-head "keep only this head"
  D  per-head characterisation: write norm, forget-gate half-life, and how much of the head's
     output is explained by token identity (between-type / total variance)
"""
import sys, json, argparse, itertools, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from datasets import load_dataset
from src.runner import Runner, hook_ctx, Capture
from src.qkv import AttnWeights
from src.task import make_examples
from src.harness import group_examples

ap = argparse.ArgumentParser()
ap.add_argument("--n_eval", type=int, default=12)
ap.add_argument("--n_stat", type=int, default=200)
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--layers", default="0")      # extra layers to head-resolve besides layer x
ap.add_argument("--out", default="results/exp15_heads.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
R.model.config.get_text_config()._attn_implementation = "eager"
NH = R.cfg.linear_num_value_heads
HD = R.cfg.linear_value_head_dim
print(f"GDN: {NH} value heads x {HD} dims", flush=True)

d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
ids = R.tok("\n\n".join(t for t in d["text"] if t.strip()), return_tensors="pt").input_ids[0]
seqs = [ids[i * a.seq_len:(i + 1) * a.seq_len].unsqueeze(0).to(R.device) for i in range(a.n_eval)]
stat_seqs = [ids[(a.n_eval + i) * a.seq_len:(a.n_eval + i + 1) * a.seq_len].unsqueeze(0).to(R.device)
             for i in range(a.n_stat)]

exs = []
for i, (ti, di) in enumerate([(2, 5), (5, 2), (1, 6), (6, 1)]):
    exs += make_examples(25, 8, seed=11 + i, target_pos=ti, distract_pos=di)
groups_ex = group_examples(R, exs)


class ZeroMixer:
    def __init__(self, L): self.L = L
    def register(self):
        def fn(mod, args, output):
            return torch.zeros_like(output)
        return R.mixer(self.L).register_forward_hook(fn)


class HeadZero:
    """Zero the given delta-rule heads of GDN layer L (slice of the out_proj input)."""

    def __init__(self, L, heads, keep=False):
        self.L, self.heads, self.keep = L, set(heads), keep

    def register(self):
        def fn(mod, args, kwargs):
            t = (args[0] if args else kwargs["input"]).clone()
            for h in range(NH):
                drop = (h in self.heads) if not self.keep else (h not in self.heads)
                if drop:
                    t[..., h * HD:(h + 1) * HD] = 0
            return (t,) + tuple(args[1:]), kwargs
        return R.mixer(self.L).out_proj.register_forward_pre_hook(fn, with_kwargs=True)


@torch.no_grad()
def ppl(hooks=()):
    v = []
    for s in seqs:
        with hook_ctx(hooks):
            lg = R.model(s, use_cache=False).logits
        v.append(torch.nn.functional.cross_entropy(lg[0, :-1].float(), s[0, 1:]).item())
    return float(np.mean(v))


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


@torch.no_grad()
def reader_attn(hooks=(), L=15, h=5):
    vals = []
    for g in groups_ex:
        aw = AttnWeights(R.mixer(L))
        with hook_ctx([aw] + list(hooks)):
            R.model(g.corr_ids, use_cache=False)
        vals += aw.value[:, h, g.pos["final"], g.pos["target_value"]].float().cpu().tolist()
    return float(np.mean(vals))


base_nll = ppl(); base_acc, base_gap = retrieval(); base_att = reader_attn()
print(f"intact: PPL {np.exp(base_nll):.2f}  acc {base_acc:.3f}  gap {base_gap:+.2f}  "
      f"L15H5 attn {base_att:.3f}\n", flush=True)
rows = []

# =============================================================== A. resolve G3
print("== A. subsets of G3 = {12,13,14} ==", flush=True)
G3 = [12, 13, 14]
for k in (1, 2, 3):
    for S in itertools.combinations(G3, k):
        h = [ZeroMixer(L) for L in S]
        acc, gap = retrieval(h)
        r = dict(kind="g3_subset", layers=list(S), n=k, ppl=float(np.exp(ppl(h))),
                 dnll=float(ppl(h) - base_nll), acc=acc, gap=gap, attn=reader_attn(h))
        rows.append(r)
        print(f"  remove {str(list(S)):14s} PPL {r['ppl']:8.2f}  acc {acc:.3f}  "
              f"gap {gap:+6.2f}  L15H5 attn {r['attn']:.3f}", flush=True)
# which single layer is most needed = keep only one of the three
for L in G3:
    S = [x for x in G3 if x != L]
    h = [ZeroMixer(x) for x in S]
    acc, gap = retrieval(h)
    rows.append(dict(kind="g3_keep_one", keep=L, ppl=float(np.exp(ppl(h))), acc=acc, gap=gap,
                     attn=reader_attn(h)))
    print(f"  keep only {L:2d} of G3    acc {acc:.3f}  gap {gap:+6.2f}", flush=True)

g3 = [r for r in rows if r["kind"] == "g3_subset" and r["n"] == 1]
LX = min(g3, key=lambda r: r["acc"])["layers"][0]
print(f"\n  -> most retrieval-critical single layer in G3: {LX}", flush=True)
HEAD_LAYERS = sorted({0, LX} | {int(x) for x in a.layers.split(",") if x})

# =============================================================== B/C. per-head
print(f"\n== B. per-head ablation in layers {HEAD_LAYERS} ==", flush=True)
for L in HEAD_LAYERS:
    print(f"  --- GDN layer {L} ---", flush=True)
    for h in range(NH):
        hk = [HeadZero(L, [h])]
        acc, gap = retrieval(hk)
        r = dict(kind="head_zero", layer=L, head=h, ppl=float(np.exp(ppl(hk))),
                 dnll=float(ppl(hk) - base_nll), acc=acc, gap=gap)
        rows.append(r)
        print(f"    zero head {h:2d}  PPL {r['ppl']:9.2f}  dNLL {r['dnll']:+7.4f}  "
              f"acc {acc:.3f}  gap {gap:+6.2f}", flush=True)

print(f"\n== C. keep only one head ==", flush=True)
for L in HEAD_LAYERS:
    for h in range(NH):
        hk = [HeadZero(L, [h], keep=True)]
        acc, gap = retrieval(hk)
        rows.append(dict(kind="head_keep", layer=L, head=h, ppl=float(np.exp(ppl(hk))),
                         dnll=float(ppl(hk) - base_nll), acc=acc, gap=gap))
    v = [r for r in rows if r["kind"] == "head_keep" and r["layer"] == L]
    best = max(v, key=lambda r: r["acc"])
    print(f"  layer {L}: best single head for retrieval = head {best['head']} "
          f"(acc {best['acc']:.3f}), PPL range {min(x['ppl'] for x in v):.0f}"
          f"-{max(x['ppl'] for x in v):.0f}", flush=True)

# =============================================================== D. characterise heads
print(f"\n== D. head characterisation (write norm, memory half-life, token-identity R2) ==",
      flush=True)
for L in HEAD_LAYERS:
    cap_in = Capture(R.mixer(L), mode="in")
    hnorm = torch.zeros(NH, device=R.device)
    glife = torch.zeros(NH, device=R.device)
    uniq = torch.unique(torch.cat([s[0] for s in stat_seqs]))
    remap = torch.full((R.cfg.vocab_size,), -1, dtype=torch.long, device=R.device)
    remap[uniq] = torch.arange(uniq.numel(), device=R.device)
    sums = torch.zeros(uniq.numel(), NH * HD, device=R.device)
    cnts = torch.zeros(uniq.numel(), device=R.device)
    sq = torch.zeros(NH, device=R.device)
    n_tok = 0
    store = {}

    class CapOutProjIn:
        def register(self):
            def fn(mod, args, kwargs):
                store["v"] = (args[0] if args else kwargs["input"]).detach()
            return R.mixer(L).out_proj.register_forward_pre_hook(fn, with_kwargs=True)

    m = R.mixer(L)
    with torch.no_grad():
        for s in stat_seqs:
            with hook_ctx([cap_in, CapOutProjIn()]):
                R.model(s, use_cache=False)
            o = store["v"][0].float()                      # [T, NH*HD]
            oh = o.view(-1, NH, HD)
            hnorm += oh.norm(dim=-1).mean(0)
            aa = m.in_proj_a(cap_in.value[0].float())
            gl = -m.A_log.float().exp() * F.softplus(aa + m.dt_bias.float())
            glife += (torch.log(torch.tensor(0.5, device=R.device)) / gl.mean(0)).clamp(max=1e6)
            idx = remap[s[0]]
            sums.index_add_(0, idx, o)
            cnts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
            sq += oh.pow(2).sum(-1).sum(0)
            n_tok += o.shape[0]
    nb = len(stat_seqs)
    hnorm /= nb; glife /= nb
    M = sums / cnts[:, None].clamp(min=1)
    glob = sums.sum(0) / cnts.sum()
    Mh = M.view(-1, NH, HD); gh = glob.view(NH, HD)
    var_tot = (sq - n_tok * gh.pow(2).sum(-1)) / n_tok
    var_bet = (cnts[:, None, None] * (Mh - gh).pow(2)).sum(0).sum(-1) / n_tok
    r2 = (var_bet / var_tot.clamp(min=1e-9)).cpu().numpy()
    print(f"  --- GDN layer {L} ---")
    print(f"  {'head':>5} {'|o_h|':>9} {'half-life':>11} {'R2_token_id':>12} "
          f"{'dNLL(zero)':>11} {'acc(zero)':>10}")
    hz = {r["head"]: r for r in rows if r["kind"] == "head_zero" and r["layer"] == L}
    for h in range(NH):
        rows.append(dict(kind="head_char", layer=L, head=h, norm=float(hnorm[h]),
                         half_life=float(glife[h]), r2_token=float(r2[h])))
        print(f"  {h:5d} {hnorm[h]:9.3f} {glife[h]:11.2f} {r2[h]:12.3f} "
              f"{hz[h]['dnll']:+11.4f} {hz[h]['acc']:10.3f}")

json.dump(dict(meta=vars(a), base_nll=base_nll, base_ppl=float(np.exp(base_nll)),
               base_acc=base_acc, base_gap=base_gap, base_attn=base_att, layer_x=LX,
               head_layers=HEAD_LAYERS, rows=rows), open(a.out, "w"), indent=2)
print("\nwrote", a.out)
