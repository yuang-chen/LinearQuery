"""Step 12: layer 0 vs layer 1 -- what makes the first GDN layer different?

Zeroing layer 0's mixer costs ~2,200x perplexity; zeroing layer 1's costs ~25%.  Both are
GDN layers with identical architecture, three positions apart in the same residual stream.
Why?  Five candidate explanations, each with its own test:

  G  GEOMETRY      layer 0 simply writes a much larger vector into the residual stream
                   -> A: relative write norms
  N  CONTENT       even at matched relative magnitude, layer 0's content is privileged
                   -> B: norm-matched random perturbation sweep, same epsilon for both layers
                      (separates "writes big" from "writes something special")
  I  INFORMATION   layer 0 encodes token identity; layer 1 encodes something else (or little)
                   -> C: the Part III information ladder + the Part V token-local condition
  R  REDUNDANCY    layer 1 is cheap to remove only because layers 2/4/5 duplicate it
                   -> D: joint ablations of layer sets vs matched random sets
  S  SPECIALISATION layer 1 helps a narrow set of positions rather than all of them
                   -> E: per-token NLL attribution, concentration, and whether layers 0 and 1
                      help the SAME positions
"""
import sys, json, argparse, random, re, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from datasets import load_dataset
from transformers import DynamicCache
from src.runner import Runner, hook_ctx

ap = argparse.ArgumentParser()
ap.add_argument("--n_stat", type=int, default=400)
ap.add_argument("--n_eval", type=int, default=16)
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--eps", default="0.03,0.1,0.3,1.0,3.0")
ap.add_argument("--out", default="results/exp12_layer01.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
tok, D = R.tok, R.cfg.hidden_size
GDN = R.gdn_layers


def corpus(split):
    d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split=split)
    return tok("\n\n".join(t for t in d["text"] if t.strip()), return_tensors="pt").input_ids[0]


tr_ids, te_ids = corpus("train"), corpus("test")
stat_seqs = [tr_ids[i * a.seq_len:(i + 1) * a.seq_len].unsqueeze(0).to(R.device)
             for i in range(min(a.n_stat, tr_ids.numel() // a.seq_len))]
eval_seqs = [te_ids[i * a.seq_len:(i + 1) * a.seq_len].unsqueeze(0).to(R.device)
             for i in range(a.n_eval)]
rows = []


class Cap:
    """Capture a mixer's output and the residual stream entering its decoder layer."""

    def __init__(self, L):
        self.L, self.out, self.inp = L, None, None

    def register(self):
        def fo(mod, args, output):
            self.out = output.detach()
        def fi(mod, args, kwargs):
            self.inp = (kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]).detach()
        h1 = R.mixer(self.L).register_forward_hook(fo)
        h2 = R.layers[self.L].register_forward_pre_hook(fi, with_kwargs=True)

        class Both:
            def remove(self_):
                h1.remove(); h2.remove()
        return Both()


# =============================================================== A. residual-stream geometry
print("== A. residual-stream geometry ==", flush=True)
caps = {L: Cap(L) for L in GDN}
acc = {L: [0.0, 0.0, 0.0] for L in GDN}       # sum |o|, sum |h_in|, sum |o|/|h_in|
nb = 0
with torch.no_grad():
    for s in eval_seqs:
        with hook_ctx(list(caps.values())):
            out = R.model(s, use_cache=False)
        for L in GDN:
            o, h = caps[L].out[0].float(), caps[L].inp[0].float()
            acc[L][0] += o.norm(dim=-1).mean().item()
            acc[L][1] += h.norm(dim=-1).mean().item()
            acc[L][2] += (o.norm(dim=-1) / h.norm(dim=-1).clamp(min=1e-6)).mean().item()
        nb += 1
print(f"{'layer':>6} {'|o_L|':>10} {'|h_in|':>10} {'|o|/|h_in|':>12}")
for L in GDN:
    o_, h_, r_ = [x / nb for x in acc[L]]
    print(f"{L:6d} {o_:10.2f} {h_:10.2f} {r_:12.3f}")
    rows.append(dict(kind="geometry", layer=L, o_norm=o_, h_norm=h_, ratio=r_))


# =============================================================== evaluation helpers
CUR = {}


class Zero:
    def __init__(self, L): self.L = L
    def register(self):
        def fn(mod, args, output):
            return torch.zeros_like(output)
        return R.mixer(self.L).register_forward_hook(fn)


class Noise:
    """Add eps * |o| * (random unit direction) to the mixer output, per token."""

    def __init__(self, L, eps, seed=0):
        self.L, self.eps, self.seed = L, eps, seed

    def register(self):
        def fn(mod, args, output):
            g = torch.Generator(device=output.device).manual_seed(self.seed)
            r = torch.randn(output.shape, generator=g, device=output.device, dtype=output.dtype)
            r = r / r.norm(dim=-1, keepdim=True)
            return output + self.eps * output.norm(dim=-1, keepdim=True) * r
        return R.mixer(self.L).register_forward_hook(fn)


class Surr:
    def __init__(self, L, mode, TAB): self.L, self.mode, self.TAB = L, mode, TAB
    def register(self):
        def fn(mod, args, output):
            ids = CUR["ids"]
            if self.mode == "global_mean":
                return self.TAB["glob"][self.L].expand_as(output).clone()
            idx = self.TAB["remap"][ids]
            return torch.where((idx >= 0)[..., None], self.TAB["mean"][self.L][idx.clamp(min=0)],
                               self.TAB["glob"][self.L]).to(output.dtype)
        return R.mixer(self.L).register_forward_hook(fn)


@torch.no_grad()
def nll_tokens(hooks=(), state_reset=None, seqs=None):
    """Per-position NLL over the eval corpus, concatenated."""
    out = []
    for s in (seqs or eval_seqs):
        CUR["ids"] = s
        if state_reset is None:
            with hook_ctx(hooks):
                lg = R.model(s, use_cache=False).logits
        else:
            Ls, w = state_reset
            cache = DynamicCache(config=R.model.config.get_text_config())
            outs = []
            with hook_ctx(hooks):
                for i in range(0, s.shape[1], w):
                    CUR["ids"] = s[:, i:i + w]
                    outs.append(R.model(s[:, i:i + w], past_key_values=cache, use_cache=True).logits)
                    for L in Ls:
                        rs = cache.layers[L].recurrent_states[0]
                        if rs is not None:
                            rs.zero_()
            lg = torch.cat(outs, 1)
        out.append(torch.nn.functional.cross_entropy(
            lg[0, :-1].float(), s[0, 1:], reduction="none").cpu())
    return torch.cat(out).numpy()


def rec(kind, name, d, base, **kw):
    r = dict(kind=kind, name=name, nll=float(d.mean()), ppl=float(np.exp(d.mean())),
             dnll=float(d.mean() - base.mean()), **kw)
    rows.append(r)
    print(f"  {name:44s} PPL {r['ppl']:11.2f}   dNLL {r['dnll']:+.4f}", flush=True)
    return r


base = nll_tokens()
print(f"\nintact PPL {np.exp(base.mean()):.2f}  (NLL {base.mean():.4f})")

# =============================================================== B. norm-matched perturbation
print("\n== B. norm-matched random perturbation (same relative epsilon per layer) ==", flush=True)
for L in [0, 1, 2, 4]:
    for eps in [float(x) for x in a.eps.split(",")]:
        rec("noise", f"layer {L}, eps={eps}", nll_tokens([Noise(L, eps)]), base, layer=L, eps=eps)

# =============================================================== C. ladder + token-local
print("\n== C. information ladder and token-local recomputation ==", flush=True)
print("  building token-type tables...", flush=True)
uniq = torch.unique(torch.cat([s[0] for s in stat_seqs]))
remap = torch.full((R.cfg.vocab_size,), -1, dtype=torch.long, device=R.device)
remap[uniq] = torch.arange(uniq.numel(), device=R.device)
sums = {L: torch.zeros(uniq.numel(), D, device=R.device) for L in [0, 1]}
cnts = torch.zeros(uniq.numel(), device=R.device)
with torch.no_grad():
    for s in stat_seqs:
        cs = {L: Cap(L) for L in [0, 1]}
        with hook_ctx(list(cs.values())):
            R.model(s, use_cache=False)
        idx = remap[s[0]]
        cnts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
        for L in [0, 1]:
            sums[L].index_add_(0, idx, cs[L].out[0].float())
TAB = {"remap": remap, "mean": {L: sums[L] / cnts[:, None].clamp(min=1) for L in [0, 1]},
       "glob": {L: sums[L].sum(0) / cnts.sum() for L in [0, 1]}}
cov = sum((remap[s[0]] >= 0).float().mean().item() for s in eval_seqs) / len(eval_seqs)
print(f"  table covers {cov:.3f} of eval tokens", flush=True)

w0 = {L: R.mixer(L).conv1d.weight.data.clone() for L in [0, 1]}
for L in [0, 1]:
    rec("ladder", f"layer {L}: zeros", nll_tokens([Zero(L)]), base, layer=L, rung="zero")
    rec("ladder", f"layer {L}: global mean", nll_tokens([Surr(L, "global_mean", TAB)]), base,
        layer=L, rung="global_mean")
    rec("ladder", f"layer {L}: token-type mean", nll_tokens([Surr(L, "token_mean", TAB)]), base,
        layer=L, rung="token_mean")
    R.mixer(L).conv1d.weight.data[..., :-1] = 0
    rec("ladder", f"layer {L}: token-local (conv1tap+norecur)",
        nll_tokens(state_reset=([L], 1)), base, layer=L, rung="token_local")
    rec("ladder", f"layer {L}: conv 1 tap only", nll_tokens(), base, layer=L, rung="conv1tap")
    R.mixer(L).conv1d.weight.data.copy_(w0[L])
    rec("ladder", f"layer {L}: no recurrence only", nll_tokens(state_reset=([L], 1)), base,
        layer=L, rung="norecur")

# =============================================================== D. redundancy
print("\n== D. redundancy: joint ablations vs matched random sets ==", flush=True)
SETS = [[0], [1], [2], [4], [1, 2], [1, 2, 4], [1, 2, 4, 5], [1, 2, 4, 5, 6], [0, 1]]
for S in SETS:
    rec("joint", f"zero mixers {S}", nll_tokens([Zero(L) for L in S]), base, layers=S, n=len(S))
rng = random.Random(0)
pool = [L for L in GDN if L != 0]
for n in (2, 3, 4):
    ds = []
    for t in range(3):
        S = rng.sample(pool, n)
        r = rec("joint_random", f"zero {n} random GDN mixers (trial {t}) {sorted(S)}",
                nll_tokens([Zero(L) for L in S]), base, layers=sorted(S), n=n)
        ds.append(r["dnll"])
    rows.append(dict(kind="joint_random_mean", n=n, dnll=float(np.mean(ds))))

# =============================================================== E. per-token attribution
print("\n== E. per-token attribution: do layers 0 and 1 help the same positions? ==", flush=True)
d0 = nll_tokens([Zero(0)]) - base
d1 = nll_tokens([Zero(1)]) - base
tgt = np.concatenate([s[0, 1:].cpu().numpy() for s in eval_seqs])
ctx = np.concatenate([s[0, :-1].cpu().numpy() for s in eval_seqs])
dec = {i: tok.decode([int(i)]) for i in np.unique(np.concatenate([tgt, ctx]))}
wordish, endsword = re.compile(r"^[A-Za-z0-9]"), re.compile(r"[A-Za-z0-9]$")


def cat_of(t, c):
    d, dc = dec[t], dec[c]
    if d.startswith(" ") and wordish.match(d[1:] or " "):
        return "word-initial"
    if wordish.match(d) and endsword.search(dc):
        return "word-continuation"
    if d.strip() == "":
        return "whitespace/newline"
    if re.match(r"^\s*\d", d):
        return "digit"
    if re.match(r"^\s*[^\w\s]", d):
        return "punctuation"
    return "other"


cats = np.array([cat_of(t, c) for t, c in zip(tgt, ctx)])
N = len(base)
print(f"  Pearson r(delta_L0, delta_L1) = {np.corrcoef(d0, d1)[0,1]:+.3f}")
print(f"  Spearman-ish r on ranks       = "
      f"{np.corrcoef(np.argsort(np.argsort(d0)), np.argsort(np.argsort(d1)))[0,1]:+.3f}")
rows.append(dict(kind="corr01", pearson=float(np.corrcoef(d0, d1)[0, 1])))
print(f"\n  {'cond':10s} {'mean':>9s} {'top1%':>8s} {'top5%':>8s} {'top10%':>8s} {'frac>0':>8s}")
for nm, d in (("zero L0", d0), ("zero L1", d1)):
    o = np.sort(d)[::-1]
    sh = [o[:max(1, int(N * f))].sum() / d.sum() for f in (.01, .05, .10)]
    print(f"  {nm:10s} {d.mean():+9.4f}" + "".join(f"{x:8.1%}" for x in sh) +
          f"{(d>0).mean():8.1%}")
    rows.append(dict(kind="concentration01", cond=nm, mean=float(d.mean()),
                     top1=float(sh[0]), top5=float(sh[1]), top10=float(sh[2]),
                     frac_pos=float((d > 0).mean())))
print(f"\n  {'category':22s} {'share':>7s} {'baseNLL':>8s} {'d(zero L0)':>12s} {'d(zero L1)':>12s}")
for c in ["word-initial", "word-continuation", "punctuation", "digit", "whitespace/newline", "other"]:
    m = cats == c
    if m.sum() == 0:
        continue
    print(f"  {c:22s} {m.mean():7.1%} {base[m].mean():8.3f} {d0[m].mean():+12.4f} {d1[m].mean():+12.4f}")
    rows.append(dict(kind="category01", category=c, share=float(m.mean()),
                     base=float(base[m].mean()), d0=float(d0[m].mean()), d1=float(d1[m].mean())))
print(f"\n  {'difficulty decile':18s} {'baseNLL':>8s} {'d(zero L0)':>12s} {'d(zero L1)':>12s}")
di = np.argsort(base)
for q in range(10):
    m = di[q * N // 10:(q + 1) * N // 10]
    print(f"  decile {q:<11d} {base[m].mean():8.3f} {d0[m].mean():+12.4f} {d1[m].mean():+12.4f}")
    rows.append(dict(kind="difficulty01", decile=q, base=float(base[m].mean()),
                     d0=float(d0[m].mean()), d1=float(d1[m].mean())))

json.dump(dict(meta=vars(a), base_nll=float(base.mean()), coverage=cov, rows=rows),
          open(a.out, "w"), indent=2)
np.savez_compressed(a.out.replace(".json", "_raw.npz"), base=base, d0=d0, d1=d1, cats=cats)
print("\nwrote", a.out)
