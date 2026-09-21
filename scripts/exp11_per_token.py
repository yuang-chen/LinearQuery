"""Step 11: where does the affine re-embedding actually lose its nats?

Part IV: replacing layer 0's mixer with A.embed(x)+b reaches the context-free ceiling on
average (PPL 90.8 vs a token-type lookup's 92.8) but still costs 4.2x over intact (21.9).
That residual is only ~8% of the output's variance yet a large share of the loss, which
usually means it is concentrated on a minority of positions.  This finds them.

Per-position NLL(t) = -log p(x_{t+1} | x_<=t) in nats; Delta(t) = NLL_ablated - NLL_intact.
Perplexity is exp(mean NLL), so mean Delta is the log-scale damage and the distribution of
Delta says whether it is broad or concentrated.

Conditions (all on layer 0 only):
  affine        A.embed(x)+b            -- the Part IV substitution
  token_mean    per-type mean output    -- the context-free ceiling (is the loss from the
                                          affine *form*, or from being context-free at all?)
  conv1tap      conv window 4 -> 1      -- the detokenisation probe
  norecur       recurrent state reset every token

Breakdowns test the detokenisation hypothesis (damage on word-continuation pieces) against
frequency, sequence position, and baseline difficulty.
"""
import sys, json, argparse, re, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from datasets import load_dataset
from transformers import DynamicCache
from src.runner import Runner, hook_ctx

ap = argparse.ArgumentParser()
ap.add_argument("--n_stat", type=int, default=2000)
ap.add_argument("--n_eval", type=int, default=32)
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--ridge", type=float, default=1e1)
ap.add_argument("--out", default="results/exp11_per_token.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
tok, D = R.tok, R.cfg.hidden_size
EMB = (R.model.model.embed_tokens if hasattr(R.model.model, "embed_tokens")
       else R.model.model.language_model.embed_tokens)


def corpus(split):
    d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split=split)
    return tok("\n\n".join(t for t in d["text"] if t.strip()), return_tensors="pt").input_ids[0]


tr_ids, te_ids = corpus("train"), corpus("test")
stat_seqs = [tr_ids[i * a.seq_len:(i + 1) * a.seq_len].unsqueeze(0).to(R.device)
             for i in range(min(a.n_stat, tr_ids.numel() // a.seq_len))]
eval_seqs = [te_ids[i * a.seq_len:(i + 1) * a.seq_len].unsqueeze(0).to(R.device)
             for i in range(a.n_eval)]


class Grab:
    def __init__(self): self.value = None
    def register(self):
        def fn(mod, args, output):
            self.value = output.detach()
        return R.mixer(0).register_forward_hook(fn)


# --------------------------------------------------------------- fit the affine map
print(f"fitting on {len(stat_seqs)} train sequences...", flush=True)
uniq = torch.unique(torch.cat([s[0] for s in stat_seqs]))
remap = torch.full((R.cfg.vocab_size,), -1, dtype=torch.long, device=R.device)
remap[uniq] = torch.arange(uniq.numel(), device=R.device)
sums = torch.zeros(uniq.numel(), D, device=R.device)
cnts = torch.zeros(uniq.numel(), device=R.device)
with torch.no_grad():
    for si, s in enumerate(stat_seqs):
        g = Grab()
        with hook_ctx([g]):
            R.model(s, use_cache=False)
        idx = remap[s[0]]
        sums.index_add_(0, idx, g.value[0].float())
        cnts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
        if (si + 1) % 500 == 0:
            print(f"  {si+1}/{len(stat_seqs)}", flush=True)
M = sums / cnts[:, None].clamp(min=1)
glob = sums.sum(0) / cnts.sum()
E = EMB.weight[uniq].float()
X = torch.cat([E, torch.ones(E.shape[0], 1, device=R.device)], 1)
w = cnts.clamp(min=1).sqrt()[:, None]
G = (X * w).T @ (X * w) + a.ridge * torch.eye(D + 1, device=R.device)
W = torch.linalg.solve(G, (X * w).T @ (M * w))
A, b = W[:D], W[D]
print("fit done", flush=True)

CUR = {}


class Sub:
    def __init__(self, mode): self.mode = mode
    def register(self):
        def fn(mod, args, output):
            ids = CUR["ids"]
            if self.mode == "affine":
                return (EMB.weight[ids].float() @ A + b).to(output.dtype)
            idx = remap[ids]
            return torch.where((idx >= 0)[..., None], M[idx.clamp(min=0)], glob).to(output.dtype)
        return R.mixer(0).register_forward_hook(fn)


@torch.no_grad()
def per_token_nll(seq, hooks=(), state_reset=None):
    """Returns NLL in nats for each predicted position (length seq_len-1)."""
    CUR["ids"] = seq
    if state_reset is None:
        with hook_ctx(hooks):
            lg = R.model(seq, use_cache=False).logits
    else:
        cache = DynamicCache(config=R.model.config.get_text_config())
        outs = []
        with hook_ctx(hooks):
            for i in range(0, seq.shape[1], state_reset):
                CUR["ids"] = seq[:, i:i + state_reset]
                outs.append(R.model(seq[:, i:i + state_reset],
                                    past_key_values=cache, use_cache=True).logits)
                rs = cache.layers[0].recurrent_states[0]
                if rs is not None:
                    rs.zero_()
        lg = torch.cat(outs, 1)
    return torch.nn.functional.cross_entropy(
        lg[0, :-1].float(), seq[0, 1:], reduction="none").cpu()


# "token_local" = conv window 1 AND state reset every token: layer 0 recomputed with no
# access to any other token.  This is the true context-free *function*, as opposed to
# token_mean/affine which are context-free *estimates* of it.
CONDS = {"affine": dict(hooks=[Sub("affine")]),
         "token_mean": dict(hooks=[Sub("token_mean")]),
         "token_local": dict(hooks=[], weight_edit=True, state_reset=1),
         "conv1tap": dict(hooks=[], weight_edit=True),
         "norecur": dict(hooks=[], state_reset=1)}

base_nll, cond_nll = [], {k: [] for k in CONDS}
targets, contexts, posns = [], [], []
for s in eval_seqs:
    base_nll.append(per_token_nll(s))
    targets.append(s[0, 1:].cpu()); contexts.append(s[0, :-1].cpu())
    posns.append(torch.arange(1, s.shape[1]))
w0 = R.mixer(0).conv1d.weight.data.clone()
for name, cfg in CONDS.items():
    if cfg.get("weight_edit"):
        R.mixer(0).conv1d.weight.data[..., :-1] = 0
    for s in eval_seqs:
        cond_nll[name].append(per_token_nll(s, cfg["hooks"], cfg.get("state_reset")))
    if cfg.get("weight_edit"):
        R.mixer(0).conv1d.weight.data.copy_(w0)
    print(f"  {name} done", flush=True)

base = torch.cat(base_nll).numpy()
tgt = torch.cat(targets).numpy()
ctx = torch.cat(contexts).numpy()
pos = torch.cat(posns).numpy()
delta = {k: (torch.cat(v).numpy() - base) for k, v in cond_nll.items()}
N = len(base)
print(f"\n{N} predicted positions; intact mean NLL {base.mean():.4f} "
      f"(PPL {np.exp(base.mean()):.2f})")

# --------------------------------------------------------------- token categories
dec = {i: tok.decode([i]) for i in np.unique(np.concatenate([tgt, ctx]))}
wordish = re.compile(r"^[A-Za-z0-9]")
endsword = re.compile(r"[A-Za-z0-9]$")


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
freq_rank = {}
cn = cnts.cpu().numpy()
rm = remap.cpu().numpy()
tgt_count = np.array([cn[rm[t]] if rm[t] >= 0 else 0.0 for t in tgt])

rows = []
print("\n=== mean Delta NLL (nats) by condition ===")
for k, d in delta.items():
    print(f"  {k:12s} mean {d.mean():+.4f}   median {np.median(d):+.4f}   "
          f"PPL {np.exp(base.mean()+d.mean()):8.2f}")
    rows.append(dict(kind="overall", cond=k, mean=float(d.mean()),
                     median=float(np.median(d)), ppl=float(np.exp(base.mean() + d.mean()))))

print("\n=== concentration: share of total Delta from the worst positions ===")
print(f"{'cond':12s} {'top 1%':>8s} {'top 5%':>8s} {'top 10%':>8s} {'top 25%':>8s} {'>0':>8s}")
for k, d in delta.items():
    tot = d.sum()
    o = np.sort(d)[::-1]
    shares = [o[:max(1, int(N * f))].sum() / tot for f in (.01, .05, .10, .25)]
    print(f"{k:12s} " + "".join(f"{s:8.2%}" for s in shares) + f"{(d>0).mean():8.1%}")
    rows.append(dict(kind="concentration", cond=k, top1=float(shares[0]), top5=float(shares[1]),
                     top10=float(shares[2]), top25=float(shares[3]), frac_pos=float((d > 0).mean())))

print("\n=== mean Delta by target-token category ===")
order = ["word-initial", "word-continuation", "punctuation", "digit", "whitespace/newline", "other"]
hdr = f"{'category':22s} {'n':>7s} {'share':>7s} {'baseNLL':>8s}" + "".join(f"{k:>12s}" for k in delta)
print(hdr)
for c in order:
    m = cats == c
    if m.sum() == 0:
        continue
    line = f"{c:22s} {m.sum():7d} {m.mean():7.1%} {base[m].mean():8.3f}"
    line += "".join(f"{delta[k][m].mean():+12.4f}" for k in delta)
    print(line)
    rows.append(dict(kind="category", category=c, n=int(m.sum()), share=float(m.mean()),
                     base=float(base[m].mean()),
                     **{f"d_{k}": float(delta[k][m].mean()) for k in delta}))

print("\n=== share of each condition's TOTAL damage, by category ===")
print(f"{'category':22s}" + "".join(f"{k:>12s}" for k in delta))
for c in order:
    m = cats == c
    if m.sum() == 0:
        continue
    print(f"{c:22s}" + "".join(f"{delta[k][m].sum()/delta[k].sum():11.1%} " for k in delta))
    rows.append(dict(kind="damage_share", category=c,
                     **{f"s_{k}": float(delta[k][m].sum() / delta[k].sum()) for k in delta}))

print("\n=== mean Delta by target-token frequency (train-corpus count) ===")
bins = [(0, 1), (1, 10), (10, 100), (100, 1000), (1000, 10**9)]
for lo, hi in bins:
    m = (tgt_count >= lo) & (tgt_count < hi)
    if m.sum() == 0:
        continue
    print(f"  count [{lo:>5},{hi:>6}) n={m.sum():6d} baseNLL {base[m].mean():6.3f}"
          + "".join(f"  {k} {delta[k][m].mean():+.3f}" for k in delta))
    rows.append(dict(kind="frequency", lo=lo, hi=hi, n=int(m.sum()), base=float(base[m].mean()),
                     **{f"d_{k}": float(delta[k][m].mean()) for k in delta}))

print("\n=== mean Delta by position in sequence ===")
for lo, hi in [(1, 8), (8, 32), (32, 128), (128, 512)]:
    m = (pos >= lo) & (pos < hi)
    print(f"  pos [{lo:>4},{hi:>4}) n={m.sum():6d} baseNLL {base[m].mean():6.3f}"
          + "".join(f"  {k} {delta[k][m].mean():+.3f}" for k in delta))
    rows.append(dict(kind="position", lo=lo, hi=hi, n=int(m.sum()),
                     **{f"d_{k}": float(delta[k][m].mean()) for k in delta}))

print("\n=== mean Delta by baseline difficulty decile (easy -> hard) ===")
dec_idx = np.argsort(base)
for q in range(10):
    m = dec_idx[q * N // 10:(q + 1) * N // 10]
    print(f"  decile {q} baseNLL {base[m].mean():6.3f}"
          + "".join(f"  {k} {delta[k][m].mean():+.3f}" for k in delta))
    rows.append(dict(kind="difficulty", decile=q, base=float(base[m].mean()),
                     **{f"d_{k}": float(delta[k][m].mean()) for k in delta}))

print("\n=== do the conditions damage the SAME positions? (Pearson r of Delta) ===")
ks = list(delta)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        r = float(np.corrcoef(delta[ks[i]], delta[ks[j]])[0, 1])
        print(f"  {ks[i]:12s} vs {ks[j]:12s}  r = {r:+.3f}")
        rows.append(dict(kind="corr", a=ks[i], b=ks[j], r=r))

json.dump(dict(meta=vars(a), n_positions=int(N), base_nll=float(base.mean()), rows=rows),
          open(a.out, "w"), indent=2)
np.savez_compressed(a.out.replace(".json", "_raw.npz"), base=base, tgt=tgt, ctx=ctx, pos=pos,
                    cats=cats, **{f"d_{k}": v for k, v in delta.items()})
print("\nwrote", a.out)
