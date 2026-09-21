"""Step 10: is GDN layer 0 literally an affine re-embedding?

Part III showed layer 0's mixer output is ~76% predictable from the current token alone
(a context-free token-type lookup takes perplexity from 38,919 to 111.7 against 17.5 intact).
That says "second embedding" as a *function of the token*, but not that the function is simple.

Here the map is fitted explicitly:            o_0(t)  ~=  A . embed(x_t) + b
by weighted least squares on the per-token-type means (equivalent to OLS over all
occurrences, since the regressor is constant within a token type), then substituted at run
time.  A low-rank sweep of A follows, because a rank-r fit that still works means layer 0's
mixer is foldable into the embedding table at r x 1024 x 2 parameters.

Reference points on the same axis:
  intact                 17.5   (nothing replaced)
  token-type mean       111.7   (rung 3: the *ceiling* for any context-free map of the token)
  global mean        14,729     (no token information at all)
  zeros              38,919

  affine ~ token-type mean  =>  the token-wise map is linear in the embedding: layer 0 is a
                                rank-limited recode of the embedding table and can be folded in
  affine >> token-type mean =>  the map is genuinely non-linear; identity is enough but the
                                shape of the map is not trivial

Variance decomposition (how context-free is it, and how linear):
  R2_identity = between-type variance / total variance  -- fraction of o_0 explained by token id
  R2_affine   = fraction of total variance explained by A.embed + b
Generalisation: A is fitted on 80% of token types and R2 reported on the held-out 20%, so the
map is not merely memorising the types it saw.
"""
import sys, json, argparse, torch
sys.path.insert(0, "/user/yac/LinearAblation")
from datasets import load_dataset
from src.runner import Runner, hook_ctx
from src.task import make_examples
from src.harness import group_examples

ap = argparse.ArgumentParser()
ap.add_argument("--n_stat", type=int, default=200)
ap.add_argument("--n_eval", type=int, default=16)
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--layer", type=int, default=0)
ap.add_argument("--ranks", default="8,16,32,64,128,256,512,1024")
ap.add_argument("--ridge", type=float, default=1e-2)
ap.add_argument("--out", default="results/exp10_affine.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
tok, L0, D = R.tok, a.layer, R.cfg.hidden_size
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
print(f"fit corpus: {len(stat_seqs)} train seqs; eval: {len(eval_seqs)} test seqs", flush=True)

# --------------------------------------------------------------- collect per-type statistics
uniq = torch.unique(torch.cat([s[0] for s in stat_seqs]))
remap = torch.full((R.cfg.vocab_size,), -1, dtype=torch.long, device=R.device)
remap[uniq] = torch.arange(uniq.numel(), device=R.device)
sums = torch.zeros(uniq.numel(), D, device=R.device)
cnts = torch.zeros(uniq.numel(), device=R.device)
sq_total = torch.zeros((), device=R.device)
n_tok = 0


class Grab:
    def __init__(self): self.value = None
    def register(self):
        def fn(mod, args, output):
            self.value = output.detach()
        return R.mixer(L0).register_forward_hook(fn)


with torch.no_grad():
    for si, s in enumerate(stat_seqs):
        g = Grab()
        with hook_ctx([g]):
            R.model(s, use_cache=False)
        o = g.value[0].float()
        idx = remap[s[0]]
        sums.index_add_(0, idx, o)
        cnts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
        sq_total += o.pow(2).sum()
        n_tok += o.shape[0]
        if (si + 1) % 50 == 0:
            print(f"  stats {si+1}/{a.n_stat}", flush=True)

M = sums / cnts[:, None].clamp(min=1)                 # per-type mean output  [T, D]
glob = sums.sum(0) / cnts.sum()
var_total = (sq_total - n_tok * glob.pow(2).sum()) / n_tok
var_between = (cnts[:, None] * (M - glob).pow(2)).sum() / n_tok
print(f"\n{uniq.numel()} token types, {n_tok} tokens")
print(f"R2_identity (between-type / total variance) = {(var_between/var_total).item():.4f}")

# --------------------------------------------------------------- fit A, b
E = EMB.weight[uniq].float()                          # [T, D]
X = torch.cat([E, torch.ones(E.shape[0], 1, device=R.device)], 1)     # [T, D+1]
w = cnts.clamp(min=1).sqrt()[:, None]


def fit(Xf, Yf, wf, ridge):
    Xw, Yw = Xf * wf, Yf * wf
    G = Xw.T @ Xw + ridge * torch.eye(Xf.shape[1], device=Xf.device)
    return torch.linalg.solve(G, Xw.T @ Yw)           # [D+1, D]


perm = torch.randperm(uniq.numel(), device=R.device)
tr_t, te_t = perm[: int(.8 * len(perm))], perm[int(.8 * len(perm)):]


def heldout_r2(ridge):
    Wt = fit(X[tr_t], M[tr_t], w[tr_t], ridge)
    res = (cnts[te_t][:, None] * (M[te_t] - X[te_t] @ Wt).pow(2)).sum()
    tot = (cnts[te_t][:, None] * (M[te_t] - glob).pow(2)).sum()
    return (1 - res / tot).item()


grid = [1e-2, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
r2s = {g: heldout_r2(g) for g in grid}
for g in grid:
    print(f"  ridge {g:>9.0e}  held-out-type R2 {r2s[g]:+.4f}", flush=True)
best_ridge = max(grid, key=lambda g: r2s[g])
r2_heldout = r2s[best_ridge]
print(f"selected ridge {best_ridge:.0e} (held-out-type R2 {r2_heldout:.4f})", flush=True)
W = fit(X, M, w, best_ridge)
A, b = W[:D], W[D]
pred = X @ W
rss_between = (cnts[:, None] * (M - pred).pow(2)).sum() / n_tok
r2_affine = 1 - ((rss_between + (var_total - var_between)) / var_total)
print(f"R2_affine   (A.embed + b vs total variance)  = {r2_affine.item():.4f}")
print(f"R2_affine|identity (of the token-wise part)  = "
      f"{(1 - rss_between/var_between).item():.4f}")

U, S, Vh = torch.linalg.svd(A, full_matrices=False)
print(f"A spectrum: s1={S[0]:.3f}  s64/s1={S[63]/S[0]:.4f}  s256/s1={S[255]/S[0]:.4f}")

# --------------------------------------------------------------- substitution + evaluation
CUR = {}


class Sub:
    """Replace mixer L0's output with A_r . embed(x) + b (or a stored surrogate)."""

    def __init__(self, mode, Ar=None, br=None):
        self.mode, self.Ar, self.br = mode, Ar, br

    def register(self):
        def fn(mod, args, output):
            ids = CUR["ids"]
            if self.mode == "affine":
                return (EMB.weight[ids].float() @ self.Ar + self.br).to(output.dtype)
            if self.mode == "token_mean":
                idx = remap[ids]
                return torch.where((idx >= 0)[..., None], M[idx.clamp(min=0)], glob).to(output.dtype)
            if self.mode == "token_mean_affine_fallback":
                idx = remap[ids]
                fb = EMB.weight[ids].float() @ self.Ar + self.br
                return torch.where((idx >= 0)[..., None], M[idx.clamp(min=0)], fb).to(output.dtype)
            if self.mode == "global_mean":
                return glob.expand_as(output).clone()
            raise ValueError(self.mode)
        return R.mixer(L0).register_forward_hook(fn)


@torch.no_grad()
def ppl(hooks=()):
    tot = 0.0
    for s in eval_seqs:
        CUR["ids"] = s
        with hook_ctx(hooks):
            lg = R.model(s, use_cache=False).logits
        tot += torch.nn.functional.cross_entropy(
            lg[:, :-1].float().reshape(-1, lg.shape[-1]), s[:, 1:].reshape(-1)).item()
    return tot / len(eval_seqs)


rows = []


def rec(name, nll, **kw):
    r = dict(name=name, nll=nll, ppl=float(torch.tensor(nll).exp()), **kw)
    rows.append(r)
    print(f"{name:42s} NLL {nll:7.4f}   PPL {r['ppl']:11.2f}", flush=True)
    return r


print("\n== perplexity ==", flush=True)
rec("intact", ppl())
cov = sum((remap[s_[0]] >= 0).float().mean().item() for s_ in eval_seqs) / len(eval_seqs)
print(f"  token-type table covers {cov:.4f} of eval tokens", flush=True)
rec("token-type mean (rung 3 ceiling)", ppl([Sub("token_mean")]), rung="token_mean", coverage=cov)
rec("token-type mean + affine fallback", ppl([Sub("token_mean_affine_fallback", A, b)]),
    rung="token_mean_fb", coverage=cov)
rec("global mean (no token info)", ppl([Sub("global_mean")]), rung="global_mean")
rec("affine  A.embed + b  (full rank)", ppl([Sub("affine", A, b)]), rung="affine", rank=D)
for r in [int(x) for x in a.ranks.split(",")]:
    if r >= D:
        continue
    Ar = (U[:, :r] * S[:r]) @ Vh[:r]
    rec(f"affine, rank {r}", ppl([Sub("affine", Ar, b)]), rung="affine", rank=r)

# --------------------------------------------------------------- retrieval task
print("\n== retrieval task ==", flush=True)
exs = []
for i, (ti, di) in enumerate([(2, 5), (5, 2), (1, 6), (6, 1)]):
    exs += make_examples(25, 8, seed=11 + i, target_pos=ti, distract_pos=di)
groups = group_examples(R, exs)
for name, hooks in (("intact", ()), ("affine (full rank)", [Sub("affine", A, b)]),
                    ("affine, rank 64", [Sub("affine", (U[:, :64] * S[:64]) @ Vh[:64], b)]),
                    ("global mean", [Sub("global_mean")])):
    ok = tot = 0
    gaps = []
    for g in groups:
        ds_ = []
        for ids, want in ((g.clean_ids, g.aid), (g.corr_ids, g.cid)):
            CUR["ids"] = ids
            with torch.no_grad(), hook_ctx(hooks):
                lg = R.model(ids, use_cache=False).logits[:, -1]
            ok += (lg.argmax(-1) == want).sum().item(); tot += ids.shape[0]
            ds_.append(g.D(lg))
        gaps += (ds_[0] - ds_[1]).tolist()
    rows.append(dict(name=f"retrieval {name}", acc=ok / tot, gap=sum(gaps) / len(gaps)))
    print(f"  L0 <- {name:20s} accuracy {ok/tot:.3f}   D_clean-D_corr {sum(gaps)/len(gaps):+.2f}",
          flush=True)

json.dump(dict(meta=vars(a), n_types=int(uniq.numel()), best_ridge=best_ridge,
               r2_identity=float(var_between / var_total), r2_affine=float(r2_affine),
               r2_affine_of_identity=float(1 - rss_between / var_between),
               r2_heldout_types=r2_heldout, singulars=S[:512].tolist(), rows=rows),
          open(a.out, "w"), indent=2)
