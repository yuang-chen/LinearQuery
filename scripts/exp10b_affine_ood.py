"""Step 10b: why does the affine re-embedding match WikiText but destroy the retrieval task?

Step 10 fitted o_0(t) ~= A.embed(x_t) + b on WikiText-2 train.  On WikiText it reaches the
context-free ceiling (PPL 90.8 vs 92.8 for a token-type lookup, intact 21.9), and it
generalises to token types held out of the fit (R2 0.84).  Yet substituting it destroys the
dictionary task (accuracy 1.000 -> 0.000), where an *in-distribution* token-type lookup was
perfect.  Two explanations:

  OOD    the chat-template / task tokens are off the WikiText embedding manifold, so the fitted
         map extrapolates badly there  -> refitting with task text in the fit should repair it
  NONLIN layer 0 computes something non-affine that retrieval specifically needs
         -> refitting will not repair it

Measures the affine residual per token group, then refits including held-out task prompts.
"""
import sys, json, argparse, torch
sys.path.insert(0, "/user/yac/LinearAblation")
from datasets import load_dataset
from src.runner import Runner, hook_ctx
from src.task import make_examples, CHAT_PRE, CHAT_POST
from src.harness import group_examples

ap = argparse.ArgumentParser()
ap.add_argument("--n_stat", type=int, default=2000)
ap.add_argument("--n_eval", type=int, default=16)
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--ridge", type=float, default=1e1)
ap.add_argument("--out", default="results/exp10b_affine_ood.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
tok, D = R.tok, R.cfg.hidden_size
EMB = (R.model.model.embed_tokens if hasattr(R.model.model, "embed_tokens")
       else R.model.model.language_model.embed_tokens)


class Grab:
    def __init__(self): self.value = None
    def register(self):
        def fn(mod, args, output):
            self.value = output.detach()
        return R.mixer(0).register_forward_hook(fn)


def outputs(batch):
    g = Grab()
    with torch.no_grad(), hook_ctx([g]):
        R.model(batch, use_cache=False)
    return g.value.reshape(-1, D).float(), batch.reshape(-1)


def corpus(split):
    d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split=split)
    return tok("\n\n".join(t for t in d["text"] if t.strip()), return_tensors="pt").input_ids[0]


tr_ids, te_ids = corpus("train"), corpus("test")
wiki_seqs = [tr_ids[i * a.seq_len:(i + 1) * a.seq_len].unsqueeze(0).to(R.device)
             for i in range(min(a.n_stat, tr_ids.numel() // a.seq_len))]
eval_seqs = [te_ids[i * a.seq_len:(i + 1) * a.seq_len].unsqueeze(0).to(R.device)
             for i in range(a.n_eval)]

# task prompts: a donor pool for fitting, and a disjoint pool for evaluation
donor = []
for i, (ti, di) in enumerate([(3, 6), (6, 3), (0, 4), (4, 0)]):
    donor += make_examples(25, 8, seed=901 + i, target_pos=ti, distract_pos=di)
donor_b = [R.tok([e.clean_prompt() for e in donor[i:i + 25]], return_tensors="pt").input_ids.to(R.device)
           for i in range(0, len(donor), 25)]
exs = []
for i, (ti, di) in enumerate([(2, 5), (5, 2), (1, 6), (6, 1)]):
    exs += make_examples(25, 8, seed=11 + i, target_pos=ti, distract_pos=di)
groups = group_examples(R, exs)


def accumulate(batches):
    """Per-token-type sums/counts of o_0 over a list of id batches."""
    ids_cat = torch.cat([b.reshape(-1) for b in batches])
    uq = torch.unique(ids_cat)
    rm = torch.full((R.cfg.vocab_size,), -1, dtype=torch.long, device=R.device)
    rm[uq] = torch.arange(uq.numel(), device=R.device)
    S = torch.zeros(uq.numel(), D, device=R.device)
    C = torch.zeros(uq.numel(), device=R.device)
    for i, b in enumerate(batches):
        o, ids = outputs(b)
        idx = rm[ids]
        S.index_add_(0, idx, o)
        C.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(batches)}", flush=True)
    return uq, rm, S, C


print("collecting WikiText statistics...", flush=True)
uq_w, rm_w, S_w, C_w = accumulate(wiki_seqs)
print("collecting task-prompt statistics...", flush=True)
uq_t, rm_t, S_t, C_t = accumulate(donor_b)


def fit(uq, S, C, ridge):
    M = S / C[:, None].clamp(min=1)
    E = EMB.weight[uq].float()
    X = torch.cat([E, torch.ones(E.shape[0], 1, device=R.device)], 1)
    w = C.clamp(min=1).sqrt()[:, None]
    Xw, Yw = X * w, M * w
    G = Xw.T @ Xw + ridge * torch.eye(X.shape[1], device=X.device)
    W = torch.linalg.solve(G, Xw.T @ Yw)
    return W[:D], W[D]


A_w, b_w = fit(uq_w, S_w, C_w, a.ridge)
# combined fit: WikiText types plus task types (task counts scaled to be visible)
uq_c = torch.unique(torch.cat([uq_w, uq_t]))
rm_c = torch.full((R.cfg.vocab_size,), -1, dtype=torch.long, device=R.device)
rm_c[uq_c] = torch.arange(uq_c.numel(), device=R.device)
S_c = torch.zeros(uq_c.numel(), D, device=R.device)
C_c = torch.zeros(uq_c.numel(), device=R.device)
S_c.index_add_(0, rm_c[uq_w], S_w); C_c.index_add_(0, rm_c[uq_w], C_w)
S_c.index_add_(0, rm_c[uq_t], S_t); C_c.index_add_(0, rm_c[uq_t], C_t)
A_c, b_c = fit(uq_c, S_c, C_c, a.ridge)

# --------------------------------------------------------------- residual diagnostics
SPECIAL = set(tok(CHAT_PRE).input_ids + tok(CHAT_POST).input_ids)
rows = []


def resid_report(name, batches, A, b):
    errs, errs_sp, errs_ord = [], [], []
    for bt in batches:
        o, ids = outputs(bt)
        pred = EMB.weight[ids].float() @ A + b
        rel = ((o - pred).norm(dim=-1) / o.norm(dim=-1).clamp(min=1e-6))
        errs.append(rel)
        sp = torch.tensor([int(i.item() in SPECIAL) for i in ids], device=R.device).bool()
        errs_sp.append(rel[sp]); errs_ord.append(rel[~sp])
    cat = lambda xs: torch.cat([x for x in xs if x.numel()])
    e, esp, eor = cat(errs), cat(errs_sp) if any(x.numel() for x in errs_sp) else None, cat(errs_ord)
    r = dict(name=name, rel_err_all=e.mean().item(), rel_err_ordinary=eor.mean().item(),
             rel_err_special=esp.mean().item() if esp is not None else None,
             frac_special=float(sum(x.numel() for x in errs_sp) / e.numel()))
    rows.append(r)
    print(f"  {name:38s} rel.err all {r['rel_err_all']:.3f}  ordinary {r['rel_err_ordinary']:.3f}"
          + (f"  special {r['rel_err_special']:.3f} ({r['frac_special']:.2%} of tokens)"
             if esp is not None else ""), flush=True)
    return r


print("\n== affine residual ||o - (A.emb+b)|| / ||o|| ==", flush=True)
resid_report("WikiText eval, WikiText-fit A", eval_seqs[:4], A_w, b_w)
task_b = [g.clean_ids for g in groups]
resid_report("task prompts, WikiText-fit A", task_b, A_w, b_w)
resid_report("task prompts, combined-fit A", task_b, A_c, b_c)

# --------------------------------------------------------------- substitution evaluation
CUR = {}


class Sub:
    def __init__(self, A, b): self.A, self.b = A, b
    def register(self):
        def fn(mod, args, output):
            return (EMB.weight[CUR["ids"]].float() @ self.A + self.b).to(output.dtype)
        return R.mixer(0).register_forward_hook(fn)


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


def retrieval(hooks=()):
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
    return ok / tot, sum(gaps) / len(gaps)


print("\n== substitution ==", flush=True)
for name, hooks in (("intact", ()), ("affine fitted on WikiText", [Sub(A_w, b_w)]),
                    ("affine fitted on WikiText + task", [Sub(A_c, b_c)])):
    p = ppl(hooks)
    acc, gap = retrieval(hooks)
    rows.append(dict(name=f"sub {name}", nll=p, ppl=float(torch.tensor(p).exp()),
                     acc=acc, gap=gap))
    print(f"  {name:34s} WikiText PPL {float(torch.tensor(p).exp()):8.2f}   "
          f"retrieval acc {acc:.3f}  gap {gap:+.2f}", flush=True)

json.dump(dict(meta=vars(a), rows=rows), open(a.out, "w"), indent=2)
