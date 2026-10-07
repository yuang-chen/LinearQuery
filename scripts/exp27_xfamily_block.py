"""Step 27 (paper exp 3B): transplant the query *block* across families, GnA Sec. 5.4 style.

GnA replace one layer of a distilled SSM with the teacher's attention layer, without fine-tuning,
and the capability that layer carries comes back. Across families the two residual streams live in
different spaces, so the transplanted block is wrapped in two linear maps fitted in closed form
(ridge) on intact runs of both models -- no gradient training anywhere:

  receiver R: layers lo..hi (its query block) are removed; in their place
      h_out = h_in + M_out( B_donor( M_in(h_in) ) - M_in(h_in) )
  M_in   receiver residual at the block input -> donor residual at its block input, fitted on
         tokens that cover the same characters in both tokenisations (dictionary prompts + text)
  B      the donor's block: its decoder layers run as they are, on the mapped sequence
  M_out  donor block write -> receiver block write (h_out - h_in of the intact receiver), fitted
         on every position of the fit prompts, with B running on M_in(h_in) as it will at test

Conditions (test prompts use keys never seen when fitting):
  intact                 receiver untouched
  removed                layers lo..hi skipped (h_out = h_in)
  mixers zeroed          receiver's own block with its linear mixers zeroed (Part XVIII setting)
  bypass                 h_out = h_in + M(h_in): the best position-wise linear stand-in (no mixing)
  donor=<Gi>             each donor block stitched in
  donor=<q>-nomix        the donor query block with its linear mixers zeroed (MLPs only)
Metrics: acc (full vocabulary), cacc (among the 8 values), reader attention on the target entry,
WikiText perplexity.
"""
import sys, json, argparse, time, random, os
import numpy as np
import torch
sys.path.insert(0, "/mnt/yuang/LinearQuery")
from src.runner import Runner, hook_ctx
from src.qkv import AttnWeights
from src.xfam import SPEC, VALUES, key_split, encode, align

ap = argparse.ArgumentParser()
ap.add_argument("--donor", default="0.8B")
ap.add_argument("--recv", default="G1b")
ap.add_argument("--variant", default="chat8")
ap.add_argument("--n_fit", type=int, default=1500)
ap.add_argument("--n_text", type=int, default=200, help="WikiText chunks added to the M_in fit")
ap.add_argument("--n_test", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args()
OUT = a.out or f"results/exp27_xblock_{a.donor}_into_{a.recv}_{a.variant}.json"
STYLE = {"list8": "list", "rev8": "rev"}.get(a.variant, "chat")
t0 = time.time()


def log(*s):
    print(f"[{(time.time() - t0) / 60:6.1f}m]", *s, flush=True)


D = Runner(model_path=SPEC[a.donor]["path"], dtype=torch.float32, device="cuda:0")
Y = Runner(model_path=SPEC[a.recv]["path"], dtype=torch.float32, device="cuda:0")
for M in (D, Y):
    M.model.config.get_text_config()._attn_implementation = "eager"
RB = SPEC[a.recv]["query"]; LO, HI = RB[0], RB[-1]
assert RB == list(range(LO, HI + 1)) and all(L in Y.gdn_layers for L in RB)
D_BLOCKS = [G for G in D.groups(SPEC[a.donor]["gs"]) if G]
DQ = SPEC[a.donor]["query"]
RL, RH = SPEC[a.recv]["reader"]
log(f"receiver {a.recv}: replace layers {LO}..{HI}, reader L{RL}H{RH} | donor {a.donor} blocks {D_BLOCKS}")

rng = random.Random(a.seed)
K_FIT, K_TEST = key_split(a.seed)


def make(keyset, n):
    out = []
    while len(out) < n:
        ks = rng.sample(keyset, 8); vs = rng.sample(VALUES, 8); ia = rng.randrange(8)
        pairs = list(zip(ks, vs))
        it = dict(pairs=pairs, ia=ia)
        it["y"], it["yent"], cy = encode(Y.tok, pairs, ks[ia], STYLE)
        it["d"], _, cd = encode(D.tok, pairs, ks[ia], STYLE)
        it["al"] = align(cy, cd)
        aid = Y.tok(vs[ia]).input_ids
        if len(aid) == 1:
            it["aid"] = aid[0]
            it["cand"] = [Y.tok(v).input_ids[0] for v in vs]
            out.append(it)
    return out


def text_items(n, L=128):
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
    docs = [t for t in d["text"] if len(t) > 700][:n]
    out = []
    for t in docs:
        t = t[:600]
        ey = Y.tok(t, return_offsets_mapping=True); ed = D.tok(t, return_offsets_mapping=True)
        cy = [(0, x, y) if y > x else None for x, y in ey.offset_mapping]
        cd = [(0, x, y) if y > x else None for x, y in ed.offset_mapping]
        out.append(dict(y=ey.input_ids, d=ed.input_ids, al=align(cy, cd)))
    return out


# ------------------------------------------------------------------ hooks
class InCap:
    """residual stream entering layer L (all positions)"""
    def __init__(self, M, L): self.M, self.L, self.v = M, L, None
    def register(self):
        def fn(mod, args, kwargs):
            self.v = (kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]).detach().float()
        return self.M.layers[self.L].register_forward_pre_hook(fn, with_kwargs=True)


class OutCap:
    def __init__(self, M, L): self.M, self.L, self.v = M, L, None
    def register(self):
        def fn(mod, args, out):
            self.v = (out[0] if isinstance(out, tuple) else out).detach().float()
        return self.M.layers[self.L].register_forward_hook(fn)


class Replace:
    """Skip layers LO..HI-1 and make layer HI output h_in + g(h_in)."""
    def __init__(self, M, lo, hi, g): self.M, self.lo, self.hi, self.g = M, lo, hi, g
    def register(self):
        hs = []
        for L in range(self.lo, self.hi + 1):
            def fn(mod, args, kwargs, out, L=L):
                h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                if L < self.hi:
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                new = h + self.g(h.float()).to(h.dtype)
                return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
            hs.append(self.M.layers[L].register_forward_hook(fn, with_kwargs=True))

        class H:
            def remove(self_):
                for x in hs:
                    x.remove()
        return H()


class ZeroMix:
    def __init__(self, M, L): self.M, self.L = M, L
    def register(self):
        return self.M.mixer(self.L).register_forward_hook(
            lambda m, a_, o: (torch.zeros_like(o[0]),) + tuple(o[1:]) if isinstance(o, tuple)
            else torch.zeros_like(o))


@torch.no_grad()
def run_donor_block(block, h, nomix=False):
    """The donor's decoder layers `block` applied to hidden states h [1, T, H_donor]."""
    hooks = [ZeroMix(D, L) for L in block] if nomix else []
    with hook_ctx(hooks):
        x = h.to(next(D.model.parameters()).dtype)
        for L in block:
            o = D.layers[L](x, position_embeddings=None)
            x = o[0] if isinstance(o, tuple) else o
    return x.float()


# ------------------------------------------------------------------ ridge
class Ridge:
    def __init__(self, X, T, n_val=None):
        n = len(X); perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        n_val = n_val or min(4000, n // 5)
        vi, ti = perm[:n_val], perm[n_val:]
        X = X.double().cuda(); T = T.double().cuda()
        self.mx, self.mt = X[ti].mean(0), T[ti].mean(0)
        Xc, Tc = X[ti] - self.mx, T[ti] - self.mt
        G = Xc.T @ Xc; C = Xc.T @ Tc
        ev, V = torch.linalg.eigh(G)
        VC = V.T @ C
        Xv, Tv = X[vi] - self.mx, T[vi] - self.mt
        best = None
        for p_ in np.arange(-6, 2.5, 0.5):
            lam = 10 ** p_ * float(ev[-1])
            W = V @ (VC / (ev + lam)[:, None])
            err = float(((Xv @ W - Tv) ** 2).sum() / (Tv ** 2).sum())
            if best is None or err < best[0]:
                best = (err, W)
        self.W, self.r2 = best[1], 1 - best[0]

    def __call__(self, X):
        return ((X.double() - self.mx) @ self.W + self.mt).float()


# ------------------------------------------------------------------ fit data
fit = make(K_FIT, a.n_fit)
txt = text_items(a.n_text)
log(f"fit: {len(fit)} prompts (K_fit {len(K_FIT)} keys) + {len(txt)} text chunks; "
    f"aligned tokens/prompt ~{np.mean([len(i['al']) for i in fit]):.0f} of {np.mean([len(i['y']) for i in fit]):.0f}")


@torch.no_grad()
def fwd_caps(M, ids, caps):
    with hook_ctx(caps):
        M.model(torch.tensor([ids], device=M.device), use_cache=False)


# receiver block input / output, donor block inputs, for every fit item
Yin, Yd = [], []
Din = {i: [] for i in range(len(D_BLOCKS))}
for it in fit + txt:
    ci, co = InCap(Y, LO), OutCap(Y, HI)
    fwd_caps(Y, it["y"], [ci, co])
    it["hy"] = ci.v[0].cpu(); Yd.append((co.v[0] - ci.v[0]).cpu())
    caps = [InCap(D, G[0]) for G in D_BLOCKS]
    fwd_caps(D, it["d"], caps)
    it["hd"] = [c.v[0].cpu() for c in caps]
log("captured intact activations")
ALL = fit + txt
Xy_all = torch.cat([it["hy"] for it in ALL]); Yd_all = torch.cat(Yd)
bypass = Ridge(Xy_all, Yd_all)
log(f"bypass map R^2 {bypass.r2:.3f}")

stitch = {}
for bi, G in enumerate(D_BLOCKS):
    Xa = torch.cat([it["hy"][[p for p, _ in it["al"]]] for it in ALL])
    Ta = torch.cat([it["hd"][bi][[q for _, q in it["al"]]] for it in ALL])
    m_in = Ridge(Xa, Ta)
    for nomix in ((False, True) if G == DQ else (False,)):
        dw = []
        for it in ALL:
            h = m_in(it["hy"].cuda())[None]
            dw.append((run_donor_block(G, h, nomix)[0] - h[0]).cpu())
        m_out = Ridge(torch.cat(dw), Yd_all)
        name = f"G{bi}" + ("-nomix" if nomix else "") + ("*" if G == DQ else "")
        stitch[name] = (G, nomix, m_in, m_out)
        log(f"donor {name:9s} layers {G}: M_in R^2 {m_in.r2:.3f}  M_out R^2 {m_out.r2:.3f}")
for it in ALL:
    it.pop("hy"); it.pop("hd")

# ------------------------------------------------------------------ test
test = make(K_TEST, a.n_test)
bk = {}
for it in test:
    bk.setdefault((len(it["y"]), str(it["yent"])), []).append(it)
TB = [v[i:i + 25] for v in bk.values() for i in range(0, len(v), 25)]
from datasets import load_dataset
wt = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
wids = Y.tok("\n\n".join(t for t in wt["text"] if t.strip()), return_tensors="pt").input_ids[0]
WSEQ = torch.stack([wids[i * 512:(i + 1) * 512] for i in range(8)]).to(Y.device)


@torch.no_grad()
def evaluate(hooks):
    ok = cok = n = 0; att = []
    for bt in TB:
        ids = torch.tensor([it["y"] for it in bt], device=Y.device)
        aw = AttnWeights(Y.mixer(RL))
        with hook_ctx(list(hooks) + [aw]):
            lg = Y.model(ids, use_cache=False).logits[:, -1].float()
        aid = torch.tensor([it["aid"] for it in bt], device=lg.device)
        cand = torch.tensor([it["cand"] for it in bt], device=lg.device)
        ok += (lg.argmax(-1) == aid).sum().item()
        cok += (cand.gather(1, lg.gather(1, cand).argmax(-1, keepdim=True))[:, 0] == aid).sum().item()
        n += len(bt)
        at = aw.value[:, RH, -1].float()
        att += [float(at[j, bt[0]["yent"][it["ia"]]].sum()) for j, it in enumerate(bt)]
    nll = []
    with hook_ctx(list(hooks)):
        for i in range(0, len(WSEQ), 4):
            x = WSEQ[i:i + 4]
            lg = Y.model(x, use_cache=False).logits[:, :-1].float()
            nll.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                                         x[:, 1:].reshape(-1), reduction="none"))
    return dict(acc=ok / n, cacc=cok / n, att=float(np.mean(att)),
                ppl=float(torch.exp(torch.cat(nll).mean())))


def g_stitch(G, nomix, m_in, m_out):
    def g(h):
        out = []
        for b in range(h.shape[0]):
            x = m_in(h[b])[None]
            out.append(m_out(run_donor_block(G, x, nomix)[0] - x[0]))
        return torch.stack(out)
    return g


res = dict(meta=dict(donor=a.donor, recv=a.recv, variant=a.variant, recv_block=[LO, HI],
                     donor_blocks=D_BLOCKS, donor_query=DQ, reader=[RL, RH], n_fit=len(fit),
                     n_text=len(txt), n_test=len(test), k_fit=K_FIT, k_test=K_TEST,
                     bypass_r2=bypass.r2,
                     stitch_r2={k: dict(m_in=v[2].r2, m_out=v[3].r2) for k, v in stitch.items()}),
           conditions={})
C = res["conditions"]
C["intact"] = evaluate([])
C["removed"] = evaluate([Replace(Y, LO, HI, lambda h: torch.zeros_like(h))])
C["mixers zeroed"] = evaluate([ZeroMix(Y, L) for L in RB])
C["bypass"] = evaluate([Replace(Y, LO, HI, bypass)])
for name, (G, nomix, m_in, m_out) in stitch.items():
    C[f"donor={name}"] = evaluate([Replace(Y, LO, HI, g_stitch(G, nomix, m_in, m_out))])
    log(f"donor={name}: {C[f'donor={name}']}")
for k, c in C.items():
    log(f"{k:22s} acc {c['acc']:.3f}  cacc {c['cacc']:.3f}  reader attn {c['att']:.2f}  ppl {c['ppl']:.1f}")
os.makedirs("results", exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
log(f"wrote {OUT}")
