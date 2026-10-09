"""Step 34: block stitching between the larger models (Qwen3.5-9B <-> Granite4.0-7B), where exp27's
single-slot, full-depth design is not testable (9B's later readers compensate for a missing Address
block; Granite4.0-7B's second Address block compensates for a missing first one).

Same stitch as exp27, generalised:
  * several receiver slots (--slots 16-24,26-34): each removed block is replaced by
        h_out = h_in + M_out( B_donor( M_in(h_in) ) - M_in(h_in) )
    with its own maps; slot k's maps are fitted with slots < k already replaced in the same way,
    so every map sees at test time the input distribution it was fitted on (as in exp28)
  * truncation (--cut L): layers after L are skipped and the hidden state goes to the final norm
    (the GnA-style minimal model of Section 3); evaluated alongside full depth
  * reader-level metrics: reader attention on the target and the candidate margin (answer logit
    minus the mean logit of the other dictionary values), in addition to accuracy

Conditions (each applied to every slot): intact, removed, mixers zeroed, per-token linear map
(h_in + M h_in, fitted in the receiver), donor block <Gi>, donor Address block with mixers zeroed.
Accuracy: full-vocabulary top-1 (acc) and choice among the 8 dictionary values (cacc).
"""
import sys, json, argparse, time, random, os
import numpy as np
import torch
sys.path.insert(0, "/mnt/yuang/LinearQuery")
from src.runner import Runner, hook_ctx
from src.qkv import AttnWeights
from src.xfam import SPEC, VALUES, key_split, encode, align

ap = argparse.ArgumentParser()
ap.add_argument("--donor", default="9B")
ap.add_argument("--recv", default="Gtiny")
ap.add_argument("--slots", default="16-24,26-34", help="receiver blocks to replace, e.g. 16-24,26-34")
ap.add_argument("--donor_blocks", default="", help="donor block indices to stitch (default: the donor's Address block)")
ap.add_argument("--reader", default="", help="receiver reader head 'L,H' (default from SPEC)")
ap.add_argument("--donor_custom", default="", help="extra donor units, ';'-separated: 'seq:16-24+26-34' runs the "
                "listed layers in order on one mapped input (layers not listed, e.g. a softmax layer in between, are "
                "skipped); 'par:16-24|26-34' runs each block on its own mapped input and maps their concatenated writes")
ap.add_argument("--cut", type=int, default=-1, help="also evaluate the receiver truncated after this layer")
ap.add_argument("--variant", default="chat8")
ap.add_argument("--n_fit", type=int, default=1500)
ap.add_argument("--n_text", type=int, default=200)
ap.add_argument("--n_test", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args()
OUT = a.out or f"results/exp34_xblock_{a.donor}_into_{a.recv}_{a.variant}.json"
STYLE = {"list8": "list", "rev8": "rev"}.get(a.variant, "chat")
t0 = time.time()
log = lambda *s: print(f"[{(time.time() - t0) / 60:6.1f}m]", *s, flush=True)

D = Runner(model_path=SPEC[a.donor]["path"], dtype=torch.float32, device="cuda:0")
Y = Runner(model_path=SPEC[a.recv]["path"], dtype=torch.float32, device="cuda:0")
for M in (D, Y):
    M.model.config.get_text_config()._attn_implementation = "eager"
SLOTS = [(int(p.split("-")[0]), int(p.split("-")[1])) for p in a.slots.split(",")]
for lo, hi in SLOTS:
    assert all(L in Y.gdn_layers for L in range(lo, hi + 1)), (lo, hi)
D_ALLB = [G for G in D.groups(SPEC[a.donor]["gs"]) if G]
DQ = SPEC[a.donor]["query"]
DQI = D_ALLB.index(DQ)
DB = [int(i) for i in a.donor_blocks.split(",")] if a.donor_blocks else [DQI]
RL, RH = map(int, a.reader.split(",")) if a.reader else SPEC[a.recv]["reader"]
rng_ = lambda t: list(range(int(t.split("-")[0]), int(t.split("-")[-1]) + 1))
# donor units: name -> (mode, [layer lists]); a standard block is a 'seq' unit with one layer list
UNITS = {}
for bi in DB:
    UNITS[f"donor G{bi}" + (" (Address)" if bi == DQI else "")] = ("seq", [D_ALLB[bi]], bi == DQI)
for spec in filter(None, a.donor_custom.split(";")):
    mode, body = spec.split(":")
    if mode == "seq":
        UNITS[f"donor seq {body}"] = ("seq", [[L for part in body.split("+") for L in rng_(part)]], True)
    else:
        UNITS[f"donor par {body}"] = ("par", [rng_(part) for part in body.split("|")], True)
IN_LAYERS = sorted({u[0] for _, us, _ in UNITS.values() for u in us})
log(f"receiver {a.recv}: slots {SLOTS}, reader L{RL}H{RH}, cut {a.cut} | donor {a.donor} blocks "
    f"{[(i, D_ALLB[i]) for i in DB]} (Address G{DQI})")

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
            it["aid"] = aid[0]; it["cand"] = [Y.tok(v).input_ids[0] for v in vs]
            out.append(it)
    return out


def text_items(n):
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
    out = []
    for t in [t for t in d["text"] if len(t) > 700][:n]:
        t = t[:600]
        ey, ed = Y.tok(t, return_offsets_mapping=True), D.tok(t, return_offsets_mapping=True)
        cy = [(0, x, y) if y > x else None for x, y in ey.offset_mapping]
        cd = [(0, x, y) if y > x else None for x, y in ed.offset_mapping]
        out.append(dict(y=ey.input_ids, d=ed.input_ids, al=align(cy, cd)))
    return out


# ------------------------------------------------------------------ hooks
class InCap:
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
    """Skip layers lo..hi-1 and make layer hi output h_in + g(h_in) (g=None: h_out = h_in)."""
    def __init__(self, M, lo, hi, g): self.M, self.lo, self.hi, self.g = M, lo, hi, g
    def register(self):
        hs = []
        for L in range(self.lo, self.hi + 1):
            def fn(mod, args, kwargs, out, L=L):
                h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                new = h if (L < self.hi or self.g is None) else h + self.g(h.float()).to(h.dtype)
                return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
            hs.append(self.M.layers[L].register_forward_hook(fn, with_kwargs=True))

        class H:
            def remove(self_):
                for x in hs:
                    x.remove()
        return H()


class Cut:
    """Skip every layer after `cut` (hidden state passes through to the final norm)."""
    def __init__(self, M, cut): self.M, self.cut = M, cut
    def register(self):
        hs = []
        for L in range(self.cut + 1, self.M.n_layers):
            def fn(mod, args, kwargs, out):
                h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
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
            lambda m, a_, o: (torch.zeros_like(o[0]),) + tuple(o[1:]) if isinstance(o, tuple) else torch.zeros_like(o))


@torch.no_grad()
def run_donor_block(block, h, nomix=False):
    hooks = [ZeroMix(D, L) for L in block] if nomix else []
    with hook_ctx(hooks):
        x = h.to(next(D.model.parameters()).dtype)
        for L in block:
            o = D.layers[L](x, position_embeddings=None)
            x = o[0] if isinstance(o, tuple) else o
    return x.float()


class Ridge:
    def __init__(self, X, T):
        n = len(X); perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        n_val = min(4000, n // 5); vi, ti = perm[:n_val], perm[n_val:]
        X = X.double().cuda(); T = T.double().cuda()
        self.mx, self.mt = X[ti].mean(0), T[ti].mean(0)
        Xc, Tc = X[ti] - self.mx, T[ti] - self.mt
        ev, V = torch.linalg.eigh(Xc.T @ Xc); VC = V.T @ (Xc.T @ Tc)
        Xv, Tv = X[vi] - self.mx, T[vi] - self.mt
        best = None
        for p_ in np.arange(-6, 2.5, 0.5):
            W = V @ (VC / (ev + 10 ** p_ * float(ev[-1]))[:, None])
            err = float(((Xv @ W - Tv) ** 2).sum() / (Tv ** 2).sum())
            if best is None or err < best[0]:
                best = (err, W)
        self.W, self.r2 = best[1], 1 - best[0]

    def __call__(self, X):
        return ((X.double() - self.mx) @ self.W + self.mt).float()


def unit_writes(units, m_ins, x, nomix):
    """donor writes of every unit on its own mapped input, concatenated: [T, n_units * H_donor]"""
    ws = []
    for u, m_in in zip(units, m_ins):
        h = m_in(x)[None]
        ws.append(run_donor_block(u, h, nomix)[0] - h[0])
    return torch.cat(ws, -1)


def g_stitch(units, nomix, m_ins, m_out):
    def g(h):
        return torch.stack([m_out(unit_writes(units, m_ins, h[b], nomix)) for b in range(h.shape[0])])
    return g


# ------------------------------------------------------------------ fit
fit = make(K_FIT, a.n_fit); txt = text_items(a.n_text); ALL = fit + txt
log(f"fit: {len(fit)} prompts + {len(txt)} text chunks")


@torch.no_grad()
def caps_on(M, ids, caps, hooks=()):
    with hook_ctx(list(hooks) + caps):
        M.model(torch.tensor([ids], device=M.device), use_cache=False)


# intact receiver block writes (targets) and donor block inputs (for M_in)
for it in ALL:
    co = [(InCap(Y, lo), OutCap(Y, hi)) for lo, hi in SLOTS]
    caps_on(Y, it["y"], [c for pair in co for c in pair])
    it["target"] = [(o.v[0] - i.v[0]).cpu() for i, o in co]
    dc = [InCap(D, L) for L in IN_LAYERS]
    caps_on(D, it["d"], dc)
    it["hd"] = {L: c.v[0].cpu() for L, c in zip(IN_LAYERS, dc)}
log("captured intact activations")


def fit_condition(kind, units=None, nomix=False):
    """Fit one replacement function per slot, slot k fitted with slots < k already replaced."""
    gs, r2 = [], []
    for k, (lo, hi) in enumerate(SLOTS):
        prev = [Replace(Y, SLOTS[j][0], SLOTS[j][1], gs[j]) for j in range(k)]
        Xin = []
        for it in ALL:
            c = InCap(Y, lo); caps_on(Y, it["y"], [c], prev)
            Xin.append(c.v[0].cpu())
        T = torch.cat([it["target"][k] for it in ALL])
        if kind == "bypass":
            m = Ridge(torch.cat(Xin), T); gs.append(m); r2.append(m.r2); continue
        Xa = torch.cat([x[[p for p, _ in it["al"]]] for x, it in zip(Xin, ALL)])
        m_ins = [Ridge(Xa, torch.cat([it["hd"][u[0]][[q for _, q in it["al"]]] for it in ALL])) for u in units]
        dw = [unit_writes(units, m_ins, x.cuda(), nomix).cpu() for x in Xin]
        m_out = Ridge(torch.cat(dw), T)
        gs.append(g_stitch(units, nomix, m_ins, m_out)); r2.append(([m.r2 for m in m_ins], m_out.r2))
    return gs, r2


COND = {"intact": None, "removed": "removed", "mixers zeroed": "zeromix"}
fitted, R2 = {}, {}
fitted["per-token linear map"], R2["per-token linear map"] = fit_condition("bypass")
log(f"per-token map fitted, R^2 {R2['per-token linear map']}")
for uname, (mode, units, with_nomix) in UNITS.items():
    for nomix in ((False, True) if with_nomix else (False,)):
        name = uname + (", mixers zeroed" if nomix else "")
        fitted[name], R2[name] = fit_condition("donor", units, nomix)
        log(f"{name} fitted, R^2 (M_in, M_out) per slot {R2[name]}")
for it in ALL:
    it.pop("target"); it.pop("hd")

# ------------------------------------------------------------------ test
test = make(K_TEST, a.n_test)
bk = {}
for it in test:
    bk.setdefault((len(it["y"]), str(it["yent"])), []).append(it)
TB = [v[i:i + 25] for v in bk.values() for i in range(0, len(v), 25)]


def hooks_for(name):
    if name == "intact":
        return []
    if name == "removed":
        return [Replace(Y, lo, hi, None) for lo, hi in SLOTS]
    if name == "mixers zeroed":
        return [ZeroMix(Y, L) for lo, hi in SLOTS for L in range(lo, hi + 1)]
    return [Replace(Y, lo, hi, g) for (lo, hi), g in zip(SLOTS, fitted[name])]


@torch.no_grad()
def evaluate(hooks):
    ok = cok = n = 0; att, mar = [], []
    for bt in TB:
        ids = torch.tensor([it["y"] for it in bt], device=Y.device)
        aw = AttnWeights(Y.mixer(RL))
        with hook_ctx(list(hooks) + [aw]):
            lg = Y.model(ids, use_cache=False).logits[:, -1].float()
        aid = torch.tensor([it["aid"] for it in bt], device=lg.device)
        cand = torch.tensor([it["cand"] for it in bt], device=lg.device)
        cl = lg.gather(1, cand)
        ok += (lg.argmax(-1) == aid).sum().item()
        cok += (cand.gather(1, cl.argmax(-1, keepdim=True))[:, 0] == aid).sum().item()
        n += len(bt)
        for j, it in enumerate(bt):
            al = float(lg[j, it["aid"]]); others = [float(cl[j, k]) for k in range(8) if it["cand"][k] != it["aid"]]
            mar.append(al - float(np.mean(others)))
        at = aw.value[:, RH, -1].float()
        att += [float(at[j, bt[0]["yent"][it["ia"]]].sum()) for j, it in enumerate(bt)]
    return dict(acc=ok / n, cacc=cok / n, att=float(np.mean(att)), margin=float(np.mean(mar)))


names = list(COND) + list(fitted)
res = dict(meta=dict(donor=a.donor, recv=a.recv, variant=a.variant, slots=SLOTS, donor_units={k: v[:2] for k, v in UNITS.items()},
                     donor_address=DQI, reader=[RL, RH], cut=a.cut, n_fit=len(fit), n_text=len(txt),
                     n_test=len(test), r2={k: v for k, v in R2.items()}), full={}, cut={})
for name in names:
    res["full"][name] = evaluate(hooks_for(name))
    line = f"{name:38s} full: acc {res['full'][name]['acc']:.2f} cacc {res['full'][name]['cacc']:.2f} " \
           f"attn {res['full'][name]['att']:.2f} margin {res['full'][name]['margin']:+.2f}"
    if a.cut >= 0:
        res["cut"][name] = evaluate(hooks_for(name) + [Cut(Y, a.cut)])
        line += f" | cut after L{a.cut}: cacc {res['cut'][name]['cacc']:.2f} attn {res['cut'][name]['att']:.2f} " \
                f"margin {res['cut'][name]['margin']:+.2f}"
    log(line)
    json.dump(res, open(OUT, "w"), indent=1)
log(f"wrote {OUT}")
