"""Step 29 (paper exp 2/3, Bind side): is the Bind block's write at a dictionary token sufficient to
re-bind an entry -- within a model, and across families?

Two token-level swaps, each with a donor and a host prompt of identical token length:

  value (content)  host: entry t's value swapped with entry d's (so t holds v_d); donor: unswapped.
                   Patch the Bind block's write at t's *value* token. Success = host answers v_t
                   (the donor's value) to the question about key t.
  key (address)    host: unswapped, asks key t. donor: the keys of entries t and d exchanged.
                   Patch the Bind block's write at the *key* tokens of t and d. Success = host
                   answers v_d, i.e. the key identity the reader matches against moved with the patch.

Sources written into the host's Bind block at those positions:
  in-family      the host model's own Bind writes from the donor prompt
  in-family emb  control: the donor prompt's token *embedding* only (input to layer 0), Bind recomputed
  in-family Gi   control: another block's writes at those positions (Bind left intact)
  xfam           the other family's Bind writes, mapped by ridge (fit on aligned tokens of fit
                 dictionaries + WikiText; test keys *and* values never seen in the fit)
  xfam-emb       the other family's static token embedding, mapped by ridge
  xfam-shuffled  map fitted on shuffled rows;  mean: host Bind writes replaced by their mean
"""
import sys, json, argparse, time, random, os
import numpy as np
import torch
sys.path.insert(0, "/mnt/yuang/LinearQuery")
from src.runner import Runner, hook_ctx
from src.xfam import SPEC, VALUES, KEYS, encode, align, render
from src.gen import chat_wrap
from src.stitch import Ridge

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="G1b")
ap.add_argument("--donor", default="0.8B", help="other-family model for the cross-family rows ('' = none)")
ap.add_argument("--variant", default="chat8", help="chat8 | list8")
ap.add_argument("--n_fit", type=int, default=1500)
ap.add_argument("--n_text", type=int, default=200)
ap.add_argument("--n_test", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args()
OUT = a.out or f"results/exp29_bind_{a.donor or 'none'}_into_{a.host}_{a.variant}.json"
STYLE = {"list8": "list"}.get(a.variant, "chat")
t0 = time.time()


def log(*s):
    print(f"[{(time.time() - t0) / 60:6.1f}m]", *s, flush=True)


Y = Runner(model_path=SPEC[a.host]["path"], dtype=torch.float32, device="cuda:0")
D = Runner(model_path=SPEC[a.donor]["path"], dtype=torch.float32, device="cuda:0") if a.donor else None
Y_BLOCKS = [G for G in Y.groups(SPEC[a.host]["gs"]) if G]
YB = Y_BLOCKS[0]
D_BLOCKS = [G for G in D.groups(SPEC[a.donor]["gs"]) if G] if D else []
DB = D_BLOCKS[0] if D else None
HY = Y.cfg.hidden_size
log(f"host {a.host} Bind {YB} (other blocks {Y_BLOCKS[1:]}) | donor {a.donor} Bind {DB}")

rng = random.Random(a.seed)
keys = [k for k in KEYS if k not in VALUES]; rng.shuffle(keys)
vals = list(VALUES); rng.shuffle(vals)
K_FIT, K_TEST = keys[:len(keys) // 2], keys[len(keys) // 2:]
V_FIT, V_TEST = vals[:len(vals) // 2], vals[len(vals) // 2:]


def spans(M, pairs, q):
    """ids, per-entry key-token position and value-token position (chat/list templates)."""
    s, es, _ = render(chat_wrap(M.tok), pairs, q, STYLE)
    e = M.tok(s, return_offsets_mapping=True)
    tokpos = lambda a_, b_: [i for i, (x, y) in enumerate(e.offset_mapping) if x < b_ and y > a_]
    kp, vp = [], []
    for (e0, e1), (k, v) in zip(es, pairs):
        seg = s[e0:e1]
        k0 = e0 + seg.index(k); v0 = e0 + seg.rindex(v)
        kp.append(tokpos(k0, k0 + len(k))); vp.append(tokpos(v0, v0 + len(v)))
    if any(len(x) != 1 for x in kp + vp):
        return None
    return e.input_ids, [x[0] for x in kp], [x[0] for x in vp]


def make_test(n):
    """(host, donor-prompt) pairs for both swaps, on K_TEST keys and V_TEST values."""
    out = []
    cfgs = [(2, 5), (5, 2), (1, 6), (6, 1)]
    while len(out) < n:
        t, d = cfgs[len(out) % 4]
        ks = rng.sample(K_TEST, 8); vs = rng.sample(V_TEST, 8)
        base = list(zip(ks, vs))
        vsw = list(base); vsw[t] = (ks[t], vs[d]); vsw[d] = (ks[d], vs[t])     # value-swapped
        ksw = list(base); ksw[t] = (ks[d], vs[t]); ksw[d] = (ks[t], vs[d])     # key-swapped
        it = dict(t=t, d=d, vt=vs[t], vd=vs[d])
        ok = True
        for tag, M in (("y", Y), ("x", D)):
            if M is None:
                continue
            A = spans(M, base, ks[t]); B = spans(M, vsw, ks[t]); K = spans(M, ksw, ks[t])
            if not (A and B and K) or not (len(A[0]) == len(B[0]) == len(K[0])):
                ok = False; break
            it[tag] = dict(base=A, vswap=B, kswap=K)
        aid = [Y.tok(v).input_ids for v in (vs[t], vs[d])]
        if ok and all(len(x) == 1 for x in aid):
            it["id_t"], it["id_d"] = aid[0][0], aid[1][0]
            out.append(it)
    return out


# ------------------------------------------------------------------ captures
class Writes:
    """capture each layer's mixer output of `block` (all positions)"""
    def __init__(self, M, block): self.M, self.block, self.v = M, block, {}
    def register(self):
        hs = []
        for L in self.block:
            def fn(m, a_, o, L=L):
                self.v[L] = (o[0] if isinstance(o, tuple) else o).detach().float()
            hs.append(self.M.mixer(L).register_forward_hook(fn))

        class H:
            def remove(s_):
                for h in hs:
                    h.remove()
        return H()


class SetWrites:
    """overwrite each layer's mixer output of `block` at `pos` with vec[:, pos, i*H:(i+1)*H]"""
    def __init__(self, M, block, pos, vec, H): self.M, self.block, self.pos, self.vec, self.H = M, block, pos, vec, H
    def register(self):
        hs = []
        for i, L in enumerate(self.block):
            def fn(m, a_, o, i=i):
                t = (o[0] if isinstance(o, tuple) else o).clone()
                for p in self.pos:
                    t[:, p] = self.vec[:, p, i * self.H:(i + 1) * self.H].to(t.dtype).to(t.device)
                return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
            hs.append(self.M.mixer(L).register_forward_hook(fn))

        class H_:
            def remove(s_):
                for h in hs:
                    h.remove()
        return H_()


class SetEmb:
    """overwrite the input embeddings at `pos` with the donor prompt's embeddings"""
    def __init__(self, M, pos, emb): self.M, self.pos, self.emb = M, pos, emb
    def register(self):
        def fn(m, a_, o):
            t = o.clone()
            for p in self.pos:
                t[:, p] = self.emb[:, p].to(t.dtype)
            return t
        return self.M.model.get_input_embeddings().register_forward_hook(fn)


@torch.no_grad()
def block_writes(M, ids, block):
    w = Writes(M, block)
    with hook_ctx([w]):
        M.model(torch.tensor(ids, device=M.device), use_cache=False)
    return torch.cat([w.v[L] for L in block], -1)          # [B, T, len(block)*H]


# ------------------------------------------------------------------ cross-family map (Bind writes, token level)
maps = {}
if D:
    def fit_items(n):
        out = []
        while len(out) < n:
            ks = rng.sample(K_FIT, 8); vs = rng.sample(V_FIT, 8); q = rng.choice(ks)
            pairs = list(zip(ks, vs))
            iy, _, cy = encode(Y.tok, pairs, q, STYLE); idd, _, cd = encode(D.tok, pairs, q, STYLE)
            out.append(dict(y=iy, d=idd, al=align(cy, cd)))
        return out
    from datasets import load_dataset
    wtr = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
    items = fit_items(a.n_fit)
    for t in [t for t in wtr["text"] if len(t) > 700][:a.n_text]:
        t = t[:600]
        ey = Y.tok(t, return_offsets_mapping=True); ed = D.tok(t, return_offsets_mapping=True)
        cy = [(0, x, y) if y > x else None for x, y in ey.offset_mapping]
        cd = [(0, x, y) if y > x else None for x, y in ed.offset_mapping]
        items.append(dict(y=ey.input_ids, d=ed.input_ids, al=align(cy, cd)))
    EMB_D = D.model.get_input_embeddings()
    X, Xe, T = [], [], []
    for it in items:
        if not it["al"]:
            continue
        py = [p for p, _ in it["al"]]; pd = [q for _, q in it["al"]]
        T.append(block_writes(Y, [it["y"]], YB)[0, py].cpu())
        X.append(block_writes(D, [it["d"]], DB)[0, pd].cpu())
        with torch.no_grad():
            Xe.append(EMB_D(torch.tensor([it["d"][q] for q in pd], device=D.device)).float().cpu())
    X, Xe, T = torch.cat(X), torch.cat(Xe), torch.cat(T)
    perm = torch.randperm(len(T), generator=torch.Generator().manual_seed(1))
    maps["xfam"] = Ridge(X, T); maps["xfam-emb"] = Ridge(Xe, T); maps["xfam-shuffled"] = Ridge(X, T[perm])
    log(f"cross-family Bind maps on {len(T)} aligned tokens: " +
        "  ".join(f"{k} R^2 {v.r2:.3f}" for k, v in maps.items()))
    TMEAN = T.mean(0)

# ------------------------------------------------------------------ test
test = make_test(a.n_test)
res = dict(meta=dict(host=a.host, donor=a.donor, variant=a.variant, host_bind=YB, donor_bind=DB,
                     n_test=len(test), k_test=K_TEST, v_test=V_TEST, seed=a.seed,
                     map_r2={k: v.r2 for k, v in maps.items()}), swaps={})
EMB_Y = Y.model.get_input_embeddings()


@torch.no_grad()
def host_run(ids, hooks, it_list):
    with hook_ctx(hooks):
        lg = Y.model(torch.tensor(ids, device=Y.device), use_cache=False).logits[:, -1].float()
    am = lg.argmax(-1)
    t_ = torch.tensor([i["id_t"] for i in it_list], device=lg.device)
    d_ = torch.tensor([i["id_d"] for i in it_list], device=lg.device)
    return (am == t_).float().tolist(), (am == d_).float().tolist(), \
        (lg.gather(1, t_[:, None]) - lg.gather(1, d_[:, None]))[:, 0].tolist()


# batch items that share host positions for a given swap
def batches(swap):
    bk = {}
    for it in test:
        h = it["y"]["vswap" if swap == "value" else "base"]
        bk.setdefault((len(h[0]), tuple(h[1]), tuple(h[2]), it["t"]), []).append(it)
    return [v[i:i + 25] for v in bk.values() for i in range(0, len(v), 25)]


for swap in ("value", "key"):
    # host prompt, donor prompt, positions, and which answer counts as success
    host_key, don_key = ("vswap", "base") if swap == "value" else ("base", "kswap")
    succ = "t" if swap == "value" else "d"
    R = {}
    for bt in batches(swap):
        t, d = bt[0]["t"], bt[0]["d"]
        hy = [it["y"][host_key][0] for it in bt]; dy = [it["y"][don_key][0] for it in bt]
        pos_y = [bt[0]["y"][host_key][2][t]] if swap == "value" else \
            [bt[0]["y"][host_key][1][t], bt[0]["y"][host_key][1][d]]

        def add(name, hooks):
            at, ad, dd = host_run(hy, hooks, bt)
            ok = at if succ == "t" else ad
            own = ad if succ == "t" else at
            r = R.setdefault(name, dict(success=[], own=[], d=[]))
            r["success"] += ok; r["own"] += own
            r["d"] += [x if succ == "t" else -x for x in dd]
        add("host (no patch)", [])
        wy = block_writes(Y, dy, YB)
        add("in-family Bind", [SetWrites(Y, YB, pos_y, wy, HY)])
        with torch.no_grad():
            emb = EMB_Y(torch.tensor(dy, device=Y.device))
        add("in-family embedding only", [SetEmb(Y, pos_y, emb)])
        for bi, G in enumerate(Y_BLOCKS[1:], 1):
            add(f"in-family G{bi} (control)", [SetWrites(Y, G, pos_y, block_writes(Y, dy, G), HY)])
        if D:
            dx = [it["x"][don_key] for it in bt]
            pos_x = [dx[0][2][t]] if swap == "value" else [dx[0][1][t], dx[0][1][d]]
            wx = block_writes(D, [x[0] for x in dx], DB)
            with torch.no_grad():
                ex = EMB_D(torch.tensor([x[0] for x in dx], device=D.device)).float()
            for name, src in (("xfam Bind", wx), ("xfam-shuffled", wx), ("xfam-emb", ex)):
                f = maps[{"xfam Bind": "xfam", "xfam-shuffled": "xfam-shuffled", "xfam-emb": "xfam-emb"}[name]]
                vec = torch.zeros(len(bt), len(hy[0]), T.shape[1])
                for py_, px_ in zip(pos_y, pos_x):
                    vec[:, py_] = f(src[:, px_]).cpu()
                add(name, [SetWrites(Y, YB, pos_y, vec, HY)])
            vec = torch.zeros(len(bt), len(hy[0]), T.shape[1]); vec[:, pos_y] = TMEAN
            add("mean Bind write", [SetWrites(Y, YB, pos_y, vec, HY)])
    res["swaps"][swap] = {k: {m: float(np.mean(v)) for m, v in r.items()} for k, r in R.items()}
    for k, v in res["swaps"][swap].items():
        log(f"{swap:5s} swap | {k:28s} success {v['success']:.2f}  own {v['own']:.2f}  D {v['d']:+6.2f}")
    json.dump(res, open(OUT, "w"), indent=1)
log(f"wrote {OUT}")
