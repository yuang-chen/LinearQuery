"""Step 26 (paper exp 3A): transfer the query across families (Qwen3.5 <-> Granite-4.0-H).

Part XVII: inside one model, the query block's write at the final token *is* the reader's query
(transplanting it from a run that asks key A makes a run asking key B answer A). Here the donor
is a model of the *other* family. Both models read the same prompt text (each in its own chat
template); a linear map f fitted on paired runs takes the donor's query-block write at the final
token to the receiver's query-block write, and f(donor asks A) is written into the receiver's
run that asks B.

  fit   keys from K_fit only; every dictionary is built from K_fit
  test  keys from K_test only (disjoint): the map never saw any key in the test dictionaries

Conditions on the test pairs (receiver asks B; donor/receiver share the dictionary):
  in-family      receiver's own write from its run asking A (Part XVII upper bound)
  xfam           f(donor query block, donor asks A)             <- the result
  xfam-self      f(donor query block, donor asks B)             (map fidelity: should keep B)
  xfam-shuffled  f fitted on shuffled pairs, donor asks A         (control)
  src=<block>    f fitted from another donor block (every linear block), donor asks A
  mean           receiver's write replaced by its fit-set mean    (removes the query)
Reported: ansA, ansB, D(A-B), reader attention on entries A and B, and the map's test R^2.
"""
import sys, json, argparse, time, random, os
import numpy as np
import torch
sys.path.insert(0, "/mnt/yuang/LinearQuery")
from src.runner import Runner, hook_ctx
from src.qkv import AttnWeights
from src.gen import Builder, chat_wrap

KEYS = """apple banana cherry date elder fig grape lemon mango peach pear plum berry onion garlic ginger
pepper tomato carrot potato walnut almond lion tiger bear wolf fox deer horse sheep goat cow pig dog cat
mouse rabbit eagle hawk owl duck goose crow snake frog fish shark whale seal chair table desk lamp door
window bed sofa clock mirror bottle cup plate bowl spoon fork knife pen book paper phone camera radio
piano guitar drum violin river lake ocean mountain hill forest desert island valley beach cloud rain snow
storm wind moon star planet sun sky king queen prince doctor nurse teacher farmer pilot captain soldier
judge lawyer artist singer dancer writer train plane boat ship truck car bus bike rocket wagon bread
cheese butter milk honey sugar rice pasta soup cake pie cookie candy coffee tea juice wine beer water
shirt dress coat hat shoe sock glove scarf belt ring watch hammer nail brush rope chain wheel box bag
basket key lock bell flag map coin""".split()
VALUES = ["red", "blue", "green", "gold", "gray", "black", "white", "orange", "yellow", "amber", "rose",
          "lime", "azure", "steel", "ice", "fire", "stone", "wood", "iron", "silver", "pink", "purple",
          "brown", "glass", "sand", "salt", "ash", "ruby", "mint", "tan"]

# query block and reader per model (Parts XVI-XVIII); blocks = all linear layers between softmax layers
SPEC = {
    "0.8B": dict(path="/mnt/yuang/gdn2-in-place/models/Qwen3.5-0.8B", query=[12, 13, 14], reader=(15, 5), gs=3),
    "9B": dict(path="/mnt/yuang/models/Qwen3.5-9B", query=[16, 17, 18], reader=(19, 11), gs=3),
    "G1b": dict(path="models/granite-4.0-h-1b", query=list(range(16, 25)), reader=(25, 1), gs=0),
    "Gtiny": dict(path="models/granite-4.0-h-tiny", query=list(range(16, 25)), reader=(25, 2), gs=0),
}

ap = argparse.ArgumentParser()
ap.add_argument("--donor", default="0.8B")
ap.add_argument("--recv", default="G1b")
ap.add_argument("--variant", default="chat8", help="chat8 | list8 | rev8")
ap.add_argument("--n_fit", type=int, default=6000)
ap.add_argument("--n_test", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args()
OUT = a.out or f"results/exp26_xquery_{a.donor}_to_{a.recv}_{a.variant}.json"
STYLE = {"list8": "list", "rev8": "rev"}.get(a.variant, "chat")
t0 = time.time()


def log(*s):
    print(f"[{(time.time() - t0) / 60:6.1f}m]", *s, flush=True)


dev = ["cuda:0", "cuda:0"]          # mamba_ssm Triton kernels need the current device
D = Runner(model_path=SPEC[a.donor]["path"], dtype=torch.float32, device=dev[0])
Y = Runner(model_path=SPEC[a.recv]["path"], dtype=torch.float32, device=dev[1])
for M in (D, Y):
    M.model.config.get_text_config()._attn_implementation = "eager"
D_BLOCKS = D.groups(SPEC[a.donor]["gs"])
DQ, YQ = SPEC[a.donor]["query"], SPEC[a.recv]["query"]
RL, RH = SPEC[a.recv]["reader"]
log(f"donor {a.donor} blocks {D_BLOCKS} query {DQ} | receiver {a.recv} query {YQ} reader L{RL}H{RH}")

rng = random.Random(a.seed)
keys = [k for k in KEYS if k not in VALUES]
rng.shuffle(keys)
K_FIT, K_TEST = keys[:len(keys) // 2], keys[len(keys) // 2:]


def render(wrap, pairs, q):
    b = Builder(); es = []
    b.add(wrap[0])
    if STYLE == "list":
        b.add("Lookup table:\n")
        for k, v in pairs:
            e0, _ = b.add(f"* {k}={v}"); _, e1 = b.add("\n"); es.append((e0, e1))
        b.add(f"Which value does {q} map to?" + wrap[1] + f"The entry {q} maps to **")
    elif STYLE == "rev":
        b.add("Dictionary:")
        for i, (k, v) in enumerate(pairs):
            e0, e1 = b.add(f" {v} is the value of {k}"); es.append((e0, e1))
            b.add("," if i < len(pairs) - 1 else ".")
        b.add(f"\nQuestion: What is the value of {q}?" + wrap[1] + f"The value of {q} is **")
    else:
        b.add("Dictionary:")
        for i, (k, v) in enumerate(pairs):
            e0, e1 = b.add(f" {k}={v}"); es.append((e0, e1))
            b.add("," if i < len(pairs) - 1 else ".")
        b.add(f"\nQuestion: What is the value of {q}?" + wrap[1] + f"The value of {q} is **")
    return b.s, es


def encode(M, pairs, q):
    wrap = chat_wrap(M.tok)
    s, es = render(wrap, pairs, q)
    e = M.tok(s, return_offsets_mapping=True)
    ent = [[i for i, (x, y) in enumerate(e.offset_mapping) if x < b_ and y > a_] for a_, b_ in es]
    return e.input_ids, ent


def make(keyset, n, with_pair):
    """items: dictionary of 8 pairs; asks A (and B when with_pair)."""
    out = []
    cfgs = [(2, 5), (5, 2), (1, 6), (6, 1)]
    while len(out) < n:
        ia, ib = cfgs[len(out) % 4] if with_pair else (rng.randrange(8), None)
        ks = rng.sample(keyset, 8); vs = rng.sample(VALUES, 8)
        pairs = list(zip(ks, vs))
        it = dict(pairs=pairs, ia=ia, ib=ib)
        for tag, M in (("d", D), ("y", Y)):
            it[f"{tag}A"], it[f"{tag}ent"] = encode(M, pairs, ks[ia])
            if with_pair:
                it[f"{tag}B"], _ = encode(M, pairs, ks[ib])
                if len(it[f"{tag}A"]) != len(it[f"{tag}B"]):
                    break
        else:
            aid, bid = Y.tok(vs[ia]).input_ids, (Y.tok(vs[ib]).input_ids if with_pair else [0])
            if len(aid) == 1 and len(bid) == 1:
                it["aid"], it["bid"] = aid[0], bid[0]
                out.append(it)
    return out


def buckets(items, keys_):
    bk = {}
    for it in items:
        bk.setdefault(tuple(len(it[k]) for k in keys_) + (str(it["yent"]),), []).append(it)
    return [v[i:i + 32] for v in bk.values() for i in range(0, len(v), 32)]


class Cap:
    def __init__(self, M, L): self.M, self.L, self.v = M, L, None
    def register(self):
        def fn(m, a_, o):
            self.v = (o[0] if isinstance(o, tuple) else o)[:, -1].detach().float().clone()
        return self.M.mixer(self.L).register_forward_hook(fn)


class SetFinal:
    """Overwrite a mixer's output at the final position with `vec` [B, H]."""
    def __init__(self, M, L, vec): self.M, self.L, self.vec = M, L, vec
    def register(self):
        def fn(m, a_, o):
            t = (o[0] if isinstance(o, tuple) else o).clone()
            t[:, -1] = self.vec.to(t.dtype).to(t.device)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
        return self.M.mixer(self.L).register_forward_hook(fn)


@torch.no_grad()
def writes(M, ids_list, layers):
    ids = torch.tensor(ids_list, device=M.device)
    caps = [Cap(M, L) for L in layers]
    with hook_ctx(caps):
        M.model(ids, use_cache=False)
    return torch.cat([c.v for c in caps], -1)            # [B, len(layers)*H]


# ------------------------------------------------------------------ fit data
D_ALL = sorted({L for G in D_BLOCKS for L in G})
EMB = D.model.get_input_embeddings()
HD = D.cfg.hidden_size


def kid(M, it, which):
    return M.tok(" " + it["pairs"][it[which]][0]).input_ids[0]


@torch.no_grad()
def donor_feats(bt, key):
    """{source: [B, dim]}: every donor block's write at the final token, and the donor's static
    input embedding of the queried key (no context at all)."""
    w = writes(D, [it["d" + key] for it in bt], D_ALL).cpu().double()
    per = {L: w[:, i * HD:(i + 1) * HD] for i, L in enumerate(D_ALL)}
    f = {f"G{bi}": torch.cat([per[L] for L in G], -1) for bi, G in enumerate(D_BLOCKS) if G}
    ids = torch.tensor([kid(D, it, "ia" if key == "A" else "ib") for it in bt], device=D.device)
    f["emb"] = EMB(ids).double().cpu()
    return f


fit = make(K_FIT, a.n_fit, with_pair=False)
Xf, Yf = {}, []
for bt in buckets(fit, ["dA", "yA"]):
    for k, v in donor_feats(bt, "A").items():
        Xf.setdefault(k, []).append(v)
    Yf.append(writes(Y, [it["yA"] for it in bt], YQ).cpu())
Xf = {k: torch.cat(v) for k, v in Xf.items()}
Yf = torch.cat(Yf).double()
SOURCES = list(Xf)
q_name = next(f"G{bi}" for bi, G in enumerate(D_BLOCKS) if G == DQ)
log(f"fit set: {len(Yf)} prompts, K_fit {len(K_FIT)} keys; donor dim {HD}, target dim {Yf.shape[1]}; "
    f"sources {SOURCES} (query block {q_name})")


def ridge(X, T):
    """Centred ridge, lambda chosen on a held-out fifth of the fit set (at most 500 prompts)."""
    n_val = min(500, max(5, len(X) // 5))
    Xt, Tt, Xv, Tv = X[n_val:], T[n_val:], X[:n_val].cuda(), T[:n_val].cuda()
    mx, mt = Xt.mean(0).cuda(), Tt.mean(0).cuda()
    Xc, Tc = Xt.cuda() - mx, Tt.cuda() - mt
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    UtT = U.T @ Tc
    best = None
    for p_ in np.arange(-3, 6.5, 0.5):
        lam = 10 ** p_ * float(S[0] ** 2) * 1e-6
        W = Vh.T @ (UtT * (S / (S ** 2 + lam))[:, None])
        err = float((((Xv - mx) @ W + mt - Tv) ** 2).sum() / ((Tv - mt) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, W)
    W = best[1]
    return (lambda Xn: (Xn.double().cuda() - mx) @ W + mt), 1 - best[0]


# ------------------------------------------------------------------ test data
test = make(K_TEST, a.n_test, with_pair=True)
TB = buckets(test, ["dA", "yA", "yB"])
HY = Y.cfg.hidden_size
FA = [donor_feats(bt, "A") for bt in TB]
FB = [donor_feats(bt, "B") for bt in TB]
YA = [writes(Y, [it["yA"] for it in bt], YQ) for bt in TB]


@torch.no_grad()
def recv_run(bt, vec=None):
    ids = torch.tensor([it["yB"] for it in bt], device=Y.device)
    hooks = []
    if vec is not None:
        hooks = [SetFinal(Y, L, vec[:, i * HY:(i + 1) * HY]) for i, L in enumerate(YQ)]
    aw = AttnWeights(Y.mixer(RL))
    with hook_ctx(hooks + [aw]):
        lg = Y.model(ids, use_cache=False).logits[:, -1].float()
    at = aw.value[:, RH, -1].float()
    ent = bt[0]["yent"]
    ea = torch.stack([at[j, ent[it["ia"]]].sum() for j, it in enumerate(bt)])
    eb = torch.stack([at[j, ent[it["ib"]]].sum() for j, it in enumerate(bt)])
    aid = torch.tensor([it["aid"] for it in bt], device=lg.device)
    bid = torch.tensor([it["bid"] for it in bt], device=lg.device)
    am = lg.argmax(-1)
    return dict(ansA=(am == aid).float().tolist(), ansB=(am == bid).float().tolist(),
                dAB=(lg.gather(1, aid[:, None]) - lg.gather(1, bid[:, None]))[:, 0].tolist(),
                attA=ea.tolist(), attB=eb.tolist())


def evaluate(vec_fn):
    """vec_fn(batch index) -> receiver query-block write [B, dim] or None; mean metrics."""
    acc = {}
    for i, bt in enumerate(TB):
        for k, v in recv_run(bt, vec_fn(i)).items():
            acc.setdefault(k, []).extend(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def test_r2(f, src):
    P = torch.cat([f(FA[i][src]).cpu() for i in range(len(TB))])
    T = torch.cat([y.double().cpu() for y in YA])
    return float(1 - ((P - T) ** 2).sum() / ((T - T.mean(0)) ** 2).sum())


res = dict(meta=dict(donor=a.donor, recv=a.recv, variant=a.variant, donor_query=DQ, recv_query=YQ,
                     reader=[RL, RH], donor_blocks=D_BLOCKS, query_source=q_name, n_fit=len(Yf),
                     n_test=sum(len(b) for b in TB), k_fit=K_FIT, k_test=K_TEST, seed=a.seed))
C = res["conditions"] = {}
ymean = Yf.mean(0)
C["host (no transplant)"] = evaluate(lambda i: None)
C["in-family"] = evaluate(lambda i: YA[i])
C["mean"] = evaluate(lambda i: ymean.expand(len(TB[i]), -1))
maps = {s: ridge(Xf[s], Yf) for s in SOURCES}
perm = torch.randperm(len(Yf), generator=torch.Generator().manual_seed(1))
fsh, _ = ridge(Xf[q_name], Yf[perm])
fq = maps[q_name][0]
C["xfam"] = evaluate(lambda i: fq(FA[i][q_name]))
C["xfam-self"] = evaluate(lambda i: fq(FB[i][q_name]))
C["xfam-shuffled"] = evaluate(lambda i: fsh(FA[i][q_name]))
for s in SOURCES:
    if s != q_name:
        f = maps[s][0]
        C[f"src={s}"] = evaluate(lambda i, f=f, s=s: f(FA[i][s]))
res["r2"] = {s: dict(val=maps[s][1], test=test_r2(maps[s][0], s)) for s in SOURCES}
for k, c in C.items():
    log(f"{k:22s} ansA {c['ansA']:.2f}  ansB {c['ansB']:.2f}  D(A-B) {c['dAB']:+6.2f}  "
        f"attA {c['attA']:.2f}  attB {c['attB']:.2f}")
log("R^2 (val / test on unseen keys): " + "  ".join(
    f"{s}:{v['val']:.2f}/{v['test']:.2f}" for s, v in res["r2"].items()))
os.makedirs("results", exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)

# ------------------------------------------------------------------ learning curves
# how many paired prompts does each donor source need before its mapped write steers the receiver?
res["curve"] = {s: [] for s in SOURCES}
for n in [n for n in (25, 50, 100, 200, 400, 800, 1600, 3200) if n < len(Yf)] + [len(Yf)]:
    line = []
    for s in SOURCES:
        f, v = ridge(Xf[s][:n], Yf[:n])
        r = evaluate(lambda i, f=f, s=s: f(FA[i][s]))
        r.update(n=n, val_r2=v)
        res["curve"][s].append(r)
        line.append(f"{s}:{r['ansA']:.2f}")
    log(f"curve n={n:5d}  ansA  " + "  ".join(line))
    json.dump(res, open(OUT, "w"), indent=1)
log(f"wrote {OUT}")
