"""Step 28 (paper exp 3, cont.): transplant the *Bind* block and the *whole* Bind-Query circuit
across families, with the stitching of exp27 (closed-form ridge maps, no gradient training).

Stage 1 -- Bind. The receiver's first linear block (Qwen G0 [0-2], Granite G0 [0-4]) is removed
and replaced by h_out = h_in + M_out(B_donor(M_in(h_in)) - M_in(h_in)), for every donor block.
M_in maps the receiver's *embedding* stream (the block input) to the donor's block input on tokens
that cover the same characters. Controls as in exp27: block removed, receiver mixers zeroed,
the position-wise linear bypass h_in + M(h_in), the donor Bind block with its mixers zeroed.

Stage 2 -- whole circuit. With the receiver's Bind block replaced (by the donor's Bind block, or by
the linear bypass), its Query block is replaced as well. The Query-stage maps are fitted on the
receiver *with that Bind replacement active*, i.e. on what they will see at test. Query options:
donor Query block, same with mixers zeroed, linear bypass, removed, or the receiver's own.

Test prompts use keys never seen when fitting. Metrics as exp27: acc (full vocabulary), cacc (among
the 8 values), reader attention on the target, WikiText perplexity.
"""
import sys, json, argparse, time, random, os
import numpy as np
import torch
sys.path.insert(0, "/mnt/yuang/LinearQuery")
from src.runner import Runner, hook_ctx
from src.qkv import AttnWeights
from src.xfam import SPEC, VALUES, key_split, encode, align
from src.stitch import InCap, OutCap, Replace, ZeroMix, Ridge, fit_stitch

ap = argparse.ArgumentParser()
ap.add_argument("--donor", default="0.8B")
ap.add_argument("--recv", default="G1b")
ap.add_argument("--variant", default="chat8")
ap.add_argument("--n_fit", type=int, default=1500)
ap.add_argument("--n_text", type=int, default=200)
ap.add_argument("--n_test", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args()
OUT = a.out or f"results/exp28_xcircuit_{a.donor}_into_{a.recv}_{a.variant}.json"
STYLE = {"list8": "list", "rev8": "rev"}.get(a.variant, "chat")
t0 = time.time()


def log(*s):
    print(f"[{(time.time() - t0) / 60:6.1f}m]", *s, flush=True)


D = Runner(model_path=SPEC[a.donor]["path"], dtype=torch.float32, device="cuda:0")
Y = Runner(model_path=SPEC[a.recv]["path"], dtype=torch.float32, device="cuda:0")
for M in (D, Y):
    M.model.config.get_text_config()._attn_implementation = "eager"
D_BLOCKS = [G for G in D.groups(SPEC[a.donor]["gs"]) if G]
Y_BLOCKS = [G for G in Y.groups(SPEC[a.recv]["gs"]) if G]
DB, DQ = D_BLOCKS[0], SPEC[a.donor]["query"]
YB, YQ = Y_BLOCKS[0], SPEC[a.recv]["query"]
for G in (YB, YQ):
    assert G == list(range(G[0], G[-1] + 1)) and all(L in Y.gdn_layers for L in G), G
iDB, iDQ = D_BLOCKS.index(DB), D_BLOCKS.index(DQ)
RL, RH = SPEC[a.recv]["reader"]
log(f"receiver {a.recv}: Bind {YB}, Query {YQ}, reader L{RL}H{RH} | donor {a.donor}: blocks {D_BLOCKS}, "
    f"Bind {DB}, Query {DQ}")

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


def text_items(n):
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
    out = []
    for t in [t for t in d["text"] if len(t) > 700][:n]:
        t = t[:600]
        ey = Y.tok(t, return_offsets_mapping=True); ed = D.tok(t, return_offsets_mapping=True)
        cy = [(0, x, y) if y > x else None for x, y in ey.offset_mapping]
        cd = [(0, x, y) if y > x else None for x, y in ed.offset_mapping]
        out.append(dict(y=ey.input_ids, d=ed.input_ids, al=align(cy, cd)))
    return out


fit = make(K_FIT, a.n_fit)
ALL = fit + text_items(a.n_text)
log(f"fit: {len(fit)} prompts + {len(ALL) - len(fit)} text chunks")


@torch.no_grad()
def capture(M, ids, lo, hi, hooks=()):
    ci, co = InCap(M, lo), OutCap(M, hi)
    with hook_ctx(list(hooks) + [ci, co]):
        M.model(torch.tensor([ids], device=M.device), use_cache=False)
    return ci.v[0].cpu(), (co.v[0] - ci.v[0]).cpu()


def capture_recv(key, lo, hi, hooks_fn=lambda: []):
    """it[key] = receiver residual entering layer lo; returns the block writes, all rows."""
    dws = []
    for it in ALL:
        it[key], dw = capture(Y, it["y"], lo, hi, hooks_fn())
        dws.append(dw)
    return torch.cat(dws)


def capture_donor(bi):
    for it in ALL:
        it[f"hd{bi}"], _ = capture(D, it["d"], D_BLOCKS[bi][0], D_BLOCKS[bi][0])


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


res = dict(meta=dict(donor=a.donor, recv=a.recv, variant=a.variant, seed=a.seed, recv_bind=YB,
                     recv_query=YQ, donor_blocks=D_BLOCKS, donor_bind=DB, donor_query=DQ,
                     reader=[RL, RH], n_fit=len(fit), n_text=len(ALL) - len(fit), n_test=len(test),
                     k_fit=K_FIT, k_test=K_TEST, r2={}),
           bind={}, circuit={})
save = lambda: json.dump(res, open(OUT, "w"), indent=1)
zero = lambda h: torch.zeros_like(h)
RB = lambda g: Replace(Y, YB[0], YB[-1], g)
RQ = lambda g: Replace(Y, YQ[0], YQ[-1], g)

# ------------------------------------------------------------------ stage 1: Bind
Yd_b = capture_recv("yb", YB[0], YB[-1])
bypass_b = Ridge(torch.cat([it["yb"] for it in ALL]), Yd_b)
res["meta"]["r2"]["bind_bypass"] = bypass_b.r2
S = res["bind"]
S["intact"] = evaluate([])
S["removed"] = evaluate([RB(zero)])
S["mixers zeroed"] = evaluate([ZeroMix(Y, L) for L in YB])
S["bypass"] = evaluate([RB(bypass_b)])
log(f"bind: intact {S['intact']}\n       removed {S['removed']}\n       bypass {S['bypass']} (R^2 {bypass_b.r2:.3f})")
stitch_b = {}
for bi, G in enumerate(D_BLOCKS):
    capture_donor(bi)
    for nomix in ((False, True) if bi == iDB else (False,)):
        st = fit_stitch(D, G, nomix, ALL, "yb", f"hd{bi}", Yd_b)
        name = f"G{bi}" + ("-nomix" if nomix else "") + ("*" if bi == iDB else "")
        res["meta"]["r2"][f"bind:{name}"] = dict(m_in=st.m_in.r2, m_out=st.m_out.r2)
        S[f"donor={name}"] = evaluate([RB(st)])
        log(f"bind donor={name:9s} {G}: {S[f'donor={name}']}  (M_in {st.m_in.r2:.3f}, M_out {st.m_out.r2:.3f})")
        if bi == iDB:
            stitch_b[nomix] = st
    if bi not in (iDB, iDQ):
        for it in ALL:
            it.pop(f"hd{bi}", None)
    save()

# ------------------------------------------------------------------ stage 2: whole circuit
for bname, gb in (("donor Bind", stitch_b[False]), ("bypass Bind", bypass_b)):
    Yd_q = capture_recv("yq", YQ[0], YQ[-1], lambda: [RB(gb)])
    bypass_q = Ridge(torch.cat([it["yq"] for it in ALL]), Yd_q)
    sq = fit_stitch(D, DQ, False, ALL, "yq", f"hd{iDQ}", Yd_q)
    sq0 = fit_stitch(D, DQ, True, ALL, "yq", f"hd{iDQ}", Yd_q)
    res["meta"]["r2"][f"query|{bname}"] = dict(bypass=bypass_q.r2, m_in=sq.m_in.r2, m_out=sq.m_out.r2)
    C = res["circuit"][bname] = {}
    C["own Query"] = evaluate([RB(gb)])
    C["donor Query*"] = evaluate([RB(gb), RQ(sq)])
    C["donor Query, mixers zeroed"] = evaluate([RB(gb), RQ(sq0)])
    C["bypass Query"] = evaluate([RB(gb), RQ(bypass_q)])
    C["Query removed"] = evaluate([RB(gb), RQ(zero)])
    for k, v in C.items():
        log(f"circuit [{bname}] + [{k}]: {v}")
    save()

for it in ALL:
    for k in [k for k in it if k.startswith(("hd", "yb", "yq"))]:
        it.pop(k)
res["meta"]["seconds"] = time.time() - t0
save()
log(f"wrote {OUT}")
