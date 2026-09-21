"""Step 19: which layer of G0 (+ softmax 3) puts the key's identity into the value token?

exp17: a linear probe decodes the own key of each value token at 0.996 after layer 3. Only the
sequence mixers (GDN conv / recurrence, softmax attention) move information between positions,
and the key word sits two tokens before its value (` grape`, `=`, `gray`), inside a GDN
layer's 4-tap causal conv window.

  A  where it appears: probe the residual at the value token after the embedding and after
     layers 0-3, and probe each mixer's *own output* at the value token (who writes it).
  B  what is necessary: re-probe after layer 3 under
       - one mixer's output zeroed at the value tokens only,
       - a GDN layer's conv reduced to 1 tap (current token), or its recurrence removed
         (state reset every token), or both (token-local).
  C  does retrieval need it: acc / gap on the 200 standard prompts under the conv / recurrence
     interventions (which keep token identity, unlike zeroing a layer).
Probe metrics: test accuracy (22 keys, chance 0.045) and test cross-entropy (nats; ln 22 = 3.09
is chance), which still separates conditions when accuracy is at ceiling.
"""
import sys, json, argparse, math, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from transformers import DynamicCache
from src.runner import Runner, hook_ctx
from src.task import make_examples, KEY_WORDS
from src.positions import token_positions
from src.harness import group_examples

ap = argparse.ArgumentParser()
ap.add_argument("--n_train", type=int, default=600)
ap.add_argument("--n_test", type=int, default=200)
ap.add_argument("--out", default="results/exp19_binding_origin.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
KI = {k: i for i, k in enumerate(KEY_WORDS)}
EMB = R.model.get_input_embeddings()


def dataset(n, seed):
    rows = []
    for e in make_examples(n, 8, seed=seed):
        p = token_positions(R.tok, e)
        rows.append((e, [p["value_pos"][i] for i in range(8)], [KI[k] for k, _ in e.pairs],
                     [-1] + [KI[k] for k, _ in e.pairs[:-1]]))
    return rows


class conv_1tap:
    """Context manager: GDN layers' causal conv reduced to the current-token tap."""
    def __init__(self, Ls): self.Ls = Ls
    def __enter__(self):
        self.w = {L: R.mixer(L).conv1d.weight.data.clone() for L in self.Ls}
        for L in self.Ls:
            R.mixer(L).conv1d.weight.data[..., :-1] = 0
    def __exit__(self, *e):
        for L in self.Ls:
            R.mixer(L).conv1d.weight.data.copy_(self.w[L])


@torch.no_grad()
def forward(ids, hooks, norecur):
    """Full forward, or token-by-token with the given layers' recurrent state reset each step."""
    with hook_ctx(hooks):
        if not norecur:
            return R.model(ids, use_cache=False).logits
        cache = DynamicCache(config=R.model.config.get_text_config())
        outs = []
        for i in range(ids.shape[1]):
            outs.append(R.model(ids[:, i:i + 1], past_key_values=cache, use_cache=True).logits)
            for L in norecur:
                rs = cache.layers[L].recurrent_states[0]
                if rs is not None:
                    rs.zero_()
        return torch.cat(outs, 1)


@torch.no_grad()
def features(rows, hooks_fn=lambda c: [], norecur=(), comps=False, bs=25):
    """Residual after emb and layers 0-3 at the value tokens (+ each mixer's own output)."""
    sites = ["emb", "resid0", "resid1", "resid2", "resid3"] + (
        ["mix0", "mix1", "mix2", "attn3"] if comps else [])
    F = {s: [] for s in sites}
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        ids = R.tok([r[0].clean_prompt() for r in chunk], return_tensors="pt").input_ids.to(R.device)
        store = {s: [] for s in sites}

        def grab_layer(L, name):
            class C:
                def register(s):
                    return R.layers[L].register_forward_hook(
                        lambda m, a_, o: store[name].append((o[0] if isinstance(o, tuple) else o).detach()))
            return C()

        def grab_mixer(L, name):
            class C:
                def register(s):
                    return R.mixer(L).register_forward_hook(
                        lambda m, a_, o: store[name].append((o[0] if isinstance(o, tuple) else o).detach()))
            return C()

        caps = [grab_layer(L, f"resid{L}") for L in range(4)]
        if comps:
            caps += [grab_mixer(0, "mix0"), grab_mixer(1, "mix1"), grab_mixer(2, "mix2"),
                     grab_mixer(3, "attn3")]
        forward(ids, caps + hooks_fn(chunk), norecur)
        idx = torch.tensor([r[1] for r in chunk], device=R.device)
        b = torch.arange(len(chunk), device=R.device)[:, None]
        e = EMB(ids)
        F["emb"].append(e[b, idx].reshape(-1, e.shape[-1]).float())
        for s in sites[1:]:
            v = torch.cat(store[s], 1)                    # token-by-token parts -> [B, T, H]
            F[s].append(v[b, idx].reshape(-1, v.shape[-1]).float())
    return {s: torch.cat(v) for s, v in F.items()}


def fit(X, y, n_cls=len(KEY_WORDS), steps=400, wd=1e-3):
    keep = y >= 0
    X, y = X[keep], y[keep]
    mu, sd = X.mean(0), X.std(0) + 1e-5
    W = torch.zeros(X.shape[1], n_cls, device=X.device, requires_grad=True)
    b = torch.zeros(n_cls, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=steps, line_search_fn="strong_wolfe")
    Xn = (X - mu) / sd

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xn @ W + b, y) + wd * W.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach(), mu, sd


def score(p, X, y):
    W, b, mu, sd = p
    keep = y >= 0
    lg = (X[keep] - mu) / sd @ W + b
    return (float((lg.argmax(-1) == y[keep]).float().mean()),
            float(torch.nn.functional.cross_entropy(lg, y[keep])))


tr, te = dataset(a.n_train, 1001), dataset(a.n_test, 2002)
ytr = torch.tensor([x for r in tr for x in r[2]], device=R.device)
yte = torch.tensor([x for r in te for x in r[2]], device=R.device)
ptr = torch.tensor([x for r in tr for x in r[3]], device=R.device)
pte = torch.tensor([x for r in te for x in r[3]], device=R.device)
rows = []
print(f"chance: acc {1 / 22:.3f}, CE {math.log(22):.2f}", flush=True)

# ================================================================== A
print("\n== A. where does the own key appear at the value token? (intact) ==", flush=True)
Ftr, Fte = features(tr, comps=True), features(te, comps=True)
probes = {}
print(f"  {'site':8s} {'own key acc':>11s} {'CE':>6s}   {'prev key acc':>12s} {'CE':>6s}", flush=True)
for s in Ftr:
    p = fit(Ftr[s], ytr); probes[s] = p
    ao, co = score(p, Fte[s], yte)
    ap_, cp = score(fit(Ftr[s], ptr), Fte[s], pte)
    rows.append(dict(section="A", site=s, own_acc=ao, own_ce=co, prev_acc=ap_, prev_ce=cp))
    print(f"  {s:8s} {ao:11.3f} {co:6.2f}   {ap_:12.3f} {cp:6.2f}", flush=True)

# ================================================================== B
print("\n== B. what is necessary? own key after layer 3 (retrained / intact probe) ==", flush=True)
def at_values(L):
    def hooks(chunk):
        pos = [r[1] for r in chunk]
        class H:
            def register(s):
                def fn(m, a_, o):
                    t = (o[0] if isinstance(o, tuple) else o).clone()
                    for bi, ps in enumerate(pos):
                        for p_ in ps:
                            t[bi, p_] = 0
                    return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
                return R.mixer(L).register_forward_hook(fn)
        return [H()]
    return hooks


B_CONDS = [("intact", {}),
           ("layer 0 mixer zeroed @ value tokens", dict(hooks_fn=at_values(0))),
           ("layer 1 mixer zeroed @ value tokens", dict(hooks_fn=at_values(1))),
           ("layer 2 mixer zeroed @ value tokens", dict(hooks_fn=at_values(2))),
           ("softmax 3 zeroed @ value tokens", dict(hooks_fn=at_values(3))),
           ("layers 1+2 zeroed @ value tokens",
            dict(hooks_fn=lambda c: at_values(1)(c) + at_values(2)(c)))]
for L in (0, 1, 2):
    B_CONDS += [(f"layer {L} conv 1 tap", dict(conv=[L])),
                (f"layer {L} no recurrence", dict(norecur=(L,))),
                (f"layer {L} token-local (both)", dict(conv=[L], norecur=(L,)))]
B_CONDS += [("layers 0-2 conv 1 tap", dict(conv=[0, 1, 2])),
            ("layers 0-2 no recurrence", dict(norecur=(0, 1, 2))),
            ("layers 0-2 token-local", dict(conv=[0, 1, 2], norecur=(0, 1, 2)))]

print(f"  {'condition':38s} {'retrain acc':>11s} {'CE':>6s}  {'fixed acc':>9s} {'CE':>6s}", flush=True)
for name, kw in B_CONDS:
    with conv_1tap(kw.get("conv", [])):
        fa = features(tr, kw.get("hooks_fn", lambda c: []), kw.get("norecur", ()))
        fb = features(te, kw.get("hooks_fn", lambda c: []), kw.get("norecur", ()))
    ra, rc = score(fit(fa["resid3"], ytr), fb["resid3"], yte)
    fa_, fc = score(probes["resid3"], fb["resid3"], yte)
    rows.append(dict(section="B", cond=name, retrain_acc=ra, retrain_ce=rc, fixed_acc=fa_, fixed_ce=fc))
    print(f"  {name:38s} {ra:11.3f} {rc:6.2f}  {fa_:9.3f} {fc:6.2f}", flush=True)

# ================================================================== C
print("\n== C. retrieval under the conv / recurrence interventions ==", flush=True)
exs = []
for i, (ti, di) in enumerate([(2, 5), (5, 2), (1, 6), (6, 1)]):
    exs += make_examples(25, 8, seed=11 + i, target_pos=ti, distract_pos=di)
groups = group_examples(R, exs)
for name, kw in [("intact", {})] + [c for c in B_CONDS if "conv" in c[1] or "norecur" in c[1]]:
    ok = tot = 0; gaps = []
    with conv_1tap(kw.get("conv", [])):
        for g in groups:
            ds = []
            for ids, want in ((g.clean_ids, g.aid), (g.corr_ids, g.cid)):
                lg = forward(ids, [], kw.get("norecur", ()))[:, -1]
                ok += (lg.argmax(-1) == want).sum().item(); tot += ids.shape[0]
                ds.append(g.D(lg))
            gaps += (ds[0] - ds[1]).tolist()
    rows.append(dict(section="C", cond=name, acc=ok / tot, gap=float(np.mean(gaps))))
    print(f"  {name:38s} acc {ok / tot:.3f}  gap {np.mean(gaps):+6.2f}", flush=True)

json.dump(dict(meta=vars(a), rows=rows), open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
