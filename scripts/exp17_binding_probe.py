"""Step 17: does G2 bind each value token to its key?

Part XI: G2 (mostly layer 10) acts on the reader's KEYS at the value tokens. Hypothesis: G2
copies the identity of the key word (`grape`, two tokens back across `=`) into the value token
(`gray`), so the value's key vector says which entry it belongs to.

Probe: at every value token, decode the identity of its own key word (22 classes, chance 0.045)
with a linear classifier. Keys and values are sampled independently, so the value token's own
identity carries no information about its key -- only context can.

Read-outs, per condition (intact / G2 / L10 / G1 / G3 / G1+G2 removed):
  resid after layers 3, 7, 9, 10, 11, 14 at the value token, and layer 15's k_proj output there
  (the vector the reader actually matches against, per KV head).
  - "fixed"   : probe trained on the intact model, tested on the ablated model
                (is the code the network normally uses still there?)
  - "retrain" : probe trained and tested on the ablated model (is the information there at all?)
Controls: decode the value's own identity (should be ~1.0), and the key of the PREVIOUS entry
(specificity: is it "my key", or a bag of recent keys?).
"""
import sys, json, argparse, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from src.runner import Runner, hook_ctx
from src.qkv import ProjCapture
from src.task import make_examples, KEY_WORDS, VALUE_WORDS
from src.positions import token_positions

ap = argparse.ArgumentParser()
ap.add_argument("--n_train", type=int, default=600)
ap.add_argument("--n_test", type=int, default=200)
ap.add_argument("--out", default="results/exp17_binding_probe.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
LAYERS = [3, 7, 9, 10, 11, 14]
SITES = [f"resid{L}" for L in LAYERS] + ["K15"]
CONDS = {"intact": [], "remove G1": [4, 5, 6], "remove G2": [8, 9, 10], "remove L10": [10],
         "remove G3": [12, 13, 14], "remove G1+G2": [4, 5, 6, 8, 9, 10]}
KI = {k: i for i, k in enumerate(KEY_WORDS)}
VI = {v: i for i, v in enumerate(VALUE_WORDS)}


class Zero:
    def __init__(self, L): self.L = L
    def register(self):
        return R.mixer(self.L).register_forward_hook(lambda m, a_, o: torch.zeros_like(o))


class CapLayer:
    def __init__(self, L): self.L, self.value = L, None
    def register(self):
        def fn(mod, args, output):
            self.value = (output[0] if isinstance(output, tuple) else output).detach()
        return R.layers[self.L].register_forward_hook(fn)


def dataset(n, seed):
    exs = make_examples(n, 8, seed=seed)
    rows = []   # (example, value positions, own-key labels, prev-key labels, value labels)
    for e in exs:
        p = token_positions(R.tok, e)
        vp = [p["value_pos"][i] for i in range(8)]
        rows.append((e, vp, [KI[k] for k, _ in e.pairs],
                     [-1] + [KI[k] for k, _ in e.pairs[:-1]], [VI[v] for _, v in e.pairs]))
    return rows


@torch.no_grad()
def features(rows, zl, bs=25):
    feats = {s: [] for s in SITES}
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        ids = R.tok([r[0].clean_prompt() for r in chunk], return_tensors="pt").input_ids.to(R.device)
        caps = [CapLayer(L) for L in LAYERS]
        kc = ProjCapture(R.mixer(15), "k")
        with hook_ctx(caps + [kc] + [Zero(L) for L in zl]):
            R.model(ids, use_cache=False)
        idx = torch.tensor([r[1] for r in chunk], device=R.device)          # [B, 8]
        b = torch.arange(len(chunk), device=R.device)[:, None]
        for L, c in zip(LAYERS, caps):
            feats[f"resid{L}"].append(c.value[b, idx].reshape(-1, c.value.shape[-1]).float())
        feats["K15"].append(kc.value[b, idx].reshape(-1, kc.value.shape[-1]).float())
    return {s: torch.cat(v) for s, v in feats.items()}


def labels(rows, j):
    return torch.tensor([x for r in rows for x in r[j]], device=R.device)


def fit(X, y, n_cls, steps=400, wd=1e-3):
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


def acc(probe, X, y):
    W, b, mu, sd = probe
    keep = y >= 0
    return float((((X[keep] - mu) / sd @ W + b).argmax(-1) == y[keep]).float().mean())


tr, te = dataset(a.n_train, seed=1001), dataset(a.n_test, seed=2002)
TARGETS = {"own key": (2, len(KEY_WORDS)), "prev key": (3, len(KEY_WORDS)),
           "own value": (4, len(VALUE_WORDS))}
Y = {t: (labels(tr, j), labels(te, j)) for t, (j, _) in TARGETS.items()}
print(f"{len(tr) * 8} train / {len(te) * 8} test value tokens; chance 1/{len(KEY_WORDS)} = "
      f"{1 / len(KEY_WORDS):.3f} for keys", flush=True)

F = {c: (features(tr, zl), features(te, zl)) for c, zl in CONDS.items()}
intact_probes = {}
rows = []
for t, (j, n_cls) in TARGETS.items():
    ytr, yte = Y[t]
    print(f"\n== decode {t} at the value token ==", flush=True)
    print(f"  {'condition':14s} {'mode':8s} " + " ".join(f"{s:>8s}" for s in SITES), flush=True)
    for c in CONDS:
        Xtr, Xte = F[c]
        for mode in ("fixed", "retrain") if c != "intact" else ("retrain",):
            out = []
            for s in SITES:
                if mode == "retrain":
                    p = fit(Xtr[s], ytr, n_cls)
                    if c == "intact":
                        intact_probes[(t, s)] = p
                else:
                    p = intact_probes[(t, s)]
                out.append(acc(p, Xte[s], yte))
                rows.append(dict(target=t, cond=c, mode=mode, site=s, acc=out[-1]))
            print(f"  {c:14s} {mode:8s} " + " ".join(f"{v:8.3f}" for v in out), flush=True)

json.dump(dict(meta=vars(a), sites=SITES, rows=rows), open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
