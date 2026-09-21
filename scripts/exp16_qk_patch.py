"""Step 16: does G3 / layer 14 build the reader's QUERY or its KEYS?

exp15 found that ablating GDN layer 14 (and more so 13+14, or all of G3) makes the reader head
L15H5 lose its aim: attention on the source value falls 0.78 -> 0.35 / 0.08 / 0.06, and in the
worked example it attends to the first dictionary entries instead. That says "address", not
"content", but not which side of the dot product is damaged.

Path patch at softmax layer 15 (projection outputs, before q_norm/k_norm and RoPE):

  RESCUE     run with the ablation, restore Q / K / V from the intact run of the same prompt.
             Q at the final token only; K / V at all positions, value tokens only, or key tokens.
  NECESSITY  intact run, insert Q / K taken from the ablated run of the same prompt.

Downstream layers (16+) still see the ablated residual stream, so besides final accuracy we
read the logit-lens gap right after layer 15 -- the reader's own output -- and L15H5's
attention on the source value.
"""
import sys, json, argparse, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from src.runner import Runner, hook_ctx
from src.qkv import AttnWeights, ProjPatch, ProjCapture
from src.task import make_examples
from src.harness import group_examples

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/exp16_qk_patch.json")
ap.add_argument("--abl", default="L14=14;L13+14=13,14;G3=12,13,14",
                help="ablation sets, 'name=l1,l2;name=...'")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
R.model.config.get_text_config()._attn_implementation = "eager"
HD = R.cfg.head_dim
READER_L, READER_H = 15, 5
KV_H = READER_H // (R.cfg.num_attention_heads // R.cfg.num_key_value_heads)   # GQA group of H5
NORM = R.model.model.norm if hasattr(R.model.model, "norm") else R.model.model.language_model.norm
HEAD = R.model.lm_head

exs = []
for i, (ti, di) in enumerate([(2, 5), (5, 2), (1, 6), (6, 1)]):
    exs += make_examples(25, 8, seed=11 + i, target_pos=ti, distract_pos=di)
groups_ex = group_examples(R, exs)


class Zero:
    def __init__(self, L): self.L = L
    def register(self):
        return R.mixer(self.L).register_forward_hook(lambda m, a_, o: torch.zeros_like(o))


class CapResid:
    """Residual stream at the final position after layer L."""
    def __init__(self, L): self.L, self.value = L, None
    def register(self):
        def fn(mod, args, output):
            h = output[0] if isinstance(output, tuple) else output
            self.value = h[:, -1].detach().clone()
        return R.layers[self.L].register_forward_hook(fn)


ABLATIONS = {n: [int(x) for x in ls.split(",")]
             for n, ls in (kv.split("=") for kv in a.abl.split(";"))}


@torch.no_grad()
def capture_qkv(ids, zl, layer=READER_L):
    attn = R.mixer(layer)
    caps = {w: ProjCapture(attn, w) for w in "qkv"}
    with hook_ctx(list(caps.values()) + [Zero(L) for L in zl]):
        R.model(ids, use_cache=False)
    return {w: c.value for w, c in caps.items()}


@torch.no_grad()
def evaluate(make_hooks):
    """make_hooks(g, which_ids) -> hooks. Returns acc, final gap, L15 lens gap, L15H5 attn."""
    ok = tot = 0
    gaps, lens, att = [], [], []
    for g in groups_ex:
        ds, dl = [], []
        for key, ids, want in (("clean", g.clean_ids, g.aid), ("corr", g.corr_ids, g.cid)):
            cap, aw = CapResid(READER_L), AttnWeights(R.mixer(READER_L))
            with hook_ctx([cap, aw] + make_hooks(g, key)):
                lg = R.model(ids, use_cache=False).logits[:, -1]
            ok += (lg.argmax(-1) == want).sum().item(); tot += ids.shape[0]
            ds.append(g.D(lg)); dl.append(g.D(HEAD(NORM(cap.value))))
            if key == "corr":
                att += aw.value[:, READER_H, g.pos["final"], g.pos["target_value"]].tolist()
        gaps += (ds[0] - ds[1]).tolist(); lens += (dl[0] - dl[1]).tolist()
    return dict(acc=ok / tot, gap=float(np.mean(gaps)), lens15=float(np.mean(lens)),
                attn=float(np.mean(att)))


# donors: intact and every ablation, for every group and both prompt versions
print("capturing Q/K/V donors ...", flush=True)
DON = {}
for g in groups_ex:
    for key, ids in (("clean", g.clean_ids), ("corr", g.corr_ids)):
        DON[(id(g), key, "intact")] = capture_qkv(ids, [])
        DON[(id(g), key, "intact", 19)] = capture_qkv(ids, [], layer=19)
        for an, zl in ABLATIONS.items():
            DON[(id(g), key, an)] = capture_qkv(ids, zl)


def P(g, key, src, which, where, heads=None, layer=READER_L):
    d = DON[(id(g), key, src) if layer == READER_L else (id(g), key, src, layer)]
    pos = {"final": [g.pos["final"]],
           "all": list(range(g.pos["n_tokens"])),
           "values": list(g.pos["value_pos"].values()),
           "keys": list(g.pos["key_pos"].values())}[where]
    return ProjPatch(R.mixer(layer), which, pos, d[which], HD, heads=heads)


rows = []
base = evaluate(lambda g, k: [])
print(f"intact: acc {base['acc']:.3f}  gap {base['gap']:+.2f}  lens@L15 {base['lens15']:+.2f}  "
      f"L15H5 attn {base['attn']:.3f}\n", flush=True)
rows.append(dict(kind="intact", **base))


def show(label, r, ref):
    rec = (r["gap"] - ref["gap"]) / (base["gap"] - ref["gap"]) if ref else float("nan")
    print(f"  {label:34s} acc {r['acc']:.3f}  gap {r['gap']:+6.2f}  lens@L15 {r['lens15']:+6.2f}  "
          f"attn {r['attn']:.3f}  rec {rec:+.2f}", flush=True)
    return rec


# ============================================================ A. rescue
RESCUES = [
    ("Q@final (all heads)",        lambda g, k: [P(g, k, "intact", "q", "final")]),
    ("Q@final (H5 only)",          lambda g, k: [P(g, k, "intact", "q", "final", heads=[READER_H])]),
    ("Q+gate@final (all heads)",   lambda g, k: [ProjPatch(R.mixer(READER_L), "q", [g.pos["final"]],
                                                  DON[(id(g), k, "intact")]["q"], HD, include_gate=True)]),
    ("K@all",                      lambda g, k: [P(g, k, "intact", "k", "all")]),
    ("K@all (H5's KV head)",       lambda g, k: [P(g, k, "intact", "k", "all", heads=[KV_H])]),
    ("K@value tokens",             lambda g, k: [P(g, k, "intact", "k", "values")]),
    ("K@key tokens",               lambda g, k: [P(g, k, "intact", "k", "keys")]),
    ("V@all",                      lambda g, k: [P(g, k, "intact", "v", "all")]),
    ("Q@final + K@all",            lambda g, k: [P(g, k, "intact", "q", "final"),
                                                 P(g, k, "intact", "k", "all")]),
    ("Q+K+V@all (whole L15 input)", lambda g, k: [P(g, k, "intact", w, "all") for w in "qkv"]),
    ("Q+K+V@all at L15 and L19",   lambda g, k: [P(g, k, "intact", w, "all") for w in "qkv"] +
                                                [P(g, k, "intact", w, "all", layer=19) for w in "qkv"]),
]
for an, zl in ABLATIONS.items():
    print(f"== A. rescue: ablate {an} {zl}, restore from intact run ==", flush=True)
    abl = evaluate(lambda g, k: [Zero(L) for L in zl])
    show("(ablated, no patch)", abl, None)
    rows.append(dict(kind="ablated", abl=an, **abl))
    for name, mk in RESCUES:
        r = evaluate(lambda g, k, mk=mk: [Zero(L) for L in zl] + mk(g, k))
        rec = show(name, r, abl)
        rows.append(dict(kind="rescue", abl=an, patch=name, rec=rec, **r))
    print(flush=True)

# ============================================================ B. necessity
print("== B. necessity: intact run, insert ablated Q or K ==", flush=True)
for an in ABLATIONS:
    for name, mk in (("Q@final", lambda g, k, an=an: [P(g, k, an, "q", "final")]),
                     ("K@all", lambda g, k, an=an: [P(g, k, an, "k", "all")]),
                     ("K@value tokens", lambda g, k, an=an: [P(g, k, an, "k", "values")]),
                     ("Q@final + K@all", lambda g, k, an=an: [P(g, k, an, "q", "final"),
                                                              P(g, k, an, "k", "all")])):
        r = evaluate(mk)
        print(f"  {an:7s} {name:20s} acc {r['acc']:.3f}  gap {r['gap']:+6.2f}  "
              f"lens@L15 {r['lens15']:+6.2f}  attn {r['attn']:.3f}", flush=True)
        rows.append(dict(kind="necessity", abl=an, patch=name, **r))

json.dump(dict(meta=vars(a), rows=rows), open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
