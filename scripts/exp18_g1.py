"""Step 18: what does G1 [4,5,6] do?

What is known: removing G1 is the most expensive non-G0 group for perplexity (+0.70 nats) but
mild for retrieval (acc 0.840, reader attention 0.78 -> 0.64). Its retrieval effect is mostly on
the reader's query (Part XI), attention leaks to other values rather than to the sink (Part XII),
and restoring all of layer 15 and 19's inputs recovers only part of the final gap.

  A  layers of G1: remove each subset.
  B  where G1's write is needed: zero G1's output only at selected positions.
  C  direct vs mediated: remove G1, then restore downstream components (G2, G3, softmax 7/11,
     MLPs) to their intact outputs -- what is left is G1's direct write; and the converse,
     keep G1's write but give downstream components their G1-less outputs.
  D  general text: is G1's perplexity cost concentrated on in-context copying (induction)
     tokens, compared with the other groups?
Retrieval metrics: acc, gap, lens@L15 (reader output), and L15H5 attention split into target
value / other values / <|im_start|> sink (clean prompts, final token).
"""
import sys, json, argparse, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from datasets import load_dataset
from src.runner import Runner, hook_ctx, Capture, OutPatch
from src.qkv import AttnWeights
from src.task import make_examples
from src.harness import group_examples

ap = argparse.ArgumentParser()
ap.add_argument("--n_ppl", type=int, default=40)
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--out", default="results/exp18_g1.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
R.model.config.get_text_config()._attn_implementation = "eager"
NORM = R.model.model.norm if hasattr(R.model.model, "norm") else R.model.model.language_model.norm
HEAD = R.model.lm_head
G1 = [4, 5, 6]
GROUPS = {"G0": [0, 1, 2], "G1": [4, 5, 6], "G2": [8, 9, 10], "G3": [12, 13, 14],
          "G4": [16, 17, 18], "G5": [20, 21, 22]}

exs = []
for i, (ti, di) in enumerate([(2, 5), (5, 2), (1, 6), (6, 1)]):
    exs += make_examples(25, 8, seed=11 + i, target_pos=ti, distract_pos=di)
groups = group_examples(R, exs)


class Zero:
    """Zero a mixer's output, optionally only at positions pos_fn(g)."""
    def __init__(self, L, pos=None): self.L, self.pos = L, pos
    def register(self):
        pos = self.pos
        def fn(m, a_, o):
            if pos is None:
                return torch.zeros_like(o)
            t = o.clone(); t[:, pos] = 0
            return t
        return R.mixer(self.L).register_forward_hook(fn)


class CapResid:
    def __init__(self, L): self.L, self.value = L, None
    def register(self):
        def fn(mod, args, output):
            h = output[0] if isinstance(output, tuple) else output
            self.value = h[:, -1].detach().clone()
        return R.layers[self.L].register_forward_hook(fn)


def is_attn(L): return R.layer_types[L] == "full_attention"


def modules(Ls, kinds=("mixer", "mlp")):
    out = []
    for L in Ls:
        if "mixer" in kinds: out.append((("mixer", L), R.mixer(L), 0 if is_attn(L) else None))
        if "mlp" in kinds: out.append((("mlp", L), R.mlp(L), None))
    return out


@torch.no_grad()
def capture_all(ids, hooks):
    caps = {k: Capture(m, out_index=oi) for k, m, oi in modules(range(4, 15))}
    with hook_ctx(list(caps.values()) + hooks):
        R.model(ids, use_cache=False)
    return {k: c.value for k, c in caps.items()}


print("capturing intact and G1-removed activations ...", flush=True)
DON = {}
for g in groups:
    for key, ids in (("clean", g.clean_ids), ("corr", g.corr_ids)):
        DON[(id(g), key, "intact")] = capture_all(ids, [])
        DON[(id(g), key, "noG1")] = capture_all(ids, [Zero(L) for L in G1])


def restore(src, Ls, kinds=("mixer", "mlp")):
    def mk(g, key):
        d = DON[(id(g), key, src)]
        allp = list(range(g.pos["n_tokens"]))
        return [OutPatch(m, allp, d[k], out_index=oi) for k, m, oi in modules(Ls, kinds)]
    return mk


@torch.no_grad()
def evaluate(make_hooks):
    ok = tot = 0
    gaps, lens, split = [], [], []
    for g in groups:
        ds, dl = [], []
        for key, ids, want in (("clean", g.clean_ids, g.aid), ("corr", g.corr_ids, g.cid)):
            cap, aw = CapResid(15), AttnWeights(R.mixer(15))
            with hook_ctx([cap, aw] + make_hooks(g, key)):
                lg = R.model(ids, use_cache=False).logits[:, -1]
            ok += (lg.argmax(-1) == want).sum().item(); tot += ids.shape[0]
            ds.append(g.D(lg)); dl.append(g.D(HEAD(NORM(cap.value))))
            if key == "clean":
                at = aw.value[:, 5, g.pos["final"]].float()
                vp = list(g.pos["value_pos"].values())
                tv = g.pos["target_value"]
                split += torch.stack([at[:, tv], at[:, vp].sum(1) - at[:, tv], at[:, 0]], 1).tolist()
        gaps += (ds[0] - ds[1]).tolist(); lens += (dl[0] - dl[1]).tolist()
    s = np.mean(split, 0)
    return dict(acc=ok / tot, gap=float(np.mean(gaps)), lens15=float(np.mean(lens)),
                att_target=float(s[0]), att_other_values=float(s[1]), att_sink=float(s[2]))


rows = []


def show(section, label, r):
    rows.append(dict(section=section, cond=label, **r))
    print(f"  {label:44s} acc {r['acc']:.3f}  gap {r['gap']:+6.2f}  lens@L15 {r['lens15']:+6.2f}  "
          f"attn tgt {r['att_target']:.3f} oth {r['att_other_values']:.3f} "
          f"sink {r['att_sink']:.3f}", flush=True)


show("base", "intact", evaluate(lambda g, k: []))

# ================================================================= A
print("\n== A. layers of G1 ==", flush=True)
for S in ([4], [5], [6], [4, 5], [4, 6], [5, 6], [4, 5, 6]):
    show("A", f"remove {S}", evaluate(lambda g, k, S=S: [Zero(L) for L in S]))

# ================================================================= B
print("\n== B. where is G1's write needed? (zero G1 output only at these positions) ==", flush=True)


def P(g):
    kp, vp = sorted(g.pos["key_pos"].values()), sorted(g.pos["value_pos"].values())
    dict_all = list(range(kp[0], vp[-1] + 2))            # keys, '=', values, ',' / '.'
    after = list(range(vp[-1] + 2, g.pos["n_tokens"]))   # question, template, answer prefix
    return {"keys": kp, "'=' tokens": [k + 1 for k in kp], "values": vp,
            "',' after values": [v + 1 for v in vp], "whole dictionary": dict_all,
            "query key (question)": [g.pos["query_key_question"]],
            "everything after the dictionary": after, "final token": [g.pos["final"]],
            "everything except the dictionary": [p for p in range(g.pos["n_tokens"])
                                                 if p not in set(dict_all)]}


for site in P(groups[0]):
    show("B", f"@ {site}", evaluate(lambda g, k, site=site: [Zero(L, P(g)[site]) for L in G1]))

# ================================================================= C
print("\n== C. direct vs mediated (remove G1, restore downstream to intact) ==", flush=True)
zero_g1 = lambda g, k: [Zero(L) for L in G1]
C_RESTORE = [
    ("restore G2 mixers", [8, 9, 10], ("mixer",)),
    ("restore G3 mixers", [12, 13, 14], ("mixer",)),
    ("restore G2+G3 mixers", [8, 9, 10, 12, 13, 14], ("mixer",)),
    ("restore softmax 7 + 11", [7, 11], ("mixer",)),
    ("restore all mixers 7-14", list(range(7, 15)), ("mixer",)),
    ("restore MLPs 4-14", list(range(4, 15)), ("mlp",)),
    ("restore mixers 7-14 + MLPs 4-14 (direct only)", list(range(4, 15)), ("mixer", "mlp")),
]
for label, Ls, kinds in C_RESTORE:
    Ls_ = [L for L in Ls if not (L in G1 and "mixer" in kinds and kinds == ("mixer",))]
    mk = restore("intact", Ls_, kinds)
    if kinds == ("mixer", "mlp"):     # never restore G1's own mixers
        base_mk = mk
        mk = lambda g, k, b=base_mk: [h for h in b(g, k) if h.module not in [R.mixer(L) for L in G1]]
    show("C", label, evaluate(lambda g, k, mk=mk: zero_g1(g, k) + mk(g, k)))

print("\n  converse: keep G1's write, give downstream components their G1-less outputs", flush=True)
for label, Ls, kinds in (("G2+G3 mixers from G1-less run", [8, 9, 10, 12, 13, 14], ("mixer",)),
                         ("all mixers 7-14 from G1-less run", list(range(7, 15)), ("mixer",)),
                         ("mixers 7-14 + MLPs 4-14 from G1-less run", list(range(4, 15)),
                          ("mixer", "mlp"))):
    base_mk = restore("noG1", Ls, kinds)
    mk = lambda g, k, b=base_mk: [h for h in b(g, k) if h.module not in [R.mixer(L) for L in G1]]
    show("C-converse", label, evaluate(mk))

# ================================================================= D
print("\n== D. general text: induction vs other tokens ==", flush=True)
d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
ids = R.tok("\n\n".join(t for t in d["text"] if t.strip()), return_tensors="pt").input_ids[0]
seqs = [ids[i * a.seq_len:(i + 1) * a.seq_len] for i in range(a.n_ppl)]
masks = []
for s in seqs:
    s = s.tolist(); seen = set(); m = []
    for t in range(1, len(s)):
        m.append((s[t - 1], s[t]) in seen)       # target s[t] completes a bigram seen before
        seen.add((s[t - 1], s[t]))
    masks.append(torch.tensor(m))
ind_frac = float(torch.cat(masks).float().mean())
print(f"  {ind_frac:.1%} of targets are induction tokens", flush=True)


@torch.no_grad()
def nll_split(zl):
    ind, oth = [], []
    for s, m in zip(seqs, masks):
        with hook_ctx([Zero(L) for L in zl]):
            lg = R.model(s[None].to(R.device), use_cache=False).logits[0, :-1]
        nll = torch.nn.functional.cross_entropy(lg.float(), s[1:].to(R.device), reduction="none").cpu()
        ind.append(nll[m]); oth.append(nll[~m])
    return float(torch.cat(ind).mean()), float(torch.cat(oth).mean())


bi, bo = nll_split([])
print(f"  intact: NLL induction {bi:.3f}  other {bo:.3f}", flush=True)
rows.append(dict(section="D", cond="intact", nll_ind=bi, nll_oth=bo, ind_frac=ind_frac))
for gn, zl in GROUPS.items():
    ni, no = nll_split(zl)
    rows.append(dict(section="D", cond=f"remove {gn}", nll_ind=ni, nll_oth=no,
                     d_ind=ni - bi, d_oth=no - bo))
    print(f"  remove {gn}: dNLL induction {ni - bi:+.3f}  other {no - bo:+.3f}  "
          f"ratio {(ni - bi) / max(no - bo, 1e-6):6.2f}", flush=True)

json.dump(dict(meta=vars(a), rows=rows), open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
