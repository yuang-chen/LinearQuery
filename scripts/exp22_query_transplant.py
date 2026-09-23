"""Step 22: query transplant -- is the group's write at the question *the query*?

Parts X-XVI infer "group G builds the reader's query" from damage: ablate G, the reader stops
aiming at the target. That is necessity. This is the sufficiency test.

Two prompts share one dictionary and differ only in the queried key ("What is the value of
banana?" vs "... of mango?"), so they have identical token length and identical entry positions.
Run A, record what each GDN layer of group G writes to the residual at chosen positions, then
run B with those writes substituted. If G carries the query, run B should retrieve *A's* entry:
the reader's attention should move to entry A and the model should answer A's value.

Position sets (all measured separately):
  final      the last token only
  question   every token of the question + assistant prefix (where the query is assembled)
  dict       the dictionary span only (control: the query is not built there)
  all        every position

Groups G0..Gk and the reader head come from the model, as in exp21; other groups are the
controls for the query-building one.
"""
import sys, json, argparse, time, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from src.runner import Runner, hook_ctx
from src.qkv import AttnWeights
from src.gen import Builder, chat_wrap
from src.task import KEY_WORDS, VALUE_WORDS

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/user/yac/LinearSwap/models/Qwen3.5-0.8B")
ap.add_argument("--tag", default="0.8B")
ap.add_argument("--variant", default="chat8", help="chat8 | list8 | rev8")
ap.add_argument("--reader", default="", help="reader head 'L,H' (default: from the exp21 run)")
ap.add_argument("--group_size", type=int, default=3)
ap.add_argument("--layers", default="", help="layer-resolved transplant: treat each listed "
                                             "layer as its own group, e.g. 12,13,14")
ap.add_argument("--out", default="")
ap.add_argument("--n_pairs", type=int, default=8)
ap.add_argument("--per_cfg", type=int, default=25)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
OUT = a.out or f"results/exp22_transplant_{a.tag}_{a.variant}.json"

t0 = time.time()
R = Runner(model_path=a.model, dtype=torch.float32)
R.model.config.get_text_config()._attn_implementation = "eager"
tok = R.tok
GROUPS = ([[L] for L in map(int, a.layers.split(","))] if a.layers
          else R.groups(a.group_size))
WRAP = chat_wrap(tok)
NQ = R.cfg.num_attention_heads

if a.reader:
    RL, RH = map(int, a.reader.split(","))
else:                                   # reuse the reader exp21 found for this model
    d = json.load(open(f"results/exp21/{a.tag}_{a.variant}.json"))
    RL, RH = d["meta"]["reader"]
NAMES = [f"L{G[0]}" if a.layers else f"G{i}" for i, G in enumerate(GROUPS)]
print(f"[{a.tag} {a.variant}] groups {GROUPS}  reader L{RL}H{RH}", flush=True)


# ---------------------------------------------------------------- items
def render(pairs, q, style="chat"):
    """Prompt asking for key `q` in the chat8 / list8 / rev8 template (src/gen.py wording)."""
    b = Builder(); vs, es = [], []
    b.add(WRAP[0])
    if style == "list":
        b.add("Lookup table:\n")
        for k, v in pairs:
            e0, _ = b.add(f"* {k}="); v0, v1 = b.add(v); _, e1 = b.add("\n")
            vs.append((v0, v1)); es.append((e0, e1))
        qs0, _ = b.add(f"Which value does {q} map to?")
        b.add(WRAP[1] + f"The entry {q} maps to **")
    elif style == "rev":
        b.add("Dictionary:")
        for i, (k, v) in enumerate(pairs):
            e0, _ = b.add(" "); v0, v1 = b.add(v); _, e1 = b.add(f" is the value of {k}")
            b.add("," if i < len(pairs) - 1 else ".")
            vs.append((v0, v1)); es.append((e0, e1))
        qs0, _ = b.add(f"\nQuestion: What is the value of {q}?")
        b.add(WRAP[1] + f"The value of {q} is **")
    else:
        b.add("Dictionary:")
        for i, (k, v) in enumerate(pairs):
            e0, _ = b.add(f" {k}="); v0, v1 = b.add(v)
            _, e1 = b.add("," if i < len(pairs) - 1 else ".")
            vs.append((v0, v1)); es.append((e0, e1))
        qs0, _ = b.add(f"\nQuestion: What is the value of {q}?")
        b.add(WRAP[1] + f"The value of {q} is **")
    return b.s, vs, es, qs0


def spans_to_pos(off, span):
    return [i for i, (s, e) in enumerate(off) if s < span[1] and e > span[0]]


import random
rng = random.Random(a.seed)
vals = [v for v in VALUE_WORDS if len(tok(v).input_ids) == 1 and len(tok(" " + v).input_ids) == 1]
keys = [k for k in KEY_WORDS if len(tok(" " + k).input_ids) == 1]

cfgs = [(2, 5), (5, 2), (1, 6), (6, 1)]          # (A index, B index) in the dictionary
items = []
for ia, ib in cfgs:
    for _ in range(a.per_cfg):
        ks = rng.sample(keys, a.n_pairs); vs_ = rng.sample(vals, a.n_pairs)
        pairs = list(zip(ks, vs_))
        STYLE = {"list8": "list", "rev8": "rev"}.get(a.variant, "chat")
        ta, _, _, _ = render(pairs, ks[ia], STYLE)
        tb, vspans, espans, qs0 = render(pairs, ks[ib], STYLE)
        ea = tok(ta, return_offsets_mapping=True); eb = tok(tb, return_offsets_mapping=True)
        if len(ea.input_ids) != len(eb.input_ids):
            continue
        aid = tok(vs_[ia]).input_ids; bid = tok(vs_[ib]).input_ids   # "is **lime": no space
        if len(aid) != 1 or len(bid) != 1:
            continue
        off = eb.offset_mapping
        items.append(dict(ids_a=ea.input_ids, ids_b=eb.input_ids, ia=ia, ib=ib,
                          aid=aid[0], bid=bid[0], n=len(eb.input_ids),
                          entry=[spans_to_pos(off, s) for s in espans],
                          qstart=min(spans_to_pos(off, (qs0, qs0 + 1)))))

# batch items that share length, entry positions and the question start
buckets = {}
for it in items:
    k = (it["n"], tuple(map(tuple, it["entry"])), it["qstart"], it["ia"], it["ib"])
    buckets.setdefault(k, []).append(it)
B = []
for v in buckets.values():
    B.append(dict(ia=v[0]["ia"], ib=v[0]["ib"], n=v[0]["n"], entry=v[0]["entry"],
                  qstart=v[0]["qstart"],
                  ids_a=torch.tensor([x["ids_a"] for x in v], device=R.device),
                  ids_b=torch.tensor([x["ids_b"] for x in v], device=R.device),
                  aid=torch.tensor([x["aid"] for x in v], device=R.device),
                  bid=torch.tensor([x["bid"] for x in v], device=R.device)))
N = sum(len(b["aid"]) for b in B)
print(f"  {N} prompt pairs in {len(B)} batches, {B[0]['n']} tokens", flush=True)


# ---------------------------------------------------------------- hooks
class Capture:
    def __init__(self, L): self.L, self.value = L, None
    def register(self):
        def fn(m, args, out):
            self.value = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return R.mixer(self.L).register_forward_hook(fn)


class Transplant:
    """Overwrite a mixer's output at `positions` with the donor run's output."""
    def __init__(self, L, positions, donor): self.L, self.pos, self.donor = L, positions, donor
    def register(self):
        def fn(m, args, out):
            t = (out[0] if isinstance(out, tuple) else out).clone()
            t[:, self.pos] = self.donor[:, self.pos].to(t.dtype)
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        return R.mixer(self.L).register_forward_hook(fn)


def pos_set(b, which):
    n, qs = b["n"], b["qstart"]
    return {"final": [n - 1], "question": list(range(qs, n)),
            "dict": list(range(0, qs)), "all": list(range(n))}[which]


@torch.no_grad()
def run(ids, b, hooks=()):
    aw = AttnWeights(R.mixer(RL))
    with hook_ctx(list(hooks) + [aw]):
        lg = R.model(ids, use_cache=False).logits[:, -1].float()
    at = aw.value[:, RH, b["n"] - 1].float()
    ent = torch.stack([at[:, e].sum(-1) for e in b["entry"]], 1)
    return lg, ent


def score(lg, ent, b):
    """answer-A rate, answer-B rate, logit(A)-logit(B), reader attention on entries A and B."""
    am = lg.argmax(-1)
    return dict(ansA=float((am == b["aid"]).float().mean()),
                ansB=float((am == b["bid"]).float().mean()),
                dAB=float((lg.gather(1, b["aid"][:, None]) - lg.gather(1, b["bid"][:, None])).mean()),
                attA=float(ent[:, b["ia"]].mean()), attB=float(ent[:, b["ib"]].mean()))


def agg(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


res = dict(meta=dict(model=a.model, tag=a.tag, reader=[RL, RH], groups=GROUPS, n=N,
                     tokens=B[0]["n"], n_pairs=a.n_pairs, seed=a.seed))

# ---------------------------------------------------------------- baselines
base_a, base_b = [], []
for b in B:
    la, ea = run(b["ids_a"], b); base_a.append(score(la, ea, b))
    lb, eb = run(b["ids_b"], b); base_b.append(score(lb, eb, b))
res["baseline"] = dict(run_A=agg(base_a), run_B=agg(base_b))
print(f"A1 run A (asks A): ansA {res['baseline']['run_A']['ansA']:.2f}  attA "
      f"{res['baseline']['run_A']['attA']:.2f}\n"
      f"A1 run B (asks B): ansA {res['baseline']['run_B']['ansA']:.2f}  ansB "
      f"{res['baseline']['run_B']['ansB']:.2f}  attA {res['baseline']['run_B']['attA']:.2f}  attB "
      f"{res['baseline']['run_B']['attB']:.2f}", flush=True)

# ---------------------------------------------------------------- transplants
res["transplant"] = []
for gi, G in enumerate(GROUPS):
    if not G:
        continue
    for which in ("final", "question", "dict", "all"):
        rows = []
        for b in B:
            caps = [Capture(L) for L in G]
            with hook_ctx(caps):
                R.model(b["ids_a"], use_cache=False)          # donor = run A
            p = pos_set(b, which)
            hooks = [Transplant(L, p, c.value) for L, c in zip(G, caps)]
            lg, ent = run(b["ids_b"], b, hooks)
            rows.append(score(lg, ent, b))
        r = agg(rows); r.update(group=gi, layers=G, positions=which)
        res["transplant"].append(r)
        print(f"A2 {NAMES[gi]:>4s} {which:9s} ansA {r['ansA']:.2f}  ansB {r['ansB']:.2f}  "
              f"D(A-B) {r['dAB']:+6.2f}  attA {r['attA']:.2f}  attB {r['attB']:.2f}", flush=True)

res["meta"]["seconds"] = time.time() - t0
import os
os.makedirs("results", exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"wrote {OUT}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
