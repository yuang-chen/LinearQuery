"""Step 32: how the Address block's output is assembled (paper Section 5). Donor/host pairs as in exp22
(same dictionary, different queried key); success = the host answers the donor's value.

  A  position:   transplant the Address block's mixer output over (i) the final token, (ii) the question
                 span including the final token, (iii) the question span *excluding* the final token
  B  state:      run donor and host up to the token before the final one, copy the donor's recurrent
                 state of a block's linear-attention layers into the host's cache (optionally with the
                 short-convolution state), then run the final token
  C  upstream:   transplant the Address block's final-token output from a donor whose upstream blocks
                 (G1, G2, G3, or all three) were removed, into an intact host
  D  input:      keep the host's recurrent state but give the Address block the donor's input at the final
                 token -- (i) each Address layer's mixer input, (ii) the residual stream entering the block
"""
import sys, json, argparse, time, random, torch
import numpy as np
sys.path.insert(0, "/mnt/yuang/LinearQuery")
from src.runner import Runner, hook_ctx
from src.qkv import AttnWeights
from src.gen import Builder, chat_wrap
from src.task import KEY_WORDS, VALUE_WORDS

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/mnt/yuang/models/Qwen3.5-9B")
ap.add_argument("--tag", default="9B")
ap.add_argument("--variant", default="chat8")
ap.add_argument("--reader", default="19,11")
ap.add_argument("--address", type=int, default=4, help="index of the Address block")
ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
ap.add_argument("--per_cfg", type=int, default=25)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--example", action="store_true", help="only save the reader's attention over the entries "
                "(host intact / Address removed / Address transplanted) for the paper's example figure")
a = ap.parse_args()
t0 = time.time()
R = Runner(model_path=a.model, dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16)
R.model.config.get_text_config()._attn_implementation = "eager"
tok, WRAP = R.tok, chat_wrap(R.tok)
RL, RH = map(int, a.reader.split(","))
G = R.groups(3); ADDR = G[a.address]
print(f"[{a.tag} {a.variant}] blocks {G}  Address G{a.address} {ADDR}  reader L{RL}H{RH}", flush=True)


# ---------------------------------------------------------------- items (exp22 construction)
def render(pairs, q, style):
    b = Builder(); es = []
    b.add(WRAP[0])
    if style == "list":
        b.add("Lookup table:\n")
        for k, v in pairs:
            e0, _ = b.add(f"* {k}="); b.add(v); _, e1 = b.add("\n"); es.append((e0, e1))
        qs0, _ = b.add(f"Which value does {q} map to?")
        b.add(WRAP[1] + f"The entry {q} maps to **")
    elif style == "rev":
        b.add("Dictionary:")
        for i, (k, v) in enumerate(pairs):
            e0, _ = b.add(" "); b.add(v); _, e1 = b.add(f" is the value of {k}")
            b.add("," if i < len(pairs) - 1 else "."); es.append((e0, e1))
        qs0, _ = b.add(f"\nQuestion: What is the value of {q}?")
        b.add(WRAP[1] + f"The value of {q} is **")
    else:
        b.add("Dictionary:")
        for i, (k, v) in enumerate(pairs):
            e0, _ = b.add(f" {k}="); b.add(v); _, e1 = b.add("," if i < len(pairs) - 1 else ".")
            es.append((e0, e1))
        qs0, _ = b.add(f"\nQuestion: What is the value of {q}?")
        b.add(WRAP[1] + f"The value of {q} is **")
    return b.s, es, qs0


span = lambda off, s: [i for i, (x, y) in enumerate(off) if x < s[1] and y > s[0]]
rng = random.Random(a.seed)
vals = [v for v in VALUE_WORDS if len(tok(v).input_ids) == 1 and len(tok(" " + v).input_ids) == 1]
keys = [k for k in KEY_WORDS if len(tok(" " + k).input_ids) == 1]
STYLE = {"list8": "list", "rev8": "rev"}.get(a.variant, "chat")
items = []
for ia, ib in [(2, 5), (5, 2), (1, 6), (6, 1)]:
    for _ in range(a.per_cfg):
        ks = rng.sample(keys, 8); vs = rng.sample(vals, 8); pairs = list(zip(ks, vs))
        ta, _, _ = render(pairs, ks[ia], STYLE); tb, es, qs0 = render(pairs, ks[ib], STYLE)
        ea, eb = tok(ta, return_offsets_mapping=True), tok(tb, return_offsets_mapping=True)
        aid, bid = tok(vs[ia]).input_ids, tok(vs[ib]).input_ids
        if len(ea.input_ids) != len(eb.input_ids) or len(aid) != 1 or len(bid) != 1:
            continue
        items.append(dict(ids_a=ea.input_ids, ids_b=eb.input_ids, ia=ia, ib=ib, aid=aid[0], bid=bid[0],
                          n=len(eb.input_ids), entry=[span(eb.offset_mapping, s) for s in es],
                          qstart=min(span(eb.offset_mapping, (qs0, qs0 + 1)))))
buckets = {}
for it in items:
    buckets.setdefault((it["n"], tuple(map(tuple, it["entry"])), it["qstart"], it["ia"], it["ib"]), []).append(it)
B = [dict(ia=v[0]["ia"], ib=v[0]["ib"], n=v[0]["n"], entry=v[0]["entry"], qstart=v[0]["qstart"],
          ids_a=torch.tensor([x["ids_a"] for x in v], device=R.device),
          ids_b=torch.tensor([x["ids_b"] for x in v], device=R.device),
          aid=torch.tensor([x["aid"] for x in v], device=R.device),
          bid=torch.tensor([x["bid"] for x in v], device=R.device)) for v in buckets.values()]
print(f"  {sum(len(b['aid']) for b in B)} pairs, {len(B)} batches", flush=True)


# ---------------------------------------------------------------- hooks
class Capture:
    def __init__(self, L): self.L, self.value = L, None
    def register(self):
        def fn(m, args, out):
            self.value = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return R.mixer(self.L).register_forward_hook(fn)


class Transplant:
    def __init__(self, L, pos, donor): self.L, self.pos, self.donor = L, pos, donor
    def register(self):
        def fn(m, args, out):
            t = (out[0] if isinstance(out, tuple) else out).clone()
            t[:, self.pos] = self.donor[:, self.pos].to(t.dtype)
            return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t
        return R.mixer(self.L).register_forward_hook(fn)


class InCapture:
    """input hidden states of module `mod` (all positions)"""
    def __init__(self, mod): self.mod, self.value = mod, None
    def register(self):
        def fn(m, args, kwargs):
            self.value = (kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]).detach().clone()
        return self.mod.register_forward_pre_hook(fn, with_kwargs=True)


class InPatch:
    """replace module `mod`'s input hidden states at `pos` with the donor's"""
    def __init__(self, mod, pos, donor): self.mod, self.pos, self.donor = mod, pos, donor
    def register(self):
        def fn(m, args, kwargs):
            key = "hidden_states" in kwargs
            t = (kwargs["hidden_states"] if key else args[0]).clone()
            t[:, self.pos] = self.donor[:, self.pos].to(t.dtype)
            if key:
                kwargs["hidden_states"] = t
                return args, kwargs
            return (t,) + tuple(args[1:]), kwargs
        return self.mod.register_forward_pre_hook(fn, with_kwargs=True)


@torch.no_grad()
def in_capture(ids, mods):
    caps = [InCapture(m) for m in mods]
    with hook_ctx(caps):
        R.model(ids, use_cache=False)
    return [c.value for c in caps]


class Zero:
    def __init__(self, L): self.L = L
    def register(self):
        return R.mixer(self.L).register_forward_hook(lambda m, a_, o: torch.zeros_like(o))


def score(lg, at, b):
    am = lg.argmax(-1)
    ent = torch.stack([at[:, e].sum(-1) for e in b["entry"]], 1)
    return dict(ansA=float((am == b["aid"]).float().mean()), ansB=float((am == b["bid"]).float().mean()),
                dAB=float((lg.gather(1, b["aid"][:, None]) - lg.gather(1, b["bid"][:, None])).mean()),
                attA=float(ent[:, b["ia"]].mean()), attB=float(ent[:, b["ib"]].mean()))


@torch.no_grad()
def full(ids, hooks=()):
    aw = AttnWeights(R.mixer(RL))
    with hook_ctx(list(hooks) + [aw]):
        lg = R.model(ids, use_cache=False).logits[:, -1].float()
    return lg, aw.value[:, RH, -1].float()


@torch.no_grad()
def capture(ids, layers, hooks=()):
    caps = [Capture(L) for L in layers]
    with hook_ctx(caps + list(hooks)):
        R.model(ids, use_cache=False)
    return [c.value for c in caps]


@torch.no_grad()
def state_swap(b, layers, conv=False):
    """donor/host prefixes up to n-2; copy the donor's recurrent (and conv) state of `layers`; run the final token"""
    up = b["n"] - 2
    ca, cb = R.run_prefix(b["ids_a"], up), R.run_prefix(b["ids_b"], up)
    for L in layers:
        cb.layers[L].recurrent_states[0] = ca.layers[L].recurrent_states[0].clone()
        if conv:
            cb.layers[L].conv_states[0] = ca.layers[L].conv_states[0].clone()
    aw = AttnWeights(R.mixer(RL))
    lg = R.run_suffix(b["ids_b"], up, cb, hooks=[aw]).float()
    return lg, aw.value[:, RH, -1].float()


def run(name, fn):
    rows = [score(*fn(b), b) for b in B]
    r = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}; r["name"] = name
    res["rows"].append(r)
    print(f"{name:42s} ansA {r['ansA']:.2f}  ansB {r['ansB']:.2f}  D(A-B) {r['dAB']:+6.2f}  "
          f"attA {r['attA']:.2f}  attB {r['attB']:.2f}", flush=True)


if a.example:
    @torch.no_grad()
    def entry_att(b, hooks):
        at = full(b["ids_b"], hooks)[1]
        return torch.stack([at[:, e].sum(-1) for e in b["entry"]], 1).cpu()     # [batch, 8]
    out, tot = {}, {}
    conds = {"intact": lambda b: [],
             "Address removed": lambda b: [Zero(L) for L in ADDR],
             "Address transplanted": lambda b: [Transplant(L, [b["n"] - 1], c) for L, c in
                                                zip(ADDR, capture(b["ids_a"], ADDR))]}
    for cname, hk in conds.items():
        rows = []
        for b in B:
            e = entry_att(b, hk(b))
            rows += [dict(att=e[i].tolist(), ia=b["ia"], ib=b["ib"]) for i in range(len(e))]
        out[cname] = rows[0]
        tot[cname] = dict(A=float(sum(r["att"][r["ia"]] for r in rows) / len(rows)),
                          B=float(sum(r["att"][r["ib"]] for r in rows) / len(rows)),
                          other=float(sum(sum(r["att"]) - r["att"][r["ia"]] - r["att"][r["ib"]] for r in rows) / len(rows)))
    it = items[0] if False else None
    b0 = B[0]
    ex = dict(prompt_host=tok.decode(b0["ids_b"][0]), prompt_donor=tok.decode(b0["ids_a"][0]),
              entries=[tok.decode(b0["ids_b"][0][e]) for e in b0["entry"]], example=out, mean=tot,
              reader=[RL, RH], address=ADDR)
    json.dump(ex, open(f"results/exp32_example_{a.tag}_{a.variant}.json", "w"), indent=1)
    print(json.dumps({k: [round(x, 2) for x in v["att"]] for k, v in out.items()}), "\n", tot, "\n", ex["entries"],
          "\nA =", b0["ia"], "B =", b0["ib"])
    sys.exit()

res = dict(meta=dict(model=a.model, tag=a.tag, variant=a.variant, reader=[RL, RH], blocks=G,
                     address=a.address, dtype=a.dtype), rows=[])
run("host (asks B)", lambda b: full(b["ids_b"]))
run("donor (asks A)", lambda b: full(b["ids_a"]))
run("split execution, no patch", lambda b: state_swap(b, []))

# A: positions
pos = {"final token": lambda b: [b["n"] - 1],
       "question span incl. final": lambda b: list(range(b["qstart"], b["n"])),
       "question span excl. final": lambda b: list(range(b["qstart"], b["n"] - 1))}
for pname, P in pos.items():
    run(f"A  output, {pname}", lambda b, P=P: full(
        b["ids_b"], [Transplant(L, P(b), c) for L, c in zip(ADDR, capture(b["ids_a"], ADDR))]))

# B: recurrent state just before the final token
for gi in (0, a.address - 1, a.address):
    run(f"B  state G{gi} {G[gi]}", lambda b, gi=gi: state_swap(b, G[gi]))
run(f"B  state+conv G{a.address}", lambda b: state_swap(b, ADDR, conv=True))
run("B  state, all linear-attention layers", lambda b: state_swap(b, R.gdn_layers))

# C: Address output from a donor with upstream blocks removed
ups = {f"G{i}": G[i] for i in range(1, a.address)}
ups[f"G1-G{a.address - 1}"] = [L for i in range(1, a.address) for L in G[i]]
for uname, layers in ups.items():
    run(f"C  output from donor without {uname}", lambda b, layers=layers: full(
        b["ids_b"], [Transplant(L, [b["n"] - 1], c) for L, c in
                     zip(ADDR, capture(b["ids_a"], ADDR, [Zero(x) for x in layers]))]))
    run(f"   (donor without {uname}: own answer)", lambda b, layers=layers: full(
        b["ids_a"], [Zero(x) for x in layers]))

# D: the donor's input to the Address block at the final token, host's own recurrent state
mix = [R.mixer(L) for L in ADDR]
run("D  input to each Address mixer, final token", lambda b: full(
    b["ids_b"], [InPatch(m, [b["n"] - 1], c) for m, c in zip(mix, in_capture(b["ids_a"], mix))]))
run("D  residual entering the block, final token", lambda b: full(
    b["ids_b"], [InPatch(R.layers[ADDR[0]], [b["n"] - 1], in_capture(b["ids_a"], [R.layers[ADDR[0]]])[0])]))

res["meta"]["seconds"] = time.time() - t0
json.dump(res, open(f"results/exp32_address_{a.tag}_{a.variant}.json", "w"), indent=1)
print(f"done ({(time.time() - t0) / 60:.1f} min)")
