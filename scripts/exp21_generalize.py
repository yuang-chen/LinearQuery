"""Step 21: generalisation battery -- do the Part IX-XV findings hold across tasks and scale?

For one (model, task variant) the circuit is re-discovered, not assumed:
  B1  baseline acc / gap
  B2  ablate each GDN group G0..Gk (3 GDN layers before each softmax layer)
  B3  ablate each single GDN layer
  B4  reader search: every softmax head's attention from the final token to the target entry
      (clean prompts); of the top 8, the head whose removal lowers the gap most is "the reader"
  B5  zero each value head of GDN layer 0 (is there a head-8-like single point of failure?)
  B6  q.k selectivity at the reader (Part XV design): Q at the final token and K at all positions
      from the intact or ablated run, 2x2, for the GDN group immediately before the reader layer
      ("Gr-1", the G3 analogue) and the one before it ("Gr-2", the G2 analogue).
      Entry score = log attention mass on the entry span (for a single-token value this is the
      pre-softmax score up to a shared constant). selectivity = s_target - mean s_other.
  B7  reader attention split (target entry / other entries / <|im_start|> or first token / rest)
      intact and with Gr-1 or Gr-2 removed.
MMLU variants keep only items the intact model answers correctly in both clean and corrupted
form (the retrieval question is then "which letter", not "which fact").
"""
import sys, json, argparse, time, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from src.runner import Runner, hook_ctx
from src.qkv import AttnWeights, ProjPatch, ProjCapture
from src.gen import make_items, batches

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/user/yac/LinearSwap/models/Qwen3.5-0.8B")
ap.add_argument("--tag", default="0.8B")
ap.add_argument("--variant", default="chat8")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--max_bs", type=int, default=25)
ap.add_argument("--mmlu_shots", type=int, default=5,
                help="few-shot examples in the standard MMLU prompt (0 = bare question)")
ap.add_argument("--reader", default="", help="force the B6/B7 reader head, e.g. 19,6")
ap.add_argument("--qk_groups", default="", help="GDN group indices for B6/B7, e.g. 4,3,2,1 "
                "(default: the two groups before the reader, named Gr-1 / Gr-2)")
a = ap.parse_args()
OUT = f"results/exp21/{a.tag}_{a.variant}.json"

t0 = time.time()
R = Runner(model_path=a.model, dtype=torch.float32)
R.model.config.get_text_config()._attn_implementation = "eager"
tok = R.tok
NH_GDN = R.cfg.linear_num_value_heads
VHD = R.cfg.linear_value_head_dim
HD = R.cfg.head_dim
NQ, NKV = R.cfg.num_attention_heads, R.cfg.num_key_value_heads
GROUPS = [[L for L in range(s - 3, s) if L in R.gdn_layers] for s in R.attn_layers]
print(f"[{a.tag} {a.variant}] groups {GROUPS}", flush=True)


class Zero:
    def __init__(self, L): self.L = L
    def register(self):
        return R.mixer(self.L).register_forward_hook(lambda m, a_, o: torch.zeros_like(o))


class HeadZero:
    def __init__(self, L, h): self.L, self.h = L, h
    def register(self):
        def fn(mod, args, kwargs):
            t = (args[0] if args else kwargs["input"]).clone()
            t[..., self.h * VHD:(self.h + 1) * VHD] = 0
            return (t,) + tuple(args[1:]), kwargs
        return R.mixer(self.L).out_proj.register_forward_pre_hook(fn, with_kwargs=True)


# ------------------------------------------------------------------ data
items = make_items(a.variant, tok, seed=0, mmlu_shots=a.mmlu_shots)
B = batches(tok, items, R.device, a.max_bs)


@torch.no_grad()
def run_batch(b, ids, hooks=()):
    with hook_ctx(hooks):
        return R.model(ids, use_cache=False).logits[:, -1]


if a.variant.startswith("mmlu"):                       # keep items solved in both forms
    keep = []
    for b in B:
        okc = run_batch(b, b.clean_ids).argmax(-1) == b.aid
        okx = run_batch(b, b.corr_ids).argmax(-1) == b.cid
        if bool((okc & okx).all()):
            keep.append(b)
        if len(keep) >= a.n:
            break
    n_scanned = B.index(keep[-1]) + 1 if keep else len(B)
    print(f"  mmlu: kept {len(keep)} of {n_scanned} scanned items (solved clean and corrupted)",
          flush=True)
    B = keep
else:
    n_scanned = None
N = sum(len(b.items) for b in B)
print(f"  {N} items in {len(B)} batches, ~{B[0].pos['n']} tokens", flush=True)


@torch.no_grad()
def evaluate(hooks_fn=lambda b: []):
    ok = tot = 0; gaps = []
    for b in B:
        ds = []
        for ids, want in ((b.clean_ids, b.aid), (b.corr_ids, b.cid)):
            lg = run_batch(b, ids, hooks_fn(b))
            ok += (lg.argmax(-1) == want).sum().item(); tot += ids.shape[0]
            ds.append(b.D(lg))
        gaps += (ds[0] - ds[1]).tolist()
    return dict(acc=ok / tot, gap=float(np.mean(gaps)), gap_sem=float(np.std(gaps) / len(gaps) ** .5))


res = dict(meta=dict(model=a.model, tag=a.tag, variant=a.variant, n=N, n_scanned=n_scanned,
                     tokens=B[0].pos["n"], groups=GROUPS))
base = evaluate()
res["baseline"] = base
print(f"B1 baseline acc {base['acc']:.3f} gap {base['gap']:+.2f}", flush=True)

res["groups"] = []
for gi, G in enumerate(GROUPS):
    r = evaluate(lambda b, G=G: [Zero(L) for L in G]); r.update(group=gi, layers=G)
    res["groups"].append(r)
print("B2 groups   " + "  ".join(f"G{r['group']}:{r['acc']:.2f}/{r['gap']:+.1f}" for r in res["groups"]),
      flush=True)

res["layers"] = []
for L in R.gdn_layers:
    r = evaluate(lambda b, L=L: [Zero(L)]); r.update(layer=L)
    res["layers"].append(r)
print("B3 layers   " + "  ".join(f"L{r['layer']}:{r['acc']:.2f}" for r in res["layers"]), flush=True)


# ------------------------------------------------------------------ B4 reader search
@torch.no_grad()
def attn_split(layer, hooks_fn=lambda b: [], ids_key="clean"):
    """Per head of `layer`: attention mass from the final token on target entry / other
    entries / first token, averaged over items."""
    acc = []
    for b in B:
        aw = AttnWeights(R.mixer(layer))
        with hook_ctx([aw] + hooks_fn(b)):
            R.model(b.clean_ids if ids_key == "clean" else b.corr_ids, use_cache=False)
        at = aw.value[:, :, b.pos["final"]].float()                 # [B, H, T]
        ep = b.pos["entry_pos"]
        tgt = at[:, :, ep[b.ti]].sum(-1)
        oth = sum(at[:, :, e].sum(-1) for j, e in enumerate(ep) if j != b.ti)
        acc.append(torch.stack([tgt, oth, at[:, :, 0]], -1).cpu())
    return torch.cat(acc).mean(0)                                    # [H, 3]


heads = []
for L in R.attn_layers:
    s = attn_split(L)
    for h in range(NQ):
        heads.append(dict(layer=L, head=h, target=float(s[h, 0]), others=float(s[h, 1]),
                          first=float(s[h, 2])))
heads.sort(key=lambda r: -r["target"])


class AttnHeadZero:
    """Zero one softmax head's output (its slice of the o_proj input)."""
    def __init__(self, L, h): self.L, self.h = L, h
    def register(self):
        def fn(mod, args, kwargs):
            t = (args[0] if args else kwargs["input"]).clone()
            t[..., self.h * HD:(self.h + 1) * HD] = 0
            return (t,) + tuple(args[1:]), kwargs
        return R.mixer(self.L).o_proj.register_forward_pre_hook(fn, with_kwargs=True)


# B4b: causal check of the top attention heads -- the reader is the head among the top 8 by
# attention whose removal lowers the gap most (attention alone can pick a non-causal head)
for r in heads[:8]:
    e = evaluate(lambda b, r=r: [AttnHeadZero(r["layer"], r["head"])])
    r.update(zero_acc=e["acc"], zero_gap=e["gap"])
res["reader_heads"] = heads[:10]
cands = sorted(heads[:8], key=lambda r: r["zero_gap"])
RL, RH = cands[0]["layer"], cands[0]["head"]
if a.reader:
    RL, RH = map(int, a.reader.split(","))
print("B4 readers  " + "  ".join(f"L{r['layer']}H{r['head']}:{r['target']:.2f}(zero gap {r['zero_gap']:+.1f})"
                                  for r in heads[:8]) + f"  -> reader L{RL}H{RH}", flush=True)

# ------------------------------------------------------------------ B5 layer-0 heads
res["l0_heads"] = []
for h in range(NH_GDN):
    r = evaluate(lambda b, h=h: [HeadZero(0, h)]); r.update(head=h)
    res["l0_heads"].append(r)
worst = sorted(res["l0_heads"], key=lambda r: r["gap"])[:4]
print("B5 L0 heads worst " + "  ".join(f"h{r['head']}:{r['acc']:.2f}/{r['gap']:+.1f}" for r in worst),
      flush=True)

# ------------------------------------------------------------------ B6 q.k selectivity
gi_r = R.attn_layers.index(RL)
G_r1 = GROUPS[gi_r] if gi_r >= 0 else []
G_r2 = GROUPS[gi_r - 1] if gi_r >= 1 else []
ATTN = R.mixer(RL)
QK_GROUPS = ([(f"G{i}", GROUPS[i]) for i in map(int, a.qk_groups.split(","))] if a.qk_groups
             else [("Gr-1", G_r1), ("Gr-2", G_r2)])


@torch.no_grad()
def cap_qk(ids, zl):
    cq, ck = ProjCapture(ATTN, "q"), ProjCapture(ATTN, "k")
    with hook_ctx([cq, ck] + [Zero(L) for L in zl]):
        R.model(ids, use_cache=False)
    return cq.value, ck.value


@torch.no_grad()
def selectivity(qsrc, ksrc, DON):
    sel, mar, top = [], [], []
    for bi, b in enumerate(B):
        n = b.pos["n"]; aw = AttnWeights(ATTN)
        hooks = [aw, ProjPatch(ATTN, "q", [n - 1], DON[(bi, qsrc)][0], HD),
                 ProjPatch(ATTN, "k", list(range(n)), DON[(bi, ksrc)][1], HD)]
        with hook_ctx(hooks):
            R.model(b.clean_ids, use_cache=False)
        at = aw.value[:, RH, b.pos["final"]].float()
        s = torch.stack([at[:, e].sum(-1).clamp_min(1e-30).log() for e in b.pos["entry_pos"]], 1)
        t = s[:, b.ti]; o = torch.cat([s[:, :b.ti], s[:, b.ti + 1:]], 1)
        sel += (t - o.mean(1)).tolist(); mar += (t - o.max(1).values).tolist()
        top += (t > o.max(1).values).float().tolist()
    return dict(selectivity=float(np.mean(sel)), sel_sem=float(np.std(sel) / len(sel) ** .5),
                margin=float(np.mean(mar)), top1=float(np.mean(top)))


res["selectivity"] = []
for name, G in QK_GROUPS:
    if not G:
        continue
    DON = {}
    for bi, b in enumerate(B):
        DON[(bi, "intact")] = cap_qk(b.clean_ids, [])
        DON[(bi, name)] = cap_qk(b.clean_ids, G)
    for qs in ("intact", name):
        for ks in ("intact", name):
            r = selectivity(qs, ks, DON); r.update(group=name, layers=G, q=qs, k=ks)
            res["selectivity"].append(r)
    del DON
    rs = [r for r in res["selectivity"] if r["group"] == name]
    print(f"B6 {name} {G} sel: " + "  ".join(f"Q{r['q'][:2]}K{r['k'][:2]}={r['selectivity']:.2f}"
                                          f"(top1 {r['top1']:.2f})" for r in rs), flush=True)

# ------------------------------------------------------------------ B7 attention split
res["reader_split"] = {}
for name, G in [("intact", [])] + QK_GROUPS:
    s = attn_split(RL, lambda b, G=G: [Zero(L) for L in G])[RH]
    res["reader_split"][name] = dict(target=float(s[0]), others=float(s[1]), first=float(s[2]),
                                     rest=float(1 - s.sum()))
print("B7 split    " + "  ".join(f"{k}: t{v['target']:.2f} o{v['others']:.2f} f{v['first']:.2f}"
                                  for k, v in res["reader_split"].items()), flush=True)

res["meta"].update(reader=[RL, RH], G_r1=G_r1, G_r2=G_r2, qk_groups=QK_GROUPS, seconds=time.time() - t0)
import os
os.makedirs("results/exp21", exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"wrote {OUT}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
