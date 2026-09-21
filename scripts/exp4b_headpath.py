"""Step 4b: head-level Q/K/V decomposition for the strongest reader heads.

For a head h in softmax layer L (GQA: h reads KV head h // (NH//NKV)):
  1. build a *donor run* = plain corrupted forward with exactly one of
       V@src / K@src / Q@final / V@src+delim
     transplanted from the writer-patched run (that KV head only, for K/V);
  2. transplant head h's o_proj-input slice from that donor run into the plain
     corrupted run.
The resulting recovery is the effect that reaches the answer *through head h* and
*through that specific input*.  Also records the head's attention probability from
the final position to the source position (supplementary evidence only).
"""
import sys, json, argparse, torch
sys.path.insert(0, "/user/yac/LinearAblation")
from src.task import make_examples
from src.runner import Runner, OutPatch, Capture, HeadPatch, hook_ctx
from src.qkv import ProjPatch, ProjCapture, AttnWeights
from src.harness import group_examples, recovery

ap = argparse.ArgumentParser()
ap.add_argument("--n_per_config", type=int, default=25)
ap.add_argument("--n_pairs", type=int, default=8)
ap.add_argument("--configs", default="2:5,5:2,1:6,6:1")
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--writer_layer", type=int, default=0)
ap.add_argument("--heads", default="15:5,15:3,19:1,19:5,23:3")
ap.add_argument("--out", default="results/exp4b_headpath.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
R.model.config.get_text_config()._attn_implementation = "eager"
HD, NH, NKV = R.cfg.head_dim, R.cfg.num_attention_heads, R.cfg.num_key_value_heads
GRP = NH // NKV
targets = [tuple(int(x) for x in s.split(":")) for s in a.heads.split(",")]

cfgs = [tuple(int(x) for x in c.split(":")) for c in a.configs.split(",")]
exs = []
for i, (ti, di) in enumerate(cfgs):
    exs += make_examples(a.n_per_config, a.n_pairs, seed=a.seed + i, target_pos=ti, distract_pos=di)
groups = group_examples(R, exs)
rows = []

for g in groups:
    tv, td, fin = g.pos["target_value"], g.pos["target_delim"], g.pos["final"]
    allpos = list(range(g.pos["n_tokens"]))
    d_clean = g.D(R.forward(g.clean_ids).logits[:, -1])
    d_corr = g.D(R.forward(g.corr_ids).logits[:, -1])
    rec = lambda dp: recovery(dp, d_corr, d_clean).tolist()

    W = R.mixer(a.writer_layer)
    cw = Capture(W)
    with hook_ctx([cw]):
        R.forward(g.clean_ids)
    writer = lambda: OutPatch(W, [tv], cw.value)

    for (L, h) in targets:
        attn = R.mixer(L)
        kv = h // GRP
        ck, cv, cq, aw = ProjCapture(attn, "k"), ProjCapture(attn, "v"), ProjCapture(attn, "q"), AttnWeights(attn)
        with hook_ctx([ck, cv, cq, aw, writer()]):
            R.forward(g.corr_ids)
        k_w, v_w, q_w, aw_w = ck.value, cv.value, cq.value, aw.value
        aw2 = AttnWeights(attn)
        with hook_ctx([aw2]):
            R.forward(g.corr_ids)
        aw_c = aw2.value

        variants = {
            "V@src": [ProjPatch(attn, "v", [tv], v_w, HD, heads=[kv])],
            "V@src+delim": [ProjPatch(attn, "v", [tv, td], v_w, HD, heads=[kv])],
            "K@src": [ProjPatch(attn, "k", [tv], k_w, HD, heads=[kv])],
            "K@src+delim": [ProjPatch(attn, "k", [tv, td], k_w, HD, heads=[kv])],
            "Q@final": [ProjPatch(attn, "q", [fin], q_w, HD, heads=[h])],
            "QKV@all": [ProjPatch(attn, "k", allpos, k_w, HD, heads=[kv]),
                        ProjPatch(attn, "v", allpos, v_w, HD, heads=[kv]),
                        ProjPatch(attn, "q", allpos, q_w, HD, heads=[h])],
        }
        for name, hooks in variants.items():
            cap = Capture(attn.o_proj, mode="in")
            with hook_ctx(hooks + [cap]):
                R.forward(g.corr_ids)
            d = g.D(R.forward(g.corr_ids, hooks=[HeadPatch(attn, [h], allpos, cap.value, HD)]).logits[:, -1])
            rows.append(dict(layer=L, head=h, kind=f"head-path {name}", rec=rec(d)))
        # supplementary: attention probability final -> source
        rows.append(dict(layer=L, head=h, kind="attn_prob final->src (corrupted)",
                         rec=aw_c[:, h, fin, tv].float().cpu().tolist()))
        rows.append(dict(layer=L, head=h, kind="attn_prob final->src (writer-patched)",
                         rec=aw_w[:, h, fin, tv].float().cpu().tolist()))
    print(f"group t={tv} done", flush=True)

json.dump(dict(meta=vars(a), rows=rows), open(a.out, "w"), indent=2)

import statistics as st
agg = {}
for r in rows:
    agg.setdefault((r["layer"], r["head"], r["kind"]), []).extend(r["rec"])
print("\nlayer head kind                                   mean     sem")
for (L, h, k), v in sorted(agg.items()):
    print(f"{L:5d} {h:4d} {k:38s} {sum(v)/len(v):+.3f}  {st.pstdev(v)/len(v)**.5:.3f}")
