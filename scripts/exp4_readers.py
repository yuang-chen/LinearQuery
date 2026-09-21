"""Step 4: find the modules downstream of the writer that read its write.

Writer (default): GDN layer 0, output patched at the source value token (clean -> corrupted).

A. Module screen.  For each downstream mixer / MLP M: run the writer-patched corrupted
   forward while *blocking* M (freezing M's output to its plain-corrupted values at all
   positions).  Mediation = r_writer - r_writer&blocked.  Control = the same block applied
   without the writer patch, and in the clean run (general reader importance).
B. Softmax-specific: restore K / V at the source position to their corrupted values inside
   the writer-patched run (path block), and transplant them from the writer-patched run
   into the plain corrupted run (path open).
C. Head resolution for the strongest softmax layer (per-head o_proj input patches).
D. Q vs K vs V localisation for the strongest head.
"""
import sys, json, argparse, torch
sys.path.insert(0, "/user/yac/LinearAblation")
from src.task import make_examples
from src.runner import Runner, OutPatch, Capture, HeadPatch, hook_ctx
from src.qkv import ProjPatch, ProjCapture
from src.harness import group_examples, recovery

ap = argparse.ArgumentParser()
ap.add_argument("--n_per_config", type=int, default=25)
ap.add_argument("--n_pairs", type=int, default=8)
ap.add_argument("--configs", default="2:5,5:2,1:6,6:1")
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--writer_layer", type=int, default=0)
ap.add_argument("--out", default="results/exp4_readers.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
HD = R.cfg.head_dim
NH = R.cfg.num_attention_heads
NKV = R.cfg.num_key_value_heads
cfgs = [tuple(int(x) for x in c.split(":")) for c in a.configs.split(",")]
exs = []
for i, (ti, di) in enumerate(cfgs):
    exs += make_examples(a.n_per_config, a.n_pairs, seed=a.seed + i, target_pos=ti, distract_pos=di)
groups = group_examples(R, exs)
rows = []


def add(kind, layer, head, rec_list, extra=None):
    r = dict(kind=kind, layer=layer, head=head, rec=rec_list)
    if extra:
        r.update(extra)
    rows.append(r)


for g in groups:
    tv, td = g.pos["target_value"], g.pos["target_delim"]
    fin = g.pos["final"]
    with torch.no_grad():
        d_clean = g.D(R.forward(g.clean_ids).logits[:, -1])
        d_corr = g.D(R.forward(g.corr_ids).logits[:, -1])
    rec = lambda dp: recovery(dp, d_corr, d_clean).tolist()

    # donors ------------------------------------------------------------------
    W = R.mixer(a.writer_layer)
    cap_w = Capture(W)
    with hook_ctx([cap_w]):
        R.forward(g.clean_ids)
    writer_hook = lambda: OutPatch(W, [tv], cap_w.value)

    mods = {}                       # (kind, layer) -> module, out_index
    for L in range(a.writer_layer + 1, R.n_layers):
        mods[("mixer", L)] = (R.mixer(L), 0 if R.layer_types[L] == "full_attention" else None)
        mods[("mlp", L)] = (R.mlp(L), None)
    caps = {k: Capture(m, oi) for k, (m, oi) in mods.items()}
    with hook_ctx(list(caps.values())):
        R.forward(g.corr_ids)            # plain-corrupted donors, for blocking
    corr_don = {k: caps[k].value for k in mods}

    d_w = g.D(R.forward(g.corr_ids, hooks=[writer_hook()]).logits[:, -1])
    add("writer only", a.writer_layer, None, rec(d_w))

    # --- A. module blocking screen -------------------------------------------
    allpos = list(range(g.pos["n_tokens"]))
    for k, (m, oi) in mods.items():
        blk = OutPatch(m, allpos, corr_don[k], out_index=oi)
        d_wb = g.D(R.forward(g.corr_ids, hooks=[writer_hook(), blk]).logits[:, -1])
        add(f"writer+block {k[0]}", k[1], None, rec(d_wb))
        # control: same block with no writer patch, in the clean run
        cap_c = Capture(m, oi)
        with hook_ctx([cap_c]):
            R.forward(g.clean_ids)
        blk_c = OutPatch(m, allpos, corr_don[k], out_index=oi)
        d_cb = g.D(R.forward(g.clean_ids, hooks=[blk_c]).logits[:, -1])
        add(f"cleanblock {k[0]}", k[1], None, rec(d_cb))

    # --- B. softmax K/V at the source position -------------------------------
    for L in R.attn_layers:
        attn = R.mixer(L)
        capk, capv, capq = ProjCapture(attn, "k"), ProjCapture(attn, "v"), ProjCapture(attn, "q")
        with hook_ctx([capk, capv, capq]):
            R.forward(g.corr_ids)
        k_corr, v_corr, q_corr = capk.value, capv.value, capq.value
        capk2, capv2, capq2 = ProjCapture(attn, "k"), ProjCapture(attn, "v"), ProjCapture(attn, "q")
        with hook_ctx([capk2, capv2, capq2, writer_hook()]):
            R.forward(g.corr_ids)
        k_w, v_w, q_w = capk2.value, capv2.value, capq2.value

        def P(which, donor, ps, heads=None):
            return ProjPatch(attn, which, ps, donor, HD, heads=heads)

        # path block: inside the writer-patched run, restore K/V at the source to corrupted
        for name, hooks in (
            ("pathblock K@src", [writer_hook(), P("k", k_corr, [tv])]),
            ("pathblock V@src", [writer_hook(), P("v", v_corr, [tv])]),
            ("pathblock KV@src", [writer_hook(), P("k", k_corr, [tv]), P("v", v_corr, [tv])]),
            ("pathblock KV@src+delim", [writer_hook(), P("k", k_corr, [tv, td]), P("v", v_corr, [tv, td])]),
        ):
            add(name, L, None, rec(g.D(R.forward(g.corr_ids, hooks=hooks).logits[:, -1])))
        # path open: transplant writer-run K/V (or Q) into the plain corrupted run
        for name, hooks in (
            ("pathopen K@src", [P("k", k_w, [tv])]),
            ("pathopen V@src", [P("v", v_w, [tv])]),
            ("pathopen KV@src", [P("k", k_w, [tv]), P("v", v_w, [tv])]),
            ("pathopen KV@src+delim", [P("k", k_w, [tv, td]), P("v", v_w, [tv, td])]),
            ("pathopen Q@final", [P("q", q_w, [fin])]),
            ("pathopen QKV@all", [P("k", k_w, allpos), P("v", v_w, allpos), P("q", q_w, allpos)]),
        ):
            add(name, L, None, rec(g.D(R.forward(g.corr_ids, hooks=hooks).logits[:, -1])))

        # --- C. head resolution (block each head's contribution) -------------
        caph = Capture(attn.o_proj, mode="in")
        with hook_ctx([caph]):
            R.forward(g.corr_ids)
        head_corr = caph.value
        for h in range(NH):
            hp = HeadPatch(attn, [h], allpos, head_corr, HD)
            d = g.D(R.forward(g.corr_ids, hooks=[writer_hook(), hp]).logits[:, -1])
            add("writer+block head", L, h, rec(d))
        # per-head path open: writer-run K/V at src but only for the KV head feeding head h
        caph2 = Capture(attn.o_proj, mode="in")
        with hook_ctx([caph2, writer_hook()]):
            R.forward(g.corr_ids)
        head_w = caph2.value
        for h in range(NH):
            hp = HeadPatch(attn, [h], allpos, head_w, HD)
            d = g.D(R.forward(g.corr_ids, hooks=[hp]).logits[:, -1])
            add("pathopen head", L, h, rec(d))
    print(f"group t={tv} done", flush=True)

json.dump(dict(meta=vars(a), rows=rows), open(a.out, "w"), indent=2)

import statistics as st
agg = {}
for r in rows:
    agg.setdefault((r["kind"], r["layer"], r["head"]), []).extend(r["rec"])
base = sum(agg[("writer only", a.writer_layer, None)]) / len(agg[("writer only", a.writer_layer, None)])
print(f"\nwriter-only recovery = {base:+.3f}")
print("kind                       layer head  mean_rec   sem")
for (k, L, h), v in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] if x[0][2] is not None else -1)):
    m = sum(v) / len(v)
    if abs(m) > 0.03 or abs(m - base) > 0.05:
        print(f"{k:26s} {L:4d} {str(h):4s}  {m:+.3f}  {st.pstdev(v)/len(v)**.5:.3f}")
