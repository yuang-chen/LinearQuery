"""Step 5: does the pathway hold up?

Runs a *fixed* probe battery (candidates frozen from steps 3-4) over several conditions:
  fresh      : 200 unseen example pairs, same setting as discovery
  distance   : target entry moved early/late in the list (target->query distance)
  distractors: 4 / 8 / 16 dictionary entries
  template   : a second prompt template

Probes: writer sufficiency (GDN-0 out@value), the state-patch reference, the reader heads'
V-path (head-path V@src) and layer-level path block/open at the two reader layers.
"""
import sys, json, argparse, statistics as st, torch
sys.path.insert(0, "/user/yac/LinearAblation")
from src.task import make_examples, verify_single_token
from src.runner import Runner, OutPatch, Capture, HeadPatch, hook_ctx
from src.qkv import ProjPatch, ProjCapture
from src.harness import group_examples, recovery

ap = argparse.ArgumentParser()
ap.add_argument("--writer_layer", type=int, default=0)
ap.add_argument("--heads", default="15:5,19:1,19:5")
ap.add_argument("--reader_layers", default="15,19")
ap.add_argument("--out", default="results/exp5_stability.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
HD, NH, NKV = R.cfg.head_dim, R.cfg.num_attention_heads, R.cfg.num_key_value_heads
GRP = NH // NKV
HEADS = [tuple(int(x) for x in s.split(":")) for s in a.heads.split(",")]
RLAYERS = [int(x) for x in a.reader_layers.split(",")]

# condition -> list of (n_pairs, target_pos, distract_pos, template, seed)
CONDS = {}
CONDS["fresh (8 pairs, 200 ex)"] = [(8, t, d, "chat", 300 + i)
                                    for i, (t, d) in enumerate([(2, 5), (5, 2), (1, 6), (6, 1),
                                                                (3, 7), (7, 3), (0, 4), (4, 0)])]
for t in (0, 3, 7):
    CONDS[f"distance: target_pos={t}"] = [(8, t, (t + 4) % 8, "chat", 400 + t)]
for npair in (4, 8, 16):
    CONDS[f"distractors: n_pairs={npair}"] = [(npair, 1, npair - 2, "chat", 500 + npair)]
CONDS["template: chat_alt"] = [(8, t, d, "chat_alt", 600 + i)
                               for i, (t, d) in enumerate([(2, 5), (5, 2), (1, 6), (6, 1)])]

N_PER = 25
rows = []
for cname, specs in CONDS.items():
    exs = []
    for (npair, t, d, tmpl, seed) in specs:
        exs += make_examples(N_PER, npair, seed=seed, template=tmpl, target_pos=t, distract_pos=d)
    nver = sum(verify_single_token(R.tok, e)[0] for e in exs)
    assert nver == len(exs), (cname, nver, len(exs))
    groups = group_examples(R, exs)
    acc_c = acc_x = ntot = 0
    for g in groups:
        tv, td, fin = g.pos["target_value"], g.pos["target_delim"], g.pos["final"]
        allpos = list(range(g.pos["n_tokens"]))
        lc = R.forward(g.clean_ids).logits[:, -1]
        lx = R.forward(g.corr_ids).logits[:, -1]
        d_clean, d_corr = g.D(lc), g.D(lx)
        acc_c += (lc.argmax(-1) == g.aid).sum().item()
        acc_x += (lx.argmax(-1) == g.cid).sum().item()
        ntot += len(g.exs)
        rec = lambda dp: recovery(dp, d_corr, d_clean).tolist()

        def put(kind, layer, head, r, **kw):
            rows.append(dict(cond=cname, kind=kind, layer=layer, head=head, rec=r,
                             n_pairs=len(g.exs[0].pairs), target_idx=g.exs[0].target_idx,
                             dist_tokens=fin - tv, **kw))

        put("D_clean", None, None, d_clean.tolist())
        put("D_corr", None, None, d_corr.tolist())

        W = R.mixer(a.writer_layer)
        cw = Capture(W)
        with hook_ctx([cw]):
            R.forward(g.clean_ids)
        writer = lambda: OutPatch(W, [tv], cw.value)
        put("writer out@value", a.writer_layer, None,
            rec(g.D(R.forward(g.corr_ids, hooks=[writer()]).logits[:, -1])))
        states = R.recurrent_states(g.clean_ids, tv)
        put("state@t ALL (reference)", -1, None,
            rec(g.D(R.split_forward(g.corr_ids, tv, state_patch=states))))

        for L in RLAYERS:
            attn = R.mixer(L)
            cv, cv2 = ProjCapture(attn, "v"), ProjCapture(attn, "v")
            with hook_ctx([cv]):
                R.forward(g.corr_ids)
            v_corr = cv.value
            with hook_ctx([cv2, writer()]):
                R.forward(g.corr_ids)
            v_w = cv2.value
            put("pathopen V@src", L, None, rec(g.D(R.forward(
                g.corr_ids, hooks=[ProjPatch(attn, "v", [tv], v_w, HD)]).logits[:, -1])))
            put("pathblock V@src", L, None, rec(g.D(R.forward(
                g.corr_ids, hooks=[writer(), ProjPatch(attn, "v", [tv], v_corr, HD)]).logits[:, -1])))

        for (L, h) in HEADS:
            attn = R.mixer(L)
            kv = h // GRP
            cv = ProjCapture(attn, "v")
            with hook_ctx([cv, writer()]):
                R.forward(g.corr_ids)
            cap = Capture(attn.o_proj, mode="in")
            with hook_ctx([ProjPatch(attn, "v", [tv], cv.value, HD, heads=[kv]), cap]):
                R.forward(g.corr_ids)
            d = g.D(R.forward(g.corr_ids,
                              hooks=[HeadPatch(attn, [h], allpos, cap.value, HD)]).logits[:, -1])
            put("head-path V@src", L, h, rec(d))
    print(f"{cname}: acc_clean={acc_c/ntot:.2f} acc_corr={acc_x/ntot:.2f} n={ntot}", flush=True)

json.dump(dict(meta=vars(a), rows=rows), open(a.out, "w"), indent=2)

agg = {}
for r in rows:
    agg.setdefault((r["cond"], r["kind"], r["layer"], r["head"]), []).extend(r["rec"])
print(f"\n{'condition':28s} {'probe':26s} {'L':>3s} {'H':>3s}   mean    sem")
for (c, k, L, h), v in agg.items():
    print(f"{c:28s} {k:26s} {str(L):>3s} {str(h):>3s}  {sum(v)/len(v):+.3f}  {st.pstdev(v)/len(v)**.5:.3f}")
