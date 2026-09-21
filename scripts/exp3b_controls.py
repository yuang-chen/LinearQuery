"""Step 3b: controls for the writer scan.

  emb@value          : patch the token embedding at the source value position (trivial upper bound)
  out@value          : patch GDN layer L's output there (the candidate write)
  block-out@value    : *reverse* direction -- corrupted donor into the clean run (necessity)
  out@query / @final : GDN output patches on the query side, to catch query-formation writers
"""
import sys, json, argparse, torch
sys.path.insert(0, "/user/yac/LinearAblation")
from src.task import make_examples
from src.runner import Runner, OutPatch, Capture, hook_ctx
from src.harness import group_examples, recovery

ap = argparse.ArgumentParser()
ap.add_argument("--n_per_config", type=int, default=25)
ap.add_argument("--n_pairs", type=int, default=8)
ap.add_argument("--configs", default="2:5,5:2,1:6,6:1")
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--out", default="results/exp3b_controls.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
cfgs = [tuple(int(x) for x in c.split(":")) for c in a.configs.split(",")]
exs = []
for i, (ti, di) in enumerate(cfgs):
    exs += make_examples(a.n_per_config, a.n_pairs, seed=a.seed + i, target_pos=ti, distract_pos=di)
groups = group_examples(R, exs)
emb = R.model.model.embed_tokens if hasattr(R.model.model, "embed_tokens") else R.model.model.language_model.embed_tokens

rows = []
for g in groups:
    tv, td = g.pos["target_value"], g.pos["target_delim"]
    qk, fin = g.pos["query_key_last"], g.pos["final"]
    with torch.no_grad():
        d_clean = g.D(R.forward(g.clean_ids).logits[:, -1])
        d_corr = g.D(R.forward(g.corr_ids).logits[:, -1])

    def cap_all(ids, mods):
        caps = [Capture(m) for m in mods]
        with hook_ctx(caps):
            R.forward(ids)
        return [c.value for c in caps]

    mods = [emb] + [R.mixer(L) for L in R.gdn_layers] + [R.layers[L] for L in R.gdn_layers]
    clean_d = cap_all(g.clean_ids, mods)
    corr_d = cap_all(g.corr_ids, mods)
    n_g = len(R.gdn_layers)

    def add(kind, layer, dp, ref_corr=None, ref_clean=None):
        rc = recovery(dp, d_corr if ref_corr is None else ref_corr,
                      d_clean if ref_clean is None else ref_clean)
        rows.append(dict(cfg=[tv, g.pos["distract_value"]], layer=layer, kind=kind, rec=rc.tolist()))

    # trivial upper bound: clean embedding at the source value position
    dp = g.D(R.forward(g.corr_ids, hooks=[OutPatch(emb, [tv], clean_d[0])]).logits[:, -1])
    add("emb@value", -2, dp)
    # whole decoder-layer-0 block output at that position
    dp = g.D(R.forward(g.corr_ids, hooks=[OutPatch(R.layers[0], [tv], clean_d[1 + n_g])]).logits[:, -1])
    add("resid_after_layer0@value", 0, dp)

    for j, L in enumerate(R.gdn_layers):
        # sufficiency (clean -> corrupted), already in exp3, repeated here for a paired control
        dp = g.D(R.forward(g.corr_ids, hooks=[OutPatch(R.mixer(L), [tv], clean_d[1 + j])]).logits[:, -1])
        add("out@value", L, dp)
        # necessity: corrupted donor into the clean run (recovery measured on the reversed axis)
        dp = g.D(R.forward(g.clean_ids, hooks=[OutPatch(R.mixer(L), [tv], corr_d[1 + j])]).logits[:, -1])
        add("REV out@value", L, dp, ref_corr=d_clean, ref_clean=d_corr)
        # query-side writes
        for name, ps in (("out@query", [qk]), ("out@final", [fin]), ("out@query+final", [qk, fin])):
            dp = g.D(R.forward(g.corr_ids, hooks=[OutPatch(R.mixer(L), ps, clean_d[1 + j])]).logits[:, -1])
            add(name, L, dp)
    print(f"group t={tv}: D_clean={d_clean.mean():.2f} D_corr={d_corr.mean():.2f}", flush=True)

json.dump(dict(meta=vars(a), rows=rows), open(a.out, "w"), indent=2)

import statistics as st
agg = {}
for r in rows:
    agg.setdefault((r["kind"], r["layer"]), []).extend(r["rec"])
print("\nkind                        layer  mean_rec   sem")
for (k, L), v in sorted(agg.items(), key=lambda x: -abs(sum(x[1]) / len(x[1]))):
    m = sum(v) / len(v)
    if abs(m) > 0.015:
        print(f"{k:26s} {L:5d}  {m:+.3f}  {st.pstdev(v)/len(v)**.5:.3f}")
