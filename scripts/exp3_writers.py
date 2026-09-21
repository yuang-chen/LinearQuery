"""Step 3: scan GDN layers for candidate 'writers' of the retrieved value.

Two intervention families, clean donor -> corrupted run:
  state@t      : patch layer L's recurrent state after the target value token, recompute suffix
  out@value    : patch layer L's GDN output at the target (source) value token
  out@delim    : patch layer L's GDN output at the delimiter right after the value
  out@val+delim: both output positions
Recovery = (D_patched - D_corr) / (D_clean - D_corr).
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
ap.add_argument("--dtype", default="fp32")
ap.add_argument("--out", default="results/exp3_writers.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16)
cfgs = [tuple(int(x) for x in c.split(":")) for c in a.configs.split(",")]
exs = []
for i, (ti, di) in enumerate(cfgs):
    exs += make_examples(a.n_per_config, a.n_pairs, seed=a.seed + i,
                         target_pos=ti, distract_pos=di)
groups = group_examples(R, exs)
print(f"{len(exs)} examples in {len(groups)} position groups", flush=True)

rows = []
for g in groups:
    pos = g.pos
    tv, td = pos["target_value"], pos["target_delim"]
    with torch.no_grad():
        d_clean = g.D(R.forward(g.clean_ids).logits[:, -1])
        d_corr = g.D(R.forward(g.corr_ids).logits[:, -1])
    # clean donors
    caps = {L: Capture(R.mixer(L)) for L in R.gdn_layers}
    with hook_ctx(list(caps.values())):
        R.forward(g.clean_ids)
    donors = {L: caps[L].value for L in R.gdn_layers}
    clean_states = R.recurrent_states(g.clean_ids, tv)

    def rec(dp):
        return recovery(dp, d_corr, d_clean).tolist()

    for L in R.gdn_layers:
        dp = g.D(R.split_forward(g.corr_ids, tv, state_patch={L: clean_states[L]}))
        rows.append(dict(cfg=(pos["target_value"], pos["distract_value"]), layer=L,
                         kind="state@t", rec=rec(dp)))
        for name, ps in (("out@value", [tv]), ("out@delim", [td]), ("out@val+delim", [tv, td])):
            dp = g.D(R.forward(g.corr_ids, hooks=[OutPatch(R.mixer(L), ps, donors[L])]).logits[:, -1])
            rows.append(dict(cfg=(pos["target_value"], pos["distract_value"]), layer=L,
                             kind=name, rec=rec(dp)))
    # references: all GDN layers at once
    dp = g.D(R.split_forward(g.corr_ids, tv, state_patch=clean_states))
    rows.append(dict(cfg=(tv, pos["distract_value"]), layer=-1, kind="state@t ALL", rec=rec(dp)))
    hooks = [OutPatch(R.mixer(L), [tv, td], donors[L]) for L in R.gdn_layers]
    dp = g.D(R.forward(g.corr_ids, hooks=hooks).logits[:, -1])
    rows.append(dict(cfg=(tv, pos["distract_value"]), layer=-1, kind="out ALL", rec=rec(dp)))
    print(f"group t={tv} d={pos['distract_value']}: D_clean={d_clean.mean():.2f} "
          f"D_corr={d_corr.mean():.2f}", flush=True)

json.dump(dict(meta=vars(a), rows=rows), open(a.out, "w"), indent=2)

# summary
import statistics as st
agg = {}
for r in rows:
    agg.setdefault((r["kind"], r["layer"]), []).extend(r["rec"])
print("\nkind              layer  mean_rec   sem")
for (k, L), v in sorted(agg.items(), key=lambda x: -abs(sum(x[1]) / len(x[1]))):
    m = sum(v) / len(v)
    sem = st.pstdev(v) / len(v) ** .5
    if abs(m) > 0.02:
        print(f"{k:16s} {L:5d}  {m:+.3f}  {sem:.3f}")
