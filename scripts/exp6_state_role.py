"""Step 6: what does the GDN recurrent state actually do?

The step-3 null result (all-layer state patch recovers 2%) is ambiguous between
  (a) the value is never written into the state,
  (b) it is written but decays before the question is reached,
  (c) it is present and readable, but the softmax route supplies the answer first.
These three are separated here.

A. MEMORY HORIZON.  Recompute the GDN forget gate g_t = -exp(A_log) * softplus(a_t + dt_bias)
   on real task prompts; the retention of a write made at t and read at T is exp(sum_{t<s<=T} g_s).
   Gives a per-layer/head effective half-life in tokens.  Distinguishes (a)/(b) from (c).

B. STATE DIVERGENCE.  delta = ||S_clean - S_corrupted||_F / ||S_clean||_F, measured at three
   cut points (just after the target value, at the end of the dictionary, at the final token).
   If delta stays large the binding IS in the state -> world (c); if it collapses -> (b);
   if it is ~0 even at the write position -> (a).

C. ROUTE ISOLATION.  Cut one route and see whether the task still gets solved.
     state route  : run the CLEAN prompt but replace K and V over the whole dictionary span,
                    in every softmax layer, with the CORRUPTED run's -- the softmax layers
                    can no longer see the true bindings, only the GDN state can.
     kv route     : the mirror image (corrupted prompt, clean K/V over the dictionary span).
   Recovery is on the usual [D_corr, D_clean] axis, so +1 = that route alone solves the task.
"""
import sys, json, argparse, statistics as st, torch
import torch.nn.functional as F
sys.path.insert(0, "/user/yac/LinearAblation")
from src.task import make_examples
from src.runner import Runner, Capture, hook_ctx
from src.qkv import ProjPatch, ProjCapture
from src.harness import group_examples, recovery

ap = argparse.ArgumentParser()
ap.add_argument("--n_per_config", type=int, default=25)
ap.add_argument("--n_pairs", type=int, default=8)
ap.add_argument("--configs", default="2:5,5:2,1:6,6:1")
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--out", default="results/exp6_state_role.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
HD = R.cfg.head_dim
cfgs = [tuple(int(x) for x in c.split(":")) for c in a.configs.split(",")]
exs = []
for i, (ti, di) in enumerate(cfgs):
    exs += make_examples(a.n_per_config, a.n_pairs, seed=a.seed + i, target_pos=ti, distract_pos=di)
groups = group_examples(R, exs)
rows = []

for g in groups:
    tv = g.pos["target_value"]
    dict_end = max(g.pos["value_pos"].values()) + 1      # the '.' closing the dictionary
    fin = g.pos["final"]
    d_clean = g.D(R.forward(g.clean_ids).logits[:, -1])
    d_corr = g.D(R.forward(g.corr_ids).logits[:, -1])
    rec = lambda dp: recovery(dp, d_corr, d_clean).tolist()

    # ---- A. forget gates on real prompts ------------------------------------
    caps = {L: Capture(R.mixer(L), mode="in") for L in R.gdn_layers}
    with hook_ctx(list(caps.values())):
        R.forward(g.clean_ids)
    for L in R.gdn_layers:
        m = R.mixer(L)
        h = caps[L].value
        aa = m.in_proj_a(h).float()                       # [B, T, n_heads]
        gl = -m.A_log.float().exp() * F.softplus(aa + m.dt_bias.float())   # log-decay per token
        # retention of a write made at the target value and read at the final token
        ret_fin = gl[:, tv + 1:fin + 1].sum(1).exp()      # [B, n_heads]
        ret_end = gl[:, tv + 1:dict_end + 1].sum(1).exp()
        half = (torch.log(torch.tensor(0.5)) / gl.mean(1)).clamp(max=1e6)  # tokens
        rows.append(dict(kind="gate", layer=L,
                         mean_log_decay=gl.mean().item(),
                         half_life_median=half.median().item(),
                         half_life_p90=half.flatten().quantile(0.9).item(),
                         retention_to_dict_end=ret_end.mean().item(),
                         retention_to_final=ret_fin.mean().item(),
                         retention_to_final_max_head=ret_fin.mean(0).max().item()))

    # ---- B. state divergence at three cut points ----------------------------
    for name, cut in (("after target value", tv), ("end of dictionary", dict_end), ("final token", fin)):
        sc = R.recurrent_states(g.clean_ids, cut)
        sx = R.recurrent_states(g.corr_ids, cut)
        for L in R.gdn_layers:
            A_, B_ = sc[L].float(), sx[L].float()
            num = (A_ - B_).flatten(1).norm(dim=1)
            den = A_.flatten(1).norm(dim=1)
            rows.append(dict(kind="state_div", cut=name, layer=L, dist=fin - cut,
                             delta=(num / den).mean().item()))

    # ---- C. route isolation -------------------------------------------------
    span = list(range(0, dict_end + 1))                   # everything up to and incl. the '.'
    don_c, don_x = {}, {}
    for L in R.attn_layers:
        attn = R.mixer(L)
        ck, cv = ProjCapture(attn, "k"), ProjCapture(attn, "v")
        with hook_ctx([ck, cv]):
            R.forward(g.clean_ids)
        don_c[L] = (ck.value, cv.value)
        ck, cv = ProjCapture(attn, "k"), ProjCapture(attn, "v")
        with hook_ctx([ck, cv]):
            R.forward(g.corr_ids)
        don_x[L] = (ck.value, cv.value)

    def kv_hooks(don):
        hs = []
        for L in R.attn_layers:
            attn = R.mixer(L)
            hs += [ProjPatch(attn, "k", span, don[L][0], HD),
                   ProjPatch(attn, "v", span, don[L][1], HD)]
        return hs

    # state route: clean prompt, corrupted K/V over the dictionary
    d_state = g.D(R.forward(g.clean_ids, hooks=kv_hooks(don_x)).logits[:, -1])
    # kv route: corrupted prompt, clean K/V over the dictionary
    d_kv = g.D(R.forward(g.corr_ids, hooks=kv_hooks(don_c)).logits[:, -1])
    # identity controls (donor from the same run) -- must be no-ops
    d_idc = g.D(R.forward(g.clean_ids, hooks=kv_hooks(don_c)).logits[:, -1])
    d_idx = g.D(R.forward(g.corr_ids, hooks=kv_hooks(don_x)).logits[:, -1])
    for nm, d in (("state route only", d_state), ("kv route only", d_kv),
                  ("identity ctrl (clean)", d_idc), ("identity ctrl (corrupted)", d_idx)):
        rows.append(dict(kind="route", name=nm, rec=rec(d), D=d.tolist()))
    print(f"group t={tv}: state-route rec={recovery(d_state, d_corr, d_clean).mean():+.3f}  "
          f"kv-route rec={recovery(d_kv, d_corr, d_clean).mean():+.3f}", flush=True)

json.dump(dict(meta=vars(a), rows=rows), open(a.out, "w"), indent=2)

# ---- summary ---------------------------------------------------------------
print("\nA. memory horizon (per GDN layer, averaged over prompts)")
print("layer  median half-life  p90 half-life   retention value->dict-end   value->final  (best head)")
for L in R.gdn_layers:
    v = [r for r in rows if r["kind"] == "gate" and r["layer"] == L]
    f = lambda k: sum(x[k] for x in v) / len(v)
    print(f"{L:5d}  {f('half_life_median'):12.2f} tok {f('half_life_p90'):10.2f} tok "
          f"{f('retention_to_dict_end'):18.2e} {f('retention_to_final'):14.2e} "
          f"{f('retention_to_final_max_head'):.2e}")

print("\nB. relative state divergence ||S_clean - S_corr|| / ||S_clean||")
cuts = ["after target value", "end of dictionary", "final token"]
print("layer " + "".join(f"{c:>22s}" for c in cuts))
for L in R.gdn_layers:
    line = f"{L:5d}"
    for c in cuts:
        v = [r["delta"] for r in rows if r["kind"] == "state_div" and r["layer"] == L and r["cut"] == c]
        line += f"{sum(v)/len(v):22.4f}"
    print(line)

print("\nC. route isolation")
for nm in ("state route only", "kv route only", "identity ctrl (clean)", "identity ctrl (corrupted)"):
    v = [x for r in rows if r["kind"] == "route" and r["name"] == nm for x in r["rec"]]
    d = [x for r in rows if r["kind"] == "route" and r["name"] == nm for x in r["D"]]
    print(f"  {nm:28s} recovery {sum(v)/len(v):+.3f} ± {st.pstdev(v)/len(v)**.5:.3f}   "
          f"D {sum(d)/len(d):+.2f}")
