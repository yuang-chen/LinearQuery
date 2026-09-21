"""Step 2: validate the intervention machinery.

Checks
  (a) split execution (prefix+cache, then suffix) reproduces ordinary inference
  (b) identity patches (clean donor into clean run) preserve predictions
  (c) caches/states are isolated between runs (no leakage across calls)
  (d) the stated caveat: patching a GDN *output* at position t does not repair the
      recurrent state that position t writes -- only a state patch does.
"""
import sys, json, argparse, torch
sys.path.insert(0, "/user/yac/LinearAblation")
from src.task import make_examples
from src.runner import Runner, OutPatch, Capture, hook_ctx
from src.positions import token_positions
from src.metrics import logit_diff

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--n_pairs", type=int, default=8)
ap.add_argument("--dtype", default="bf16")
ap.add_argument("--out", default="results/exp2_validate.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16); tok = R.tok
exs = make_examples(a.n, a.n_pairs, seed=7)
res = {"split_max_abs_logit_err": [], "split_argmax_match": [], "identity_state_dD": [],
       "identity_out_dD": [], "isolation_dD": [], "output_patch_vs_state_patch": []}

for e in exs:
    p = token_positions(tok, e)
    aid, cid = tok(e.answer).input_ids[0], tok(e.answer_corr).input_ids[0]
    ci, xi = R.encode(e.clean_prompt()), R.encode(e.corrupt_prompt())
    split = p["target_value"]

    # --- (a) split execution vs ordinary inference -------------------------------
    full = R.forward(ci).logits[0, -1].float()
    sp = R.split_forward(ci, split)[0].float()
    res["split_max_abs_logit_err"].append((full - sp).abs().max().item())
    res["split_argmax_match"].append(int(full.argmax().item() == sp.argmax().item()))

    d_clean = logit_diff(full, aid, cid)
    d_corr = logit_diff(R.forward(xi).logits[0, -1], aid, cid)

    # --- (b) identity patches ----------------------------------------------------
    # clean recurrent states patched into the clean run: must be a no-op
    st = R.recurrent_states(ci, split)
    d_id_state = logit_diff(R.split_forward(ci, split, state_patch=st)[0], aid, cid)
    res["identity_state_dD"].append(d_id_state - d_clean)

    # clean GDN output patched into the clean run at the target value position
    L = R.gdn_layers[len(R.gdn_layers) // 2]
    cap = Capture(R.mixer(L))
    with hook_ctx([cap]):
        R.model(ci)
    d_id_out = logit_diff(
        R.forward(ci, hooks=[OutPatch(R.mixer(L), [split], cap.value)]).logits[0, -1], aid, cid)
    res["identity_out_dD"].append(d_id_out - d_clean)

    # --- (c) isolation: a patched run must not contaminate the next plain run -----
    _ = R.split_forward(xi, split, state_patch=st)
    d_after = logit_diff(R.forward(xi).logits[0, -1], aid, cid)
    res["isolation_dD"].append(d_after - d_corr)

    # --- (d) output patch at t != state repair at t ------------------------------
    # donor = clean GDN output; patch it into the corrupted run at the target value
    # position, and compare with patching the clean recurrent state after t.
    d_outpatch = logit_diff(
        R.forward(xi, hooks=[OutPatch(R.mixer(L), [split], cap.value)]).logits[0, -1], aid, cid)
    d_statepatch = logit_diff(R.split_forward(xi, split, state_patch={L: st[L]})[0], aid, cid)
    res["output_patch_vs_state_patch"].append(
        dict(layer=L, d_clean=d_clean, d_corr=d_corr, d_outpatch=d_outpatch,
             d_statepatch=d_statepatch))

summ = {
    "n": a.n, "n_pairs": a.n_pairs,
    "split_max_abs_logit_err_max": max(res["split_max_abs_logit_err"]),
    "split_argmax_match": sum(res["split_argmax_match"]) / a.n,
    "identity_state_dD_max_abs": max(abs(x) for x in res["identity_state_dD"]),
    "identity_out_dD_max_abs": max(abs(x) for x in res["identity_out_dD"]),
    "isolation_dD_max_abs": max(abs(x) for x in res["isolation_dD"]),
}
print(json.dumps(summ, indent=2))
for k in ("d_clean", "d_corr", "d_outpatch", "d_statepatch"):
    v = [r[k] for r in res["output_patch_vs_state_patch"]]
    print(f"  mean {k:14s} = {sum(v)/len(v):+.3f}")
json.dump({"summary": summ, "raw": res}, open(a.out, "w"), indent=2)
