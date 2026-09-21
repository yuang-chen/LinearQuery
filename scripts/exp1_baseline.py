"""Step 1: task validation + baseline accuracy on clean and corrupted prompts."""
import sys, json, argparse, torch
sys.path.insert(0, "/user/yac/LinearAblation")
from src.task import make_examples, verify_single_token
from src.runner import Runner
from src.positions import token_positions
from src.metrics import logit_diff

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16])
ap.add_argument("--template", default="chat")
ap.add_argument("--out", default="results/exp1_baseline.json")
a = ap.parse_args()

R = Runner()
tok = R.tok
rows = []
for np_ in a.sizes:
    exs = make_examples(a.n, np_, seed=a.seed, template=a.template)
    nver = sum(verify_single_token(tok, e)[0] for e in exs)
    ok_clean = ok_corr = ok_both = 0
    dcs, dks = [], []
    for e in exs:
        sa, sc = e.answer_strs()
        aid, cid = tok(sa).input_ids[0], tok(sc).input_ids[0]
        ci = R.encode(e.clean_prompt()); xi = R.encode(e.corrupt_prompt())
        lc = R.forward(ci).logits[0, -1]
        lx = R.forward(xi).logits[0, -1]
        c_ok = lc.argmax().item() == aid          # clean prompt -> clean value
        x_ok = lx.argmax().item() == cid          # corrupted prompt -> swapped-in value
        ok_clean += c_ok; ok_corr += x_ok; ok_both += (c_ok and x_ok)
        dcs.append(logit_diff(lc, aid, cid)); dks.append(logit_diff(lx, aid, cid))
        token_positions(tok, e)  # asserts position bookkeeping
    row = dict(n_pairs=np_, n=a.n, template=a.template, verified=nver,
               acc_clean=ok_clean / a.n, acc_corrupt=ok_corr / a.n, acc_both=ok_both / a.n,
               D_clean=sum(dcs) / len(dcs), D_corrupt=sum(dks) / len(dks),
               n_tokens=len(tok(exs[0].clean_prompt()).input_ids))
    rows.append(row); print(json.dumps(row))
json.dump(rows, open(a.out, "w"), indent=2)
