"""Step 12b: separate layer 0's write *magnitude* from its write *content*.

Step 12 perturbed each mixer output by eps*||o_L||, i.e. matched in units of that layer's own
write.  But layer 0's write is 1.95x the residual stream it reads while layer 1's is 0.67x, so
the same eps injects a much larger absolute perturbation at layer 0 -- the comparison is
confounded by exactly the geometry it is meant to control for.

Here the perturbation is matched in absolute terms (o_L + delta*u, same delta everywhere) and,
separately, relative to the residual stream the layer writes into (o_L + eps*||h_in||*u).
Both are the fair versions.  Per-sequence standard errors included, because layer 1's effects
are small enough that ordering claims need them.
"""
import sys, json, argparse, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from datasets import load_dataset
from src.runner import Runner, hook_ctx

ap = argparse.ArgumentParser()
ap.add_argument("--n_eval", type=int, default=16)
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--layers", default="0,1,2,4")
ap.add_argument("--abs", default="0.05,0.1,0.2,0.4,0.8,1.6")
ap.add_argument("--rel", default="0.05,0.1,0.2,0.4,0.8,1.6")
ap.add_argument("--out", default="results/exp12b_matched_noise.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
LAYERS = [int(x) for x in a.layers.split(",")]
d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
ids = R.tok("\n\n".join(t for t in d["text"] if t.strip()), return_tensors="pt").input_ids[0]
seqs = [ids[i * a.seq_len:(i + 1) * a.seq_len].unsqueeze(0).to(R.device) for i in range(a.n_eval)]

HIN = {}


class CapIn:
    def __init__(self, L): self.L = L
    def register(self):
        def fn(mod, args, kwargs):
            h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            HIN[self.L] = h.detach().norm(dim=-1, keepdim=True)
        return R.layers[self.L].register_forward_pre_hook(fn, with_kwargs=True)


class Noise:
    """mode 'abs': o + delta*u ; mode 'rel_h': o + eps*||h_in||*u ; 'rel_o': o + eps*||o||*u"""

    def __init__(self, L, mode, amt, seed=0):
        self.L, self.mode, self.amt, self.seed = L, mode, amt, seed

    def register(self):
        def fn(mod, args, output):
            g = torch.Generator(device=output.device).manual_seed(self.seed)
            u = torch.randn(output.shape, generator=g, device=output.device, dtype=output.dtype)
            u = u / u.norm(dim=-1, keepdim=True)
            if self.mode == "abs":
                scale = self.amt
            elif self.mode == "rel_h":
                scale = self.amt * HIN[self.L].to(output.dtype)
            else:
                scale = self.amt * output.norm(dim=-1, keepdim=True)
            return output + scale * u
        return R.mixer(self.L).register_forward_hook(fn)


@torch.no_grad()
def nll_per_seq(hooks=()):
    out = []
    for s in seqs:
        with hook_ctx([CapIn(L) for L in LAYERS] + list(hooks)):
            lg = R.model(s, use_cache=False).logits
        out.append(torch.nn.functional.cross_entropy(
            lg[0, :-1].float(), s[0, 1:]).item())
    return np.array(out)


base = nll_per_seq()
print(f"intact NLL {base.mean():.4f} +/- {base.std()/len(base)**.5:.4f}  "
      f"(PPL {np.exp(base.mean()):.2f})\n", flush=True)
rows = []


def run(mode, L, amt):
    v = nll_per_seq([Noise(L, mode, amt)])
    dn = v - base
    r = dict(mode=mode, layer=L, amt=amt, dnll=float(dn.mean()),
             sem=float(dn.std(ddof=1) / len(dn) ** .5), ppl=float(np.exp(v.mean())))
    rows.append(r)
    return r


for mode, amts, lab in (("abs", a.abs, "absolute:  o + delta*u"),
                        ("rel_h", a.rel, "relative to residual stream:  o + eps*||h_in||*u")):
    print(f"== {lab} ==")
    print(f"{'amount':>8s}" + "".join(f"{'L'+str(L):>16s}" for L in LAYERS))
    for amt in [float(x) for x in amts.split(",")]:
        line = f"{amt:8.2f}"
        for L in LAYERS:
            r = run(mode, L, amt)
            line += f"  {r['dnll']:+8.4f}±{r['sem']:.3f}"
        print(line, flush=True)
    print()

json.dump(dict(meta=vars(a), base_nll=float(base.mean()), rows=rows), open(a.out, "w"), indent=2)
print("wrote", a.out)
