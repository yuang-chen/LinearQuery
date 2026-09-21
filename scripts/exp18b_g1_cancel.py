"""Step 18b: do downstream MLPs cancel G1's write?

exp18 C: restoring MLPs 4-14 to their intact outputs in a G1-less run is catastrophic (acc 0),
worse than removing G1. Hypothesis: in the intact model, part of the MLP output is a response to
G1's write that points *against* it. Measured in the intact run (WikiText, all positions):
  w     = sum of G1 mixer outputs (layers 4-6)
  dM    = MLP output (intact) - MLP output (G1 removed), summed over MLPs 4-14: the part of the
          MLP output that exists because of G1
  cos(dM, w), ||dM|| / ||w||, and ||w + dM|| / ||w|| (how much of G1's write survives).
"""
import sys, json, torch
sys.path.insert(0, "/user/yac/LinearAblation")
from datasets import load_dataset
from src.runner import Runner, hook_ctx, Capture

R = Runner(dtype=torch.float32)
G1 = [4, 5, 6]
d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
ids = R.tok("\n\n".join(t for t in d["text"] if t.strip()), return_tensors="pt").input_ids[0]
seqs = [ids[i * 512:(i + 1) * 512][None].to(R.device) for i in range(12)]


class Zero:
    def __init__(self, L): self.L = L
    def register(self):
        return R.mixer(self.L).register_forward_hook(lambda m, a_, o: torch.zeros_like(o))


def run(s, zl):
    caps = {("mix", L): Capture(R.mixer(L)) for L in G1}
    caps.update({("mlp", L): Capture(R.mlp(L)) for L in range(4, 15)})
    with torch.no_grad(), hook_ctx(list(caps.values()) + [Zero(L) for L in zl]):
        R.model(s, use_cache=False)
    return {k: c.value[0].float() for k, c in caps.items()}


acc = {}
for s in seqs:
    I, N = run(s, []), run(s, G1)
    w = sum(I[("mix", L)] for L in G1)
    for name, Ls in (("MLPs 4-6", range(4, 7)), ("MLPs 7-14", range(7, 15)), ("MLPs 4-14", range(4, 15))):
        dM = sum(I[("mlp", L)] - N[("mlp", L)] for L in Ls)
        r = acc.setdefault(name, {"cos": [], "ratio": [], "survive": []})
        r["cos"] += torch.nn.functional.cosine_similarity(dM, w, dim=-1).tolist()
        r["ratio"] += (dM.norm(dim=-1) / w.norm(dim=-1)).tolist()
        r["survive"] += ((w + dM).norm(dim=-1) / w.norm(dim=-1)).tolist()
out = {}
print(f"{'':12s} {'cos(dMLP, G1 write)':>20s} {'|dMLP|/|w|':>11s} {'|w+dMLP|/|w|':>13s}")
for n, r in acc.items():
    out[n] = {k: float(torch.tensor(v).median()) for k, v in r.items()}
    print(f"{n:12s} {out[n]['cos']:20.3f} {out[n]['ratio']:11.2f} {out[n]['survive']:13.2f}   (medians)")
json.dump(out, open("results/exp18b_g1_cancel.json", "w"), indent=2)
