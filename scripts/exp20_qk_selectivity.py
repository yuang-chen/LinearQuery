"""Step 20: q.k selectivity of the reader L15H5 with and without G2's contribution to the keys.

Part XII left one claim inferred: without G2 the key identity is still linearly decodable at every
value token (probe 0.99), yet the reader confuses values -- so G2 must make the keys selective
*along the direction the query reads*. Here the reader's actual pre-softmax scores are measured.

Scores: L15H5's attention log-probabilities from the final token (eager attention). Differences
of log-probs are exact differences of pre-softmax scores (after q/k-norm, RoPE, scaling), so
s_j - s_ref needs no re-implementation of the attention.

Q x K design (Q at the final token, K at all positions, from the same prompt's donor run):
  intact Q + intact K | intact Q + ablated K | ablated Q + intact K | ablated Q + ablated K
for ablations G2, L10, and G3 (contrast: G3 is query-side).

Per prompt, over the 8 value tokens:
  margin     s_target - max_other_value           (> 0 : target wins among values)
  top1       target has the highest score among values
  tgt-sink   s_target - s_<|im_start|>             (absolute match strength vs the sink)
  oth-sink   mean_other_value - s_<|im_start|>
  spread     std of the 7 non-target value scores
Key geometry (k_proj output of H5's KV head at the 8 value tokens, before k-norm and RoPE):
  mean pairwise cosine between entries' keys, and the entry-specific share of the key energy
  (mean |k_i - k_mean|^2 / mean |k_i|^2).
"""
import sys, json, argparse, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from src.runner import Runner, hook_ctx
from src.qkv import AttnWeights, ProjPatch, ProjCapture
from src.task import make_examples
from src.harness import group_examples

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/exp20_qk_selectivity.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
R.model.config.get_text_config()._attn_implementation = "eager"
HD = R.cfg.head_dim
L, H = 15, 5
KV = H // (R.cfg.num_attention_heads // R.cfg.num_key_value_heads)
ATTN = R.mixer(L)

exs = []
for i, (ti, di) in enumerate([(2, 5), (5, 2), (1, 6), (6, 1)]):
    exs += make_examples(25, 8, seed=11 + i, target_pos=ti, distract_pos=di)
groups = group_examples(R, exs)
ABL = {"G2": [8, 9, 10], "L10": [10], "G3": [12, 13, 14]}


class Zero:
    def __init__(self, L_): self.L = L_
    def register(self):
        return R.mixer(self.L).register_forward_hook(lambda m, a_, o: torch.zeros_like(o))


@torch.no_grad()
def capture(ids, zl):
    cq, ck = ProjCapture(ATTN, "q"), ProjCapture(ATTN, "k")
    with hook_ctx([cq, ck] + [Zero(x) for x in zl]):
        R.model(ids, use_cache=False)
    return cq.value, ck.value


@torch.no_grad()
def scores(g, ids, q_src, k_src):
    """Run the INTACT model, overwrite L15 Q (final) and K (all) with the given donors."""
    aw = AttnWeights(ATTN)
    n = ids.shape[1]
    hooks = [aw, ProjPatch(ATTN, "q", [n - 1], q_src, HD),
             ProjPatch(ATTN, "k", list(range(n)), k_src, HD)]
    with hook_ctx(hooks):
        R.model(ids, use_cache=False)
    return aw.value[:, H, g.pos["final"]].float().clamp_min(1e-30).log()     # [B, T]


def metrics(g, logp):
    vp = [g.pos["value_pos"][i] for i in range(8)]
    ti = g.exs[0].target_idx
    s = logp[:, vp]                                   # [B, 8]
    tgt = s[:, ti]
    oth = torch.cat([s[:, :ti], s[:, ti + 1:]], 1)
    sink = logp[:, 0]
    return dict(margin=(tgt - oth.max(1).values), top1=(tgt > oth.max(1).values).float(),
                tgt_sink=tgt - sink, oth_sink=oth.mean(1) - sink, spread=oth.std(1))


def key_geometry(g, kvec):
    vp = [g.pos["value_pos"][i] for i in range(8)]
    k = kvec[:, vp, KV * HD:(KV + 1) * HD].float()           # [B, 8, HD]
    iu = torch.triu_indices(8, 8, 1)

    def mean_cos(x):
        x = torch.nn.functional.normalize(x, dim=-1)
        c = x @ x.transpose(1, 2)
        return c[:, iu[0], iu[1]].mean(1)
    dev = (k - k.mean(1, keepdim=True)).pow(2).sum(-1).mean(1)
    return dict(cos_raw=mean_cos(k), entry_share=dev / k.pow(2).sum(-1).mean(1))


rows, out = [], {}
DON = {}
for g in groups:
    DON[(id(g), "intact")] = capture(g.clean_ids, [])
    for an, zl in ABL.items():
        DON[(id(g), an)] = capture(g.clean_ids, zl)

print("L15H5 scores at the 8 value tokens (clean prompts, 100 examples), s = pre-softmax score\n",
      flush=True)
hdr = f"  {'Q from':8s} {'K from':8s} {'margin':>7s} {'top1':>6s} {'tgt-sink':>9s} {'oth-sink':>9s} {'spread':>7s}"
for an in ABL:
    print(f"== ablation {an} {ABL[an]} ==\n{hdr}", flush=True)
    for qs in ("intact", an):
        for ks in ("intact", an):
            acc = {}
            for g in groups:
                lp = scores(g, g.clean_ids, DON[(id(g), qs)][0], DON[(id(g), ks)][1])
                for k_, v in metrics(g, lp).items():
                    acc.setdefault(k_, []).append(v)
            m = {k_: float(torch.cat(v).mean()) for k_, v in acc.items()}
            m_sem = {k_ + "_sem": float(torch.cat(v).std() / len(torch.cat(v)) ** 0.5)
                     for k_, v in acc.items()}
            rows.append(dict(section="scores", abl=an, q=qs, k=ks, **m, **m_sem))
            print(f"  {qs:8s} {ks:8s} {m['margin']:+7.2f} {m['top1']:6.3f} {m['tgt_sink']:+9.2f} "
                  f"{m['oth_sink']:+9.2f} {m['spread']:7.2f}", flush=True)
    print(flush=True)

print("== key geometry: H5's KV head, k_proj output at the 8 value tokens ==", flush=True)
for src in ["intact"] + list(ABL):
    acc = {}
    for g in groups:
        for k_, v in key_geometry(g, DON[(id(g), src)][1]).items():
            acc.setdefault(k_, []).append(v)
    m = {k_: float(torch.cat(v).mean()) for k_, v in acc.items()}
    rows.append(dict(section="geometry", src=src, **m))
    print(f"  keys from {src:7s} mean pairwise cos {m['cos_raw']:.3f}  "
          f"entry-specific share {m['entry_share']:.3f}", flush=True)

json.dump(dict(meta=vars(a), rows=rows), open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
