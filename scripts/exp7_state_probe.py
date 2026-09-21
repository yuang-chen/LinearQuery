"""Step 7: is the key->value binding present in the GDN recurrent state at all?

Route isolation (step 6) shows the state is not *used*.  That leaves two possibilities:
the binding is absent from the state, or it is present but nothing downstream reads it.
A decoding probe settles it without touching the model.

Task for the probe: from a frozen GDN recurrent state, predict **which value word the
queried key is bound to** (20-way).  Clean and corrupted prompts of the same example are
separate samples with different labels, so the probe cannot win by memorising the example.

Three floors to beat:
  0.05  uninformed chance (20 value words)
  0.125 "presence" floor -- a probe that knows only which 8 values appear in the prompt but
        not the bindings can do no better than 1/8
  shuffled-label control, refit, as an empirical null

Readouts:
  flat      : whole state, random-projected to 2048 dims then PCA (JL projection keeps
              linear separability; the raw state is 16x128x128 = 262144 dims)
  query-key : S contracted with the query key's own k-vector at the query position -- the
              delta-rule's own read operation, i.e. what the layer would itself retrieve
  ceiling   : final-token residual stream -- a pipeline check; the model answers from this,
              so it must decode near 1.0
"""
import sys, json, argparse, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, "/user/yac/LinearAblation")
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.task import make_examples, VALUE_WORDS
from src.runner import Runner, Capture, hook_ctx
from src.positions import token_positions

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--n_pairs", type=int, default=8)
ap.add_argument("--seed", type=int, default=77)
ap.add_argument("--cut", default="final", choices=["target", "dict_end", "final"])
ap.add_argument("--pca", type=int, default=96)
ap.add_argument("--proj", type=int, default=2048)
ap.add_argument("--out", default="results/exp7_state_probe.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
exs = make_examples(a.n, a.n_pairs, seed=a.seed, target_pos=2, distract_pos=5)
VIDX = {v: i for i, v in enumerate(VALUE_WORDS)}
gen = torch.Generator(device="cpu").manual_seed(0)
PROJ = {}


def gdn_qk(module, hidden):
    """Recompute the layer's l2-normalised query/key, mirroring Qwen3_5GatedDeltaNet.forward."""
    b, t, _ = hidden.shape
    mixed = module.in_proj_qkv(hidden).transpose(1, 2)
    mixed = F.silu(F.conv1d(mixed, module.conv1d.weight, module.conv1d.bias,
                            groups=module.conv_dim, padding=module.conv_kernel_size - 1)[..., :t])
    mixed = mixed.transpose(1, 2)
    q, k, _ = torch.split(mixed, [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    q = q.reshape(b, t, -1, module.head_k_dim)
    k = k.reshape(b, t, -1, module.head_k_dim)
    return F.normalize(q, dim=-1), F.normalize(k, dim=-1)


def project(name, X):
    """Fixed random Gaussian projection to keep the flat state tractable."""
    if X.shape[1] <= a.proj:
        return X
    if name not in PROJ:
        PROJ[name] = torch.randn(X.shape[1], a.proj, generator=gen) / a.proj ** .5
    return X @ PROJ[name].to(X.device, X.dtype)


feats, labels = {}, []
BS = 25
for i0 in range(0, len(exs), BS):
    chunk = exs[i0:i0 + BS]
    pos = token_positions(R.tok, chunk[0])
    cut = {"target": pos["target_value"],
           "dict_end": max(pos["value_pos"].values()) + 1,
           "final": pos["final"]}[a.cut]
    qk_pos = pos["query_key_question"]
    for prompt_fn, ans_fn in ((lambda e: e.clean_prompt(), lambda e: e.answer),
                              (lambda e: e.corrupt_prompt(), lambda e: e.answer_corr)):
        ids = R.tok([prompt_fn(e) for e in chunk], return_tensors="pt").input_ids.to(R.device)
        states = R.recurrent_states(ids, cut)
        caps = {L: Capture(R.mixer(L), mode="in") for L in R.gdn_layers}
        cap_last = Capture(R.layers[R.n_layers - 1])
        with hook_ctx(list(caps.values()) + [cap_last]):
            R.forward(ids)
        for L in R.gdn_layers:
            S = states[L].float()                                   # [B, H, dk, dv]
            feats.setdefault((L, "flat"), []).append(
                project(f"flat{L}", S.flatten(1)).cpu().numpy())
            _, k = gdn_qk(R.mixer(L), caps[L].value.float())
            read = torch.einsum("bhd,bhdv->bhv", k[:, qk_pos].float(), S)
            feats.setdefault((L, "query-key"), []).append(read.flatten(1).cpu().numpy())
        feats.setdefault((-1, "ceiling"), []).append(cap_last.value[:, -1].float().cpu().numpy())
        labels += [VIDX[ans_fn(e)] for e in chunk]
    print(f"{i0 + len(chunk)}/{len(exs)}", flush=True)

y = np.array(labels)
rng = np.random.default_rng(0)
y_shuf = rng.permutation(y)


def score(X, yy):
    clf = make_pipeline(StandardScaler(),
                        PCA(n_components=min(a.pca, X.shape[0] - 1, X.shape[1])),
                        LogisticRegression(max_iter=5000, C=1.0))
    return cross_val_score(clf, X, yy, cv=5, scoring="accuracy")


rows = []
print(f"\nfloors: uninformed {1/len(VALUE_WORDS):.3f}   presence-only {1/a.n_pairs:.3f}")
print("layer  readout     accuracy        shuffled-label null")
for key in sorted(feats, key=lambda k: (k[1], k[0])):
    X = np.concatenate(feats[key], 0)
    s = score(X, y)
    s0 = score(X, y_shuf)
    rows.append(dict(layer=key[0], family=key[1], acc=float(s.mean()),
                     sem=float(s.std() / len(s) ** .5), null=float(s0.mean())))
    print(f"{key[0]:5d}  {key[1]:10s}  {s.mean():.3f} ± {s.std()/len(s)**.5:.3f}   {s0.mean():.3f}")

json.dump(dict(meta=vars(a), floors={"uninformed": 1 / len(VALUE_WORDS),
                                     "presence_only": 1 / a.n_pairs}, rows=rows),
          open(a.out, "w"), indent=2)
