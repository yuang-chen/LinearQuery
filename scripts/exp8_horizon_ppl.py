"""Step 8: how far back does the GDN recurrent state usefully reach, for *general* language
modelling rather than for the retrieval task?

Motivated by Borobia et al. (arXiv:2603.22473), who ablate whole GDN layers in this same
model and find WikiText-2 perplexity 35,200x worse without the linear pathway (vs 82x
without attention).  That intervention removes the layer's token mixer *and* its local
feed-forward, so it cannot say whether the value lies in the recurrence or in the layer's
local (conv + pointwise) computation.  Here the recurrence alone is dialled down.

Main sweep -- HORIZON.  Reset every GDN recurrent state to zero every `w` tokens, leaving
the convolution cache and the full softmax KV cache untouched.  w = 1 keeps only the
4-token conv window; w = inf is the intact model.  The perplexity-vs-w curve is the
functional memory horizon of the state.

Calibration conditions (their protocol, reimplemented here so the numbers are comparable):
  layer skip: replace a decoder layer's output with its input (mixer + local MLP removed)
  mixer only: zero the token mixer's output, keeping the layer-local MLP
  matched random controls for both.
"""
import sys, json, argparse, random, statistics as st, torch
sys.path.insert(0, "/user/yac/LinearAblation")
from datasets import load_dataset
from src.runner import Runner, hook_ctx

ap = argparse.ArgumentParser()
ap.add_argument("--n_seq", type=int, default=32)
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--windows", default="1,2,4,8,16,32,64,128,256,inf")
ap.add_argument("--n_random", type=int, default=3)
ap.add_argument("--out", default="results/exp8_horizon_ppl.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
tok = R.tok

ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
text = "\n\n".join(t for t in ds["text"] if t.strip())
ids_all = tok(text, return_tensors="pt").input_ids[0]
seqs = [ids_all[i * a.seq_len:(i + 1) * a.seq_len].unsqueeze(0).to(R.device)
        for i in range(a.n_seq)]
print(f"{len(seqs)} sequences x {a.seq_len} tokens", flush=True)


def nll_windowed(ids, w, hooks=()):
    """Teacher-forced NLL, resetting GDN recurrent states every `w` tokens (w=None: never)."""
    from transformers import DynamicCache
    cache = DynamicCache(config=R.model.config.get_text_config())
    logits = []
    step = ids.shape[1] if w is None else w
    with hook_ctx(hooks), torch.no_grad():
        for i in range(0, ids.shape[1], step):
            out = R.model(ids[:, i:i + step], past_key_values=cache, use_cache=True)
            logits.append(out.logits)
            if w is not None:
                for L in R.gdn_layers:
                    rs = cache.layers[L].recurrent_states[0]
                    if rs is not None:
                        rs.zero_()
    lg = torch.cat(logits, 1)[:, :-1].float()
    tgt = ids[:, 1:]
    return torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1)).item()


class LayerSkip:
    """Replace a decoder layer's output with its input (the paper's sequential-skip ablation)."""
    def __init__(self, layer):
        self.layer = layer

    def register(self):
        def fn(mod, args, kwargs, output):
            h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            return h
        return self.layer.register_forward_hook(fn, with_kwargs=True)


class MixerOff:
    """Zero a token mixer's output, keeping the layer-local MLP intact."""
    def __init__(self, module, is_attn):
        self.module, self.is_attn = module, is_attn

    def register(self):
        def fn(mod, args, output):
            if self.is_attn:
                return (torch.zeros_like(output[0]),) + tuple(output[1:])
            return torch.zeros_like(output)
        return self.module.register_forward_hook(fn)


rows = []


def run(name, w, hooks=(), **kw):
    v = [nll_windowed(s, w, hooks) for s in seqs]
    m = sum(v) / len(v)
    row = dict(name=name, window=w, nll=m, ppl=float(torch.tensor(m).exp()),
               sem=st.pstdev(v) / len(v) ** .5, **kw)
    rows.append(row)
    print(f"{name:34s} NLL {m:7.4f}   PPL {row['ppl']:14.2f}", flush=True)
    return row


base = run("baseline", None)
print("\n-- horizon sweep: GDN recurrent state reset every w tokens --", flush=True)
for wtxt in a.windows.split(","):
    if wtxt == "inf":
        continue
    run(f"state reset every {wtxt} tok", int(wtxt))

print("\n-- calibration: layer / mixer removal (paper's protocol) --", flush=True)
run("all GDN layers skipped", None, [LayerSkip(R.layers[L]) for L in R.gdn_layers])
run("all softmax layers skipped", None, [LayerSkip(R.layers[L]) for L in R.attn_layers])
run("all GDN mixers zeroed (MLP kept)", None,
    [MixerOff(R.mixer(L), False) for L in R.gdn_layers])
run("all softmax mixers zeroed (MLP kept)", None,
    [MixerOff(R.mixer(L), True) for L in R.attn_layers])
run("GDN layer 0 skipped", None, [LayerSkip(R.layers[0])])
run("GDN layer 0 mixer zeroed", None, [MixerOff(R.mixer(0), False)])

print("\n-- matched random controls --", flush=True)
rng = random.Random(0)
for n_rm, label in ((18, "18 random layers skipped"), (6, "6 random layers skipped")):
    for t in range(a.n_random):
        pick = rng.sample(range(R.n_layers), n_rm)
        run(f"{label} (trial {t})", None, [LayerSkip(R.layers[L]) for L in pick],
            layers=sorted(pick))

json.dump(dict(meta=vars(a), rows=rows), open(a.out, "w"), indent=2)
print(f"\nbaseline PPL {base['ppl']:.2f}")
