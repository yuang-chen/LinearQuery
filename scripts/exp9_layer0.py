"""Step 9: why is GDN layer 0 so important, and what is it computing?

Layer 0 is the single most important linear layer in this model (Borobia et al. find the
largest hidden-state norm change, 1.98, far above every other layer; Part I finds its output
at the source value token is sufficient AND necessary for retrieval; removing it alone costs
~1600x perplexity).  Three hypotheses make different predictions:

  H1 "second embedding"   : a context-free token-identity lookup, just rescaled/relocated
  H2 "local mixer"        : detokenisation over the 4-tap conv window / short recurrence
  H3 "scale-setter"       : a large near-constant write that sets the residual-stream scale
                            downstream RMSNorms expect (a bias, carrying no information)

INFORMATION LADDER.  Replace the layer's mixer output o_L(t) with surrogates carrying
progressively more information, and measure WikiText-2 perplexity:

  rung 0  zeros                      no information, no scale
  rung 1  global mean vector mu      scale + direction, ZERO information  -> tests H3
  rung 2  norm-matched random dir    scale only, wrong direction          -> control for rung 1
  rung 3  token-type mean mu[x_t]    current token identity ONLY, no ctx  -> tests H1
  rung 4  intact                     full

  rung 3 ~ rung 4  => context-free lookup table (H1)
  rung 1 ~ rung 4  => bias/scale offset (H3)
  only rung 4      => genuine contextual computation (H2 or other)

MECHANISM SPLIT.  Within layer 0's mixer, separate the 4-tap depthwise convolution from the
delta-rule recurrence (conv window forced to 1 tap; recurrent state reset every token).

CHANNEL CONCENTRATION.  Is the effect carried by a handful of outlier "massive activation"
channels?  Ablate only the top-k channels by mean |o| and sweep k.

MATCHED CONTROLS.  The whole ladder is repeated on GDN layers 1, 2 and 4, so "layer 0 is
special" is a contrast and not an isolated number.
"""
import sys, json, argparse, copy, statistics as st, torch
sys.path.insert(0, "/user/yac/LinearAblation")
from datasets import load_dataset
from transformers import DynamicCache
from src.runner import Runner, hook_ctx
from src.task import make_examples
from src.harness import group_examples

ap = argparse.ArgumentParser()
ap.add_argument("--n_stat", type=int, default=200)      # sequences for the token-type means
ap.add_argument("--n_eval", type=int, default=16)       # held-out sequences for perplexity
ap.add_argument("--seq_len", type=int, default=512)
ap.add_argument("--layers", default="0,1,2,4")
ap.add_argument("--topk", default="1,2,4,8,16,32,64,128")
ap.add_argument("--out", default="results/exp9_layer0.json")
a = ap.parse_args()

R = Runner(dtype=torch.float32)
tok = R.tok
LAYERS = [int(x) for x in a.layers.split(",")]
D = R.cfg.hidden_size

ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
text = "\n\n".join(t for t in ds["text"] if t.strip())
ids_all = tok(text, return_tensors="pt").input_ids[0]
need = (a.n_stat + a.n_eval) * a.seq_len
assert ids_all.numel() >= need, (ids_all.numel(), need)
stat_seqs = [ids_all[i * a.seq_len:(i + 1) * a.seq_len].unsqueeze(0).to(R.device)
             for i in range(a.n_stat)]
eval_seqs = [ids_all[(a.n_stat + i) * a.seq_len:(a.n_stat + i + 1) * a.seq_len].unsqueeze(0).to(R.device)
             for i in range(a.n_eval)]
print(f"stats on {a.n_stat} seqs, eval on {a.n_eval} held-out seqs x {a.seq_len} tokens", flush=True)

# --------------------------------------------------------------------------- statistics
uniq = torch.unique(torch.cat([s[0] for s in stat_seqs]))
remap = torch.full((R.cfg.vocab_size,), -1, dtype=torch.long, device=R.device)
remap[uniq] = torch.arange(uniq.numel(), device=R.device)
sums = {L: torch.zeros(uniq.numel(), D, device=R.device) for L in LAYERS}
samp = {L: torch.zeros(uniq.numel(), D, device=R.device) for L in LAYERS}
cnts = torch.zeros(uniq.numel(), device=R.device)
absmean = {L: torch.zeros(D, device=R.device) for L in LAYERS}
nchunk = 0


class Grab:
    def __init__(self, L):
        self.L, self.value = L, None

    def register(self):
        def fn(mod, args, output):
            self.value = output.detach()
        return R.mixer(self.L).register_forward_hook(fn)


with torch.no_grad():
    for si, s in enumerate(stat_seqs):
        gs = {L: Grab(L) for L in LAYERS}
        with hook_ctx(list(gs.values())):
            R.model(s, use_cache=False)
        idx = remap[s[0]]
        cnts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
        for L in LAYERS:
            o = gs[L].value[0].float()
            sums[L].index_add_(0, idx, o)
            samp[L][idx] = o          # last occurrence wins -> one real vector per type
            absmean[L] += o.abs().mean(0)
        nchunk += 1
        if (si + 1) % 50 == 0:
            print(f"  stats {si+1}/{a.n_stat}", flush=True)

tok_mean = {L: sums[L] / cnts[:, None].clamp(min=1) for L in LAYERS}
glob_mean = {L: (sums[L].sum(0) / cnts.sum()) for L in LAYERS}
TAB = {"remap": remap, "mean": tok_mean, "samp": samp, "glob": glob_mean}
absmean = {L: absmean[L] / nchunk for L in LAYERS}
cover = [(remap[s[0]] >= 0).float().mean().item() for s in eval_seqs]
print(f"token-type table covers {sum(cover)/len(cover):.3f} of eval tokens "
      f"({uniq.numel()} types)", flush=True)

# --------------------------------------------------------------------------- surrogates
CUR = {}          # current input ids, set before each forward


class Surrogate:
    """Replace mixer L's output with a surrogate; `channels` restricts it to a channel subset."""

    def __init__(self, L, mode, channels=None, seed=0):
        self.L, self.mode, self.channels, self.seed = L, mode, channels, seed

    def register(self):
        L, mode = self.L, self.mode

        def fn(mod, args, output):
            o = output
            ids = CUR["ids"]
            if mode == "zero":
                sub = torch.zeros_like(o)
            elif mode == "global_mean":
                sub = TAB["glob"][L].expand_as(o).clone()
            elif mode == "rand_dir":
                gcpu = torch.Generator(device=o.device).manual_seed(self.seed)
                r = torch.randn(o.shape, generator=gcpu, device=o.device, dtype=o.dtype)
                r = r / r.norm(dim=-1, keepdim=True)
                sub = r * o.norm(dim=-1, keepdim=True)
            elif mode == "token_mean":
                idx = TAB["remap"][ids]
                sub = torch.where((idx >= 0)[..., None],
                                  TAB["mean"][L][idx.clamp(min=0)], TAB["glob"][L])
            elif mode == "token_sample":
                idx = TAB["remap"][ids]
                sub = torch.where((idx >= 0)[..., None],
                                  TAB["samp"][L][idx.clamp(min=0)], TAB["glob"][L])
            else:
                raise ValueError(mode)
            if self.channels is None:
                return sub
            out = o.clone()
            out[..., self.channels] = sub[..., self.channels]
            return out
        return R.mixer(L).register_forward_hook(fn)


@torch.no_grad()
def ppl(hooks=(), state_reset=None, seqs=None):
    tot, n = 0.0, 0
    for s in (seqs or eval_seqs):
        CUR["ids"] = s
        if state_reset is None:
            with hook_ctx(hooks):
                lg = R.model(s, use_cache=False).logits
        else:
            L0, w = state_reset
            cache = DynamicCache(config=R.model.config.get_text_config())
            outs = []
            with hook_ctx(hooks):
                for i in range(0, s.shape[1], w):
                    CUR["ids"] = s[:, i:i + w]
                    outs.append(R.model(s[:, i:i + w], past_key_values=cache, use_cache=True).logits)
                    rs = cache.layers[L0].recurrent_states[0]
                    if rs is not None:
                        rs.zero_()
            lg = torch.cat(outs, 1)
        loss = torch.nn.functional.cross_entropy(
            lg[:, :-1].float().reshape(-1, lg.shape[-1]), s[:, 1:].reshape(-1))
        tot += loss.item(); n += 1
    return tot / n


rows = []


def rec(name, nll, **kw):
    r = dict(name=name, nll=nll, ppl=float(torch.tensor(nll).exp()), **kw)
    rows.append(r)
    print(f"{name:44s} NLL {nll:7.4f}   PPL {r['ppl']:13.2f}", flush=True)
    return r


base = rec("intact (rung 4)", ppl())

print("\n== information ladder ==", flush=True)
for L in LAYERS:
    for mode, lab in (("zero", "rung 0 zeros"), ("global_mean", "rung 1 global mean"),
                      ("rand_dir", "rung 2 norm-matched random"),
                      ("token_mean", "rung 3 token-type mean"),
                      ("token_sample", "rung 3b same token, other context")):
        rec(f"L{L} mixer <- {lab}", ppl([Surrogate(L, mode)]), layer=L, rung=mode)

print("\n== mechanism split (layer 0) ==", flush=True)
w0 = R.mixer(0).conv1d.weight.data.clone()
R.mixer(0).conv1d.weight.data[..., :-1] = 0
rec("L0 conv window forced to 1 tap", ppl(), layer=0, rung="conv1tap")
R.mixer(0).conv1d.weight.data.copy_(w0)
rec("L0 recurrent state reset every token", ppl(state_reset=(0, 1)), layer=0, rung="norecur")
rec("L0 recurrence reset every 8 tokens", ppl(state_reset=(0, 8)), layer=0, rung="recur8")

print("\n== channel concentration (layer 0) ==", flush=True)
order = absmean[0].argsort(descending=True)
for k in [int(x) for x in a.topk.split(",")]:
    ch = order[:k]
    rec(f"L0 top-{k} channels <- token-type mean", ppl([Surrogate(0, "token_mean", channels=ch)]),
        layer=0, rung=f"top{k}_tokmean", k=k)
    rec(f"L0 top-{k} channels <- global mean", ppl([Surrogate(0, "global_mean", channels=ch)]),
        layer=0, rung=f"top{k}_globmean", k=k)
mag = absmean[0][order]
rows.append(dict(name="channel_profile", absmean_sorted=mag[:64].tolist(),
                 total=float(absmean[0].sum()), top8_share=float(mag[:8].sum() / absmean[0].sum()),
                 top32_share=float(mag[:32].sum() / absmean[0].sum())))
print(f"top-8 channels carry {rows[-1]['top8_share']:.3f} of mean |o_0|; "
      f"top-32 carry {rows[-1]['top32_share']:.3f}", flush=True)

# --------------------------------------------------------------------------- retrieval task
# The WikiText table does not cover the chat-template tokens, so the retrieval ladder gets its
# own token-type table built from *held-out task prompts* (different examples, same template).
print("\n== retrieval task under the ladder ==", flush=True)
donor_exs = []
for i, (ti, di) in enumerate([(3, 6), (6, 3), (0, 4), (4, 0)]):
    donor_exs += make_examples(25, 8, seed=901 + i, target_pos=ti, distract_pos=di)
donor_ids = [R.tok([e.clean_prompt() for e in donor_exs[i:i + 25]], return_tensors="pt"
                   ).input_ids.to(R.device) for i in range(0, len(donor_exs), 25)]
t_uniq = torch.unique(torch.cat([b.reshape(-1) for b in donor_ids]))
t_remap = torch.full((R.cfg.vocab_size,), -1, dtype=torch.long, device=R.device)
t_remap[t_uniq] = torch.arange(t_uniq.numel(), device=R.device)
t_sums = {L: torch.zeros(t_uniq.numel(), D, device=R.device) for L in LAYERS}
t_samp = {L: torch.zeros(t_uniq.numel(), D, device=R.device) for L in LAYERS}
t_cnts = torch.zeros(t_uniq.numel(), device=R.device)
with torch.no_grad():
    for b in donor_ids:
        gs = {L: Grab(L) for L in LAYERS}
        with hook_ctx(list(gs.values())):
            R.model(b, use_cache=False)
        idx = t_remap[b.reshape(-1)]
        t_cnts.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float))
        for L in LAYERS:
            o = gs[L].value.reshape(-1, D).float()
            t_sums[L].index_add_(0, idx, o)
            t_samp[L][idx] = o
TAB["remap"] = t_remap
TAB["mean"] = {L: t_sums[L] / t_cnts[:, None].clamp(min=1) for L in LAYERS}
TAB["samp"] = t_samp
TAB["glob"] = {L: t_sums[L].sum(0) / t_cnts.sum() for L in LAYERS}

exs = []
for i, (ti, di) in enumerate([(2, 5), (5, 2), (1, 6), (6, 1)]):
    exs += make_examples(25, 8, seed=11 + i, target_pos=ti, distract_pos=di)
groups = group_examples(R, exs)
tcov = float((t_remap[torch.cat([g.clean_ids.reshape(-1) for g in groups])] >= 0).float().mean())
print(f"  task token-type table covers {tcov:.3f} of evaluation-prompt tokens", flush=True)
accs = {}
for mode in ("intact", "token_sample", "token_mean", "global_mean", "zero"):
    ok = tot = 0
    dcs = []
    for g in groups:
        hooks = () if mode == "intact" else [Surrogate(0, mode)]
        ds_ = []
        for ids, want in ((g.clean_ids, g.aid), (g.corr_ids, g.cid)):
            CUR["ids"] = ids
            with torch.no_grad(), hook_ctx(hooks):
                lg = R.model(ids, use_cache=False).logits[:, -1]
            ok += (lg.argmax(-1) == want).sum().item(); tot += ids.shape[0]
            ds_.append(g.D(lg))
        dcs += (ds_[0] - ds_[1]).tolist()
    accs[mode] = dict(acc=ok / tot, gap=sum(dcs) / len(dcs))
    print(f"  L0 <- {mode:13s} accuracy {ok/tot:.3f}   D_clean-D_corr {accs[mode]['gap']:+.2f}",
          flush=True)
rows.append(dict(name="retrieval_ladder", coverage=tcov, **accs))

json.dump(dict(meta=vars(a), coverage=sum(cover) / len(cover), rows=rows), open(a.out, "w"), indent=2)
print(f"\nbaseline PPL {base['ppl']:.2f}")
