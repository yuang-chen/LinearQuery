"""Shared machinery for the minimal-hybrid experiments (exp25) and the cross-family transfer (exp26/27).

A *configuration* of a hybrid is (cut, keep, removal):
  cut      last layer that runs; layers after it are skipped entirely (GnA's "minimal model":
           the residual stream after `cut` goes straight to the final norm and the LM head)
  keep     the linear (GDN / Mamba-2) layers whose token mixer stays; every other linear layer
           at or before `cut` has its mixer removed. Softmax layers and all MLPs are kept.
  removal  how a mixer is removed:
             zero  output set to 0 (Parts VII-XVIII)
             mean  output replaced by its WikiText mean (a constant bias: no token mixing, no state)
           a removed *layer* (mixer and MLP) is only used for the truncation.

Evaluators: the dictionary suite (src/gen.py variants, acc over clean+corrupted, and gap), the
GnA KV-retrieval task (answer scoring over the dictionary values, trailing space restored),
WikiText-2 perplexity, and lm-evaluation-harness tasks with the hooks active.
"""
import random
import numpy as np
import torch
from .runner import hook_ctx
from .gen import make_items, batches


# ------------------------------------------------------------------ hooks
def _t(out):
    return out[0] if isinstance(out, tuple) else out


def _put(out, t):
    return (t,) + tuple(out[1:]) if isinstance(out, tuple) else t


class ZeroMixer:
    def __init__(self, R, L): self.R, self.L = R, L
    def register(self):
        return self.R.mixer(self.L).register_forward_hook(
            lambda m, a, o: _put(o, torch.zeros_like(_t(o))))


class MeanMixer:
    def __init__(self, R, L, mu): self.R, self.L, self.mu = R, L, mu
    def register(self):
        def fn(m, a, o):
            t = _t(o)
            return _put(o, self.mu.to(t.dtype).expand_as(t).contiguous())
        return self.R.mixer(self.L).register_forward_hook(fn)


class SkipLayer:
    def __init__(self, R, L): self.R, self.L = R, L
    def register(self):
        def fn(mod, args, kwargs, output):
            h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            return _put(output, h)
        return self.R.layers[self.L].register_forward_hook(fn, with_kwargs=True)


@torch.no_grad()
def mixer_means(R, n_seq=16, seq_len=512):
    """Mean mixer output of every linear layer over WikiText-2 train tokens (intact model)."""
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
    ids = R.tok("\n\n".join(t for t in d["text"][:4000] if t.strip()),
                return_tensors="pt").input_ids[0]
    sums = {L: 0 for L in R.gdn_layers}; cnt = 0
    caps = {}

    class Cap:
        def __init__(s, L): s.L = L
        def register(s):
            def fn(m, a, o):
                caps[s.L] = _t(o).float().sum((0, 1))
            return R.mixer(s.L).register_forward_hook(fn)
    for i in range(n_seq):
        x = ids[i * seq_len:(i + 1) * seq_len].unsqueeze(0).to(R.device)
        with hook_ctx([Cap(L) for L in R.gdn_layers]):
            R.model(x, use_cache=False)
        for L in R.gdn_layers:
            sums[L] = sums[L] + caps[L]
        cnt += x.shape[1]
    return {L: (sums[L] / cnt) for L in R.gdn_layers}


def config_hooks(R, cut=None, keep=None, removal="mean", means=None):
    """Hooks realising a configuration. cut=None: full depth. keep=None: every linear layer kept."""
    hooks = []
    last = R.n_layers - 1 if cut is None else cut
    for L in range(last + 1, R.n_layers):
        hooks.append(SkipLayer(R, L))
    if keep is not None:
        for L in R.gdn_layers:
            if L > last or L in keep:
                continue
            hooks.append(ZeroMixer(R, L) if removal == "zero" else MeanMixer(R, L, means[L]))
    return hooks


# ------------------------------------------------------------------ dictionary suite
class DictSuite:
    """acc: full-vocabulary argmax is the answer. cacc: GnA-style choice scoring -- the answer
    has the highest logit among the dictionary's values (chance 1/n_pairs)."""

    def __init__(self, R, variants=("chat8", "chat16", "list8", "rev8", "long512"), seed=0,
                 per_cfg=25, max_bs=25):
        self.R = R
        self.B = {}
        for v in variants:
            items = make_items(v, R.tok, n_per_cfg=per_cfg, seed=seed)
            self.B[v] = batches(R.tok, items, R.device, max_bs)
            for b in self.B[v]:          # candidates = the bare value words (the answer's form)
                b.cand = torch.tensor([[R.tok(it.clean[s0:s1]).input_ids[0] for s0, s1 in it.value_spans]
                                       for it in b.items], device=R.device)

    @torch.no_grad()
    def run(self, hooks=(), variants=None):
        out = {}
        with hook_ctx(hooks):
            for v, B in self.B.items():
                if variants and v not in variants:
                    continue
                ok = cok = tot = 0; gaps = []
                for b in B:
                    ds = []
                    for ids, want in ((b.clean_ids, b.aid), (b.corr_ids, b.cid)):
                        lg = self.R.model(ids, use_cache=False).logits[:, -1].float()
                        ok += (lg.argmax(-1) == want).sum().item(); tot += ids.shape[0]
                        cand = b.cand                            # the dictionary's values
                        pick = cand.gather(1, lg.gather(1, cand).argmax(-1, keepdim=True))[:, 0]
                        cok += (pick == want).sum().item()
                        ds.append(b.D(lg))
                    gaps += (ds[0] - ds[1]).tolist()
                out[v] = dict(acc=ok / tot, cacc=cok / tot, gap=float(np.mean(gaps)))
        return out


# ------------------------------------------------------------------ GnA KV retrieval
class KVRetrieval:
    """arXiv:2504.18574 App. A.5, variant (2) (trailing space), answer scoring over the values.
    Batched: the prompt is run once and every choice is scored from the prompt's cache-free
    concatenation, as in their `logprob_of_sequence`."""

    def __init__(self, R, pairs=20, n=200, seed=0, trailing_space=True, batch_size=32):
        from wonderwords import RandomWord
        self.R, self.bs = R, batch_size
        random.seed(seed)
        self.ex = []
        for _ in range(n):
            keys = RandomWord().random_words(pairs)
            vals = random.sample(range(100), pairs)
            k = random.choice(keys)
            q = ("Memorize the following dictionary:\n" + "\n".join(f"{a}:{b}" for a, b in zip(keys, vals))
                 + f"\nThe value of the key '{k}' is" + (" " if trailing_space else ""))
            ch = [str(v) for v in vals]
            self.ex.append(dict(q=q, choices=ch, ans=ch.index(str(vals[keys.index(k)]))))
        tok = R.tok
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.nspec = len(tok("x").input_ids) - len(tok("x", add_special_tokens=False).input_ids)

    @torch.no_grad()
    def _lp(self, prompts, comps):
        tok = self.R.tok
        tok.padding_side = "right"
        pl = [len(e) for e in tok(prompts, add_special_tokens=False)["input_ids"]]
        full = tok([p + c for p, c in zip(prompts, comps)], return_tensors="pt", padding=True)
        ids = full["input_ids"].to(self.R.device); am = full["attention_mask"].to(self.R.device)
        lp = torch.log_softmax(self.R.model(input_ids=ids, attention_mask=am, use_cache=False)
                               .logits.float(), -1)
        out = []
        for i in range(len(prompts)):
            n = int(am[i].sum()); s0 = pl[i] + self.nspec
            out.append(float(sum(lp[i, j - 1, ids[i, j]] for j in range(s0, n))))
        return torch.tensor(out)

    @torch.no_grad()
    def run(self, hooks=()):
        cors = []
        with hook_ctx(hooks):
            for i in range(0, len(self.ex), self.bs):
                b = self.ex[i:i + self.bs]
                nc = len(b[0]["choices"])
                lps = torch.stack([self._lp([e["q"] for e in b], [e["choices"][c] for e in b])
                                   for c in range(nc)], 1)
                cors += (lps.argmax(-1).numpy() == np.array([e["ans"] for e in b])).tolist()
        return float(np.mean(cors))


# ------------------------------------------------------------------ perplexity
class WikiPPL:
    def __init__(self, R, n_seq=16, seq_len=512):
        from datasets import load_dataset
        d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
        ids = R.tok("\n\n".join(t for t in d["text"] if t.strip()), return_tensors="pt").input_ids[0]
        self.R = R
        self.x = torch.stack([ids[i * seq_len:(i + 1) * seq_len] for i in range(n_seq)]).to(R.device)

    @torch.no_grad()
    def run(self, hooks=(), bs=4):
        nll = []
        with hook_ctx(hooks):
            for i in range(0, len(self.x), bs):
                x = self.x[i:i + bs]
                lg = self.R.model(x, use_cache=False).logits[:, :-1].float()
                nll.append(torch.nn.functional.cross_entropy(
                    lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="none"))
        return float(torch.exp(torch.cat(nll).mean()))


# ------------------------------------------------------------------ lm-eval
KNOWLEDGE = ["arc_easy", "arc_challenge", "piqa", "hellaswag", "winogrande"]


def lm_eval_tasks(R, hooks=(), tasks=KNOWLEDGE, limit=500, num_fewshot=0, batch_size=16):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    lm = HFLM(pretrained=R.model, tokenizer=R.tok, batch_size=batch_size)
    with hook_ctx(hooks):
        res = lm_eval.simple_evaluate(model=lm, tasks=list(tasks), num_fewshot=num_fewshot,
                                      limit=limit, log_samples=False, verbosity="ERROR")
    out = {}
    for t, r in res["results"].items():
        m = next((k for k in ("acc_norm,none", "acc,none", "contains,none", "exact_match,none")
                  if k in r), None)
        if m:
            out[t] = float(r[m])
    return out
