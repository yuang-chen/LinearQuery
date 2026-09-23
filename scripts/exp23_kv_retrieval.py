"""Step 23: the KV-Retrieval recipe of Arora et al.'s G&A paper, run on our hybrid models.

Strict port of https://github.com/goombalab/Gather-and-Aggregate (`kv_retrieval.py`, `utils.py`)
and Appendix A.5 of arXiv:2504.18574. Nothing about the task is our own:

  prompt      "Memorize the following dictionary:\\n" + "\\n".join(f"{k}:{v}") +
              f"\\nThe value of the key '{selected_key}' is"          (no trailing space)
  keys        wonderwords.RandomWord().random_words(num_pairs)
  values      random.sample(range(100), num_pairs)                     (distinct, 0..99)
  choices     every value in the dictionary, as a string
  answer      the value of the queried key

  Answer Scoring (their `eval_kv` / `logprob_of_sequence`): each choice is appended to the
  prompt, the summed log-probability of the appended tokens is computed, and the arg-max choice
  is the prediction. Prompt length comes from `tokenizer(prompt, add_special_tokens=False)`.
  Generation: generate 10 tokens greedily and check whether the correct value appears anywhere.

  Head ablation (their `remove_heads`): zero the head's slice of the output-projection *weight*,
  evaluate, then restore. Their grid over dictionary sizes is `range(10, 55, 5)`.

Extensions, marked as such in the output: (a) the same ablation applied to the linear mixer's
value heads (GDN / Mamba-2), which is the analogue of their Falcon-Mamba "channels"; (b) head
sweeps may use fewer samples than their 1,000 (--n), because our sweep covers every head of
every softmax layer on four models.
"""
import sys, json, argparse, time, random, torch
import numpy as np
sys.path.insert(0, "/user/yac/LinearAblation")
from src.runner import Runner

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/user/yac/LinearSwap/models/Qwen3.5-0.8B")
ap.add_argument("--tag", default="0.8B")
ap.add_argument("--pairs", default="20", help="dictionary sizes, comma separated (theirs: 10,15,...,50)")
ap.add_argument("--n", type=int, default=1000, help="samples per configuration (theirs: 1000)")
ap.add_argument("--batch_size", type=int, default=64)
ap.add_argument("--trailing_space", action="store_true", help="their prompt variant (2)")
ap.add_argument("--gen", action="store_true", help="also run Generation scoring (10 tokens)")
ap.add_argument("--heads", default="", help="'attn' = sweep every softmax head; 'linear' = every "
                                            "linear-mixer value head; 'both'; '' = no sweep")
ap.add_argument("--head_shard", default="1/1", help="i/j: run only shard i of j (head sweeps)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args()
PAIRS = [int(x) for x in a.pairs.split(",")]
OUT = a.out or f"results/exp23_kv_{a.tag}.json"

t0 = time.time()
R = Runner(model_path=a.model, dtype=torch.float32)
model, tok = R.model, R.tok
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
N_SPECIAL = len(tok("x").input_ids) - len(tok("x", add_special_tokens=False).input_ids)
print(f"[{a.tag}] layers {R.n_layers}  attn {R.attn_layers}  heads {R.cfg.num_attention_heads}"
      f"  linear heads {R.n_linear_heads}  special tokens added: {N_SPECIAL}", flush=True)

from wonderwords import RandomWord


# ----------------------------------------------------------- their MemorizationDataset
class MemorizationDataset:
    def __init__(self, num_pairs=5, num_examples=100):
        self.num_pairs, self.num_examples = num_pairs, num_examples
        self.numbers = list(range(100))

    def __len__(self):
        return self.num_examples

    def __iter__(self):
        for _ in range(self.num_examples):
            yield self.generate_example()

    def generate_example(self):
        from collections import OrderedDict
        keys = RandomWord().random_words(self.num_pairs)
        values = random.sample(self.numbers, self.num_pairs)
        kv = OrderedDict(zip(keys, values))
        selected_key = random.choice(keys)
        answer = kv[selected_key]
        question = ("Memorize the following dictionary:\n"
                    + "\n".join(f"{k}:{v}" for k, v in kv.items())
                    + f"\nThe value of the key '{selected_key}' is"
                    + (" " if a.trailing_space else ""))
        choices = [str(v) for v in kv.values()]
        return dict(question=question, choices=choices, correct_index=choices.index(str(answer)))


# ----------------------------------------------------------- their logprob_of_sequence / eval_kv
@torch.no_grad()
def logprob_of_sequence(prompts, completions):
    prompt_lengths = [len(e) for e in tok(prompts, add_special_tokens=False)["input_ids"]]
    full = tok([p + c for p, c in zip(prompts, completions)], return_tensors="pt", padding=True)
    ids = full["input_ids"].to(model.device)
    am = full["attention_mask"].to(model.device)
    logits = model(input_ids=ids, attention_mask=am).logits.float()
    lp = torch.log_softmax(logits, dim=-1)
    out = []
    for i in range(len(prompts)):
        seq_len = int(am[i].sum())
        start = prompt_lengths[i] + N_SPECIAL      # their code omits N_SPECIAL; 0 for our models
        s = sum(float(lp[i, j - 1, ids[i, j]]) for j in range(start, seq_len))
        out.append(s)
    return torch.tensor(out)


@torch.no_grad()
def eval_kv(dataset):
    cors = []
    for i in range(0, len(dataset), a.batch_size):
        batch = dataset[i:i + a.batch_size]
        if not batch:
            break
        nc = len(batch[0]["choices"])
        lps = torch.stack([logprob_of_sequence([ex["question"] for ex in batch],
                                               [ex["choices"][c] for ex in batch])
                           for c in range(nc)], dim=1)
        preds = lps.argmax(-1).numpy()
        cors += [int(preds[j] == ex["correct_index"]) for j, ex in enumerate(batch)]
    return float(np.mean(cors))


@torch.no_grad()
def eval_kv_generation(dataset, new_tokens=10):
    """Their Generation setting: 10 new tokens, correct value appearing anywhere counts."""
    cors = []
    for i in range(0, len(dataset), a.batch_size):
        batch = dataset[i:i + a.batch_size]
        if not batch:
            break
        tok.padding_side = "left"
        enc = tok([ex["question"] for ex in batch], return_tensors="pt", padding=True).to(model.device)
        tok.padding_side = "right"
        gen = model.generate(**enc, max_new_tokens=new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        cors += [int(ex["choices"][ex["correct_index"]] in t) for ex, t in zip(batch, txt)]
    return float(np.mean(cors))


# ----------------------------------------------------------- their remove_heads (weight masking)
def head_slice(proj, n_heads, head_dim, head):
    """Zero one head's block of columns of an output projection, returning an undo closure."""
    w = proj.weight.data                       # [out, n_heads*head_dim]
    v = w.view(w.shape[0], n_heads, head_dim)
    keep = v[:, head, :].clone()
    v[:, head, :] = 0.0

    def undo():
        v[:, head, :] = keep
    return undo


def attn_heads():
    for L in R.attn_layers:
        for h in range(R.cfg.num_attention_heads):
            yield ("attn", L, h)


def linear_heads():
    for L in R.gdn_layers:
        for h in range(R.n_linear_heads):
            yield ("linear", L, h)


def ablate(kind, L, h):
    m = R.mixer(L)
    if kind == "attn":
        return head_slice(m.o_proj, R.cfg.num_attention_heads, R.head_dim, h)
    return head_slice(m.out_proj, R.n_linear_heads, R.linear_head_dim, h)


# ----------------------------------------------------------- run
res = dict(meta=dict(model=a.model, tag=a.tag, pairs=PAIRS, n=a.n, seed=a.seed,
                     trailing_space=a.trailing_space, n_special=N_SPECIAL,
                     layers=R.n_layers, attn_layers=R.attn_layers,
                     n_attn_heads=R.cfg.num_attention_heads, n_linear_heads=R.n_linear_heads,
                     head_shard=a.head_shard, source="goombalab/Gather-and-Aggregate"))

random.seed(a.seed)
datasets = {p: list(MemorizationDataset(num_pairs=p, num_examples=a.n)) for p in PAIRS}
print(f"  built {len(PAIRS)} configs x {a.n} samples; example prompt tokens: "
      f"{len(tok(datasets[PAIRS[0]][0]['question']).input_ids)}", flush=True)

res["intact"] = {}
for p in PAIRS:
    acc = eval_kv(datasets[p])
    res["intact"][p] = dict(score=acc)
    line = f"C1 pairs {p:3d}  answer-scoring acc {acc:.3f}"
    if a.gen:
        g = eval_kv_generation(datasets[p])
        res["intact"][p]["generation"] = g
        line += f"   generation acc {g:.3f}"
    print(line + f"   ({(time.time() - t0) / 60:.1f} min)", flush=True)

if a.heads:
    kinds = {"attn": list(attn_heads()), "linear": list(linear_heads())}
    todo = (kinds["attn"] if a.heads == "attn" else kinds["linear"] if a.heads == "linear"
            else kinds["attn"] + kinds["linear"])
    i, j = (int(x) for x in a.head_shard.split("/"))
    todo = todo[i - 1::j]
    P = PAIRS[0]
    print(f"C2 head sweep: {len(todo)} heads (shard {a.head_shard}) at pairs={P}, n={a.n}", flush=True)
    res["heads"] = []
    for k, (kind, L, h) in enumerate(todo):
        undo = ablate(kind, L, h)
        acc = eval_kv(datasets[P])
        undo()
        res["heads"].append(dict(kind=kind, layer=L, head=h, score=acc, pairs=P))
        if k % 5 == 0 or acc < res["intact"][P]["score"] - 0.05:
            print(f"   {kind} L{L}H{h}: {acc:.3f}  (intact {res['intact'][P]['score']:.3f})"
                  f"  [{k + 1}/{len(todo)}, {(time.time() - t0) / 60:.1f} min]", flush=True)
    worst = sorted(res["heads"], key=lambda r: r["score"])[:10]
    print("C2 most retrieval-critical: " + "  ".join(
        f"{r['kind'][0]}L{r['layer']}H{r['head']}:{r['score']:.3f}" for r in worst), flush=True)

res["meta"]["seconds"] = time.time() - t0
import os
os.makedirs("results", exist_ok=True)
json.dump(res, open(OUT, "w"), indent=2)
print(f"wrote {OUT}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
