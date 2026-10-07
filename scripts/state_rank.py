#!/usr/bin/env python
"""Singular values of every linear-attention memory state along a prefill, for any model of this
project (adapted from LinearSwap tools/state_rank.py).

The state of one head is a matrix: Gated DeltaNet (Qwen3.5) key_dim x value_dim = 128 x 128;
Mamba-2 (Granite-4.0-H) head_dim x state_dim = 64 x 128. It is read from the HF cache
(`cache.layers[i].recurrent_states[0]`, [B, H, *, *]) after each prefill chunk.

Sources
  dclm      LinearSwap's DCLM validation split, packed (default; the same token stream as
            LinearSwap for Qwen tokenizers, decoded and re-tokenised for other families)
  wikitext  WikiText-2 train, packed
  longbench one long document per sequence (LongBench-v2, Single-Document QA, English, not JSON):
            the first --batch qualifying documents in dataset order, each cut to the last mark,
            so "rank vs length" is not confounded by document switching (DCLM packs 11-19
            documents into one 16k window)
  dict      the dictionary task (src/gen.py variant, clean prompts): states at the end of the
            dictionary and at the final token, i.e. where the Bind-Query circuit reads them

    python scripts/state_rank.py --tag 0.8B
    python scripts/state_rank.py --tag G1b --model models/granite-4.0-h-1b
    python scripts/state_rank.py --tag 0.8B --source dict --variant chat16
`scripts/state_rank_report.py` turns the dumps into ranks.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from transformers import AutoTokenizer, DynamicCache

sys.path.insert(0, "/mnt/yuang/LinearQuery")
from src.runner import Runner
from src.circuit import CIRCUIT, role

QWEN_TOK = "/mnt/yuang/gdn2-in-place/models/Qwen3.5-0.8B"           # tokenizer of the DCLM split
DCLM = "/mnt/yuang/gdn2-in-place/data/text/dclm-10shard/validation"

p = argparse.ArgumentParser()
p.add_argument("--model", default="/mnt/yuang/gdn2-in-place/models/Qwen3.5-0.8B")
p.add_argument("--tag", required=True, help="0.8B | 9B | G1b | Gtiny (circuit roles come from src/circuit.py)")
p.add_argument("--source", default="dclm", choices=["dclm", "wikitext", "longbench", "dict"])
p.add_argument("--variant", default="chat16", help="dictionary variant for --source dict")
p.add_argument("--batch", type=int, default=8)
p.add_argument("--marks", default="64,128,256,1024,4096,16384", help="prefix lengths at which to read the state")
p.add_argument("--dtype", default="fp32")
p.add_argument("--out", default="results/state_rank")
a = p.parse_args()
torch.set_num_threads(8)        # the per-mark CPU SVD; the default (all 224 cores per process) thrashes
name = a.tag if a.source == "dclm" else f"{a.tag}_{a.source}"

t0 = time.time()
R = Runner(model_path=a.model, dtype=torch.float32 if a.dtype == "fp32" else torch.bfloat16)
lin = R.gdn_layers
cfg = R.model.config.get_text_config()
res = {"name": name, "tag": a.tag, "model": a.model, "source": a.source, "layers": lin,
       "roles": {str(i): role(a.tag, i) for i in lin}, "circuit": CIRCUIT[a.tag], "sv": {}}


def read_states(mark):
    shape = None
    for i in lin:
        st = cache.layers[i].recurrent_states[0]                       # [B, H, d1, d2]
        shape = list(st.shape[-2:])
        # fp32 LAPACK on the CPU: cuSOLVER's batched Jacobi path only covers small square matrices, and
        # Mamba-2's 64x128 states fall back to a per-matrix GPU loop that takes minutes per mark
        res["sv"].setdefault(f"{i}@{mark}", []).extend(torch.linalg.svdvals(st.float().cpu()).tolist())
    res["state_shape"] = shape


with torch.no_grad():
    if a.source == "longbench":
        marks = [int(m) for m in a.marks.split(",")]
        from datasets import load_dataset
        lb = load_dataset("THUDM/LongBench-v2", split="train")
        rows, docs_used = [], []
        for x in lb:
            c = x["context"]
            if x["domain"] != "Single-Document QA" or c.lstrip().startswith("{") or len(c) < 120000 \
                    or sum(ch.isascii() for ch in c[:20000]) / 20000 < 0.97:
                continue
            t = R.tok(c, add_special_tokens=False).input_ids
            if len(t) < marks[-1]:
                continue
            rows.append(t[:marks[-1]]); docs_used.append(dict(id=x["_id"], sub_domain=x["sub_domain"]))
            if len(rows) == a.batch:
                break
        assert len(rows) == a.batch, f"only {len(rows)} long documents"
        res["docs"] = docs_used
        ids = torch.tensor(rows).to(R.device)
        cache = DynamicCache(config=cfg)
        pos = 0
        for m in marks:
            R.model(ids[:, pos:m], past_key_values=cache, use_cache=True, logits_to_keep=1,
                    position_ids=torch.arange(pos, m, device=R.device)[None].expand(a.batch, -1))
            pos = m
            read_states(m)
            print(f"[state_rank] {name}: {m} tokens ({(time.time() - t0) / 60:.1f} min)", flush=True)
        res["marks"] = marks
    elif a.source in ("dclm", "wikitext"):
        marks = [int(m) for m in a.marks.split(",")]
        need = a.batch * marks[-1]
        if a.source == "dclm":
            from datasets import load_from_disk
            docs = load_from_disk(DCLM)["input_ids"]
            qtok = AutoTokenizer.from_pretrained(QWEN_TOK)
            probe = "Linear attention builds the query."
            if R.tok(probe, add_special_tokens=False).input_ids == qtok(probe, add_special_tokens=False).input_ids \
                    and len(R.tok) == len(qtok):
                flat = [t for doc in docs for t in doc]                 # same tokenizer: LinearSwap's stream
            else:
                flat = []                                               # re-tokenise; each document
                for doc in docs:                                        # ends in *this* model's EOS
                    flat += R.tok(qtok.decode(doc, skip_special_tokens=True), add_special_tokens=False).input_ids
                    flat.append(R.tok.eos_token_id)
                    if len(flat) >= need:
                        break
        else:
            from datasets import load_dataset
            d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            flat = R.tok("\n\n".join(t for t in d["text"] if t.strip()), add_special_tokens=False).input_ids
        assert len(flat) >= need, f"{a.source}: {len(flat)} tokens < {need}"
        ids = torch.tensor(flat[:need]).view(a.batch, marks[-1]).to(R.device)
        cache = DynamicCache(config=cfg)
        pos = 0
        for m in marks:
            R.model(ids[:, pos:m], past_key_values=cache, use_cache=True, logits_to_keep=1,
                    position_ids=torch.arange(pos, m, device=R.device)[None].expand(a.batch, -1))
            pos = m
            read_states(m)
            print(f"[state_rank] {name}: {m} tokens ({(time.time() - t0) / 60:.1f} min)", flush=True)
        res["marks"] = marks
    else:
        from src.gen import make_items, batches
        B = batches(R.tok, make_items(a.variant, R.tok, seed=0), R.device, a.batch)
        for b in B:
            ids = b.clean_ids
            dict_end = max(max(e) for e in b.pos["entry_pos"])          # last token of the dictionary
            cache = DynamicCache(config=cfg)
            R.model(ids[:, :dict_end + 1], past_key_values=cache, use_cache=True, logits_to_keep=1)
            read_states("dict_end")
            n = ids.shape[1]
            R.model(ids[:, dict_end + 1:], past_key_values=cache, use_cache=True, logits_to_keep=1,
                    position_ids=torch.arange(dict_end + 1, n, device=R.device)[None].expand(ids.shape[0], -1))
            read_states("final")
        res["marks"] = ["dict_end", "final"]
        res["n_prompts"] = sum(len(b.items) for b in B)
        print(f"[state_rank] {name}: {res['n_prompts']} {a.variant} prompts", flush=True)

Path(a.out).mkdir(parents=True, exist_ok=True)
json.dump(res, open(Path(a.out) / f"{name}.json", "w"))
print(f"[state_rank] wrote {Path(a.out) / f'{name}.json'} ({(time.time() - t0) / 60:.1f} min)", flush=True)
