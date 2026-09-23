"""Task variants for the generalisation battery (Part XVI).

Every variant produces clean/corrupted prompt pairs of identical token length, with token
positions recovered from character spans via the fast tokenizer's offset mapping, so any
template works without hand-written position bookkeeping.

Variants
  chat{N}        the original dictionary task with N pairs (4 / 8 / 16)
  list8          second template: a bulleted lookup table and a differently phrased question
  rev8           value before key: "gray is the value of grape"
  perm8          original template, target entry at every index 0..7 (incl. first and last)
  long{L}        chat8 with ~L tokens of WikiText between the dictionary and the question
  mmlu           MMLU, closed book: corrupt = swap the correct option's text with a distractor
                 of equal token length (the correct letter changes). Prompt = the standard
                 Hendrycks / lm-evaluation-harness format: subject header + 5 dev-split
                 examples + the question (--mmlu_shots changes the count).
  mmlu_hint      MMLU with "Hint: the answer is <text>." first -- content -> letter retrieval
                 from context, no knowledge needed
Entries: dictionary pairs, or MMLU options. Each entry has a "value" token (the dictionary value,
or the option letter), and an entry span (all of its tokens).
"""
import random
from dataclasses import dataclass, field
import torch
from .task import KEY_WORDS, VALUE_WORDS

CHAT_PRE = "<|im_start|>user\n"
CHAT_POST = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def chat_wrap(tok):
    """(prefix, suffix) that put a user turn and open the assistant turn, for any model.

    Qwen3.5 keeps the hand-written strings above (empty think block, so the answer is the next
    token) so earlier results stay comparable; other families are derived from the tokenizer's
    own chat template."""
    tpl = getattr(tok, "chat_template", None) or ""
    if "<|im_start|>" in tpl:
        return CHAT_PRE, CHAT_POST
    s = tok.apply_chat_template([{"role": "user", "content": "\x00"}], tokenize=False,
                                add_generation_prompt=True)
    pre, post = s.split("\x00", 1)
    return pre, post
LETTERS = "ABCD"


@dataclass
class Item:
    clean: str
    corr: str
    answer: str                  # surface form of the next token (clean)
    answer_corr: str
    target_idx: int
    distract_idx: int
    value_spans: list            # per entry: (start, end) char span of its value/letter token
    entry_spans: list            # per entry: (start, end) char span of the whole entry
    meta: dict = field(default_factory=dict)


class Builder:
    """Append text pieces while recording character spans."""

    def __init__(self): self.s = ""
    def add(self, t):
        a = len(self.s); self.s += t
        return a, len(self.s)


def _dict_item(pairs, pairs_c, ti, di, style, filler="", wrap=(CHAT_PRE, CHAT_POST)):
    def render(ps):
        b = Builder(); vs, es = [], []
        b.add(wrap[0])
        q = ps[ti][0]
        if style == "list":
            b.add("Lookup table:\n")
            for k, v in ps:
                e0, _ = b.add(f"* {k}="); v0, v1 = b.add(v); _, e1 = b.add("\n")
                vs.append((v0, v1)); es.append((e0, e1))
            b.add(filler)
            b.add(f"Which value does {q} map to?" + wrap[1] + f"The entry {q} maps to **")
        elif style == "rev":
            b.add("Dictionary:")
            for i, (k, v) in enumerate(ps):
                e0, _ = b.add(" "); v0, v1 = b.add(v); _, e1 = b.add(f" is the value of {k}")
                b.add("," if i < len(ps) - 1 else ".")
                vs.append((v0, v1)); es.append((e0, e1))
            b.add(filler)
            b.add(f"\nQuestion: What is the value of {q}?" + wrap[1] + f"The value of {q} is **")
        else:
            b.add("Dictionary:")
            for i, (k, v) in enumerate(ps):
                e0, _ = b.add(f" {k}="); v0, v1 = b.add(v); _, e1 = b.add("," if i < len(ps) - 1 else ".")
                vs.append((v0, v1)); es.append((e0, e1))
            b.add(filler)
            b.add(f"\nQuestion: What is the value of {q}?" + wrap[1] + f"The value of {q} is **")
        return b.s, vs, es
    c, vs, es = render(pairs)
    x, _, _ = render(pairs_c)
    return Item(c, x, pairs[ti][1], pairs_c[ti][1], ti, di, vs, es)


def _mmlu_prefix(dev_rows, subject, n_shot):
    """Standard MMLU prompt prefix: subject header + n_shot solved dev examples."""
    head = ("The following are multiple choice questions (with answers) about "
            f"{subject.replace('_', ' ')}.\n\n")
    for r in dev_rows[:n_shot]:
        head += r["question"].strip() + "\n"
        for L, o in zip(LETTERS, r["choices"]):
            head += f"{L}. {o.strip()}\n"
        head += f"Answer: {LETTERS[r['answer']]}\n\n"
    return head


def _mmlu_item(row, di, hint, prefix=""):
    opts = list(row["choices"]); ci = row["answer"]
    oc = list(opts); oc[ci], oc[di] = opts[di], opts[ci]

    def render(os_):
        b = Builder(); vs, es = [], []
        b.add(prefix)
        if hint:
            b.add(f"Hint: the answer is {opts[ci].strip()}.\n")
        b.add(row["question"].strip() + "\n")
        for L, o in zip(LETTERS, os_):
            v0, v1 = b.add(L); _, e1 = b.add(". " + o.strip()); b.add("\n")
            vs.append((v0, v1)); es.append((v0, e1))
        b.add("Answer:")
        return b.s, vs, es
    c, vs, es = render(opts)
    x, _, _ = render(oc)
    return Item(c, x, " " + LETTERS[ci], " " + LETTERS[di], ci, di, vs, es,
                meta=dict(subject=row["subject"]))


def filler_text(tok, n_tokens):
    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
    ids = tok("\n\n".join(t for t in d["text"][200:] if t.strip())).input_ids[:n_tokens]
    return "\n" + tok.decode(ids).strip() + "\n"


def make_items(variant, tok, n_per_cfg=25, seed=0, mmlu_shots=5):
    wrap = chat_wrap(tok)
    rng = random.Random(seed)
    if variant.startswith("mmlu"):
        from datasets import load_dataset
        d = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed)
        dev = {}
        if mmlu_shots:
            for r in load_dataset("cais/mmlu", "all", split="dev"):
                dev.setdefault(r["subject"], []).append(r)
        out = []
        for row in d:
            if any("\n" in o for o in row["choices"]) or len(row["question"]) > 600:
                continue
            ci = row["answer"]
            L = lambda o: len(tok(" " + o.strip()).input_ids)
            same = [j for j in range(4) if j != ci and L(row["choices"][j]) == L(row["choices"][ci])
                    and row["choices"][j].strip() != row["choices"][ci].strip()]
            if not same:
                continue
            pre = (_mmlu_prefix(dev.get(row["subject"], []), row["subject"], mmlu_shots)
                   if mmlu_shots else "")
            it = _mmlu_item(row, rng.choice(same), hint=(variant == "mmlu_hint"), prefix=pre)
            if it is not None:
                out.append(it)
            if len(out) >= 1500:
                break
        return out

    style, n_pairs, filler = "chat", 8, ""
    if variant.startswith("chat"):
        n_pairs = int(variant[4:])
    elif variant == "list8":
        style = "list"
    elif variant == "rev8":
        style = "rev"
    elif variant.startswith("long"):
        filler = filler_text(tok, int(variant[4:]))
    cfgs = {4: [(0, 3), (3, 0), (1, 2), (2, 1)], 16: [(3, 12), (12, 3), (6, 9), (9, 6)]}.get(
        n_pairs, [(2, 5), (5, 2), (1, 6), (6, 1)])
    if variant == "perm8":
        cfgs = [(t, rng.choice([j for j in range(8) if j != t])) for t in range(8)]
    vals = [v for v in VALUE_WORDS if len(tok(v).input_ids) == 1 and len(tok(" " + v).input_ids) == 1]
    out = []
    for ti, di in cfgs:
        for _ in range(n_per_cfg):
            keys = rng.sample(KEY_WORDS, n_pairs)
            vs = rng.sample(vals, n_pairs) if n_pairs <= len(vals) else None
            pairs = list(zip(keys, vs))
            pc = list(pairs)
            pc[ti] = (keys[ti], vs[di]); pc[di] = (keys[di], vs[ti])
            out.append(_dict_item(pairs, pc, ti, di, style, filler, wrap))
    return out


def positions(tok, it):
    """Token positions from the clean prompt's char spans. Returns None unless clean and corrupted
    prompts have equal token length and differ only inside the target and distractor entries,
    every value/letter is a single token, and both answers are single tokens."""
    ec = tok(it.clean, return_offsets_mapping=True)
    ex = tok(it.corr, return_offsets_mapping=True)
    if len(ec.input_ids) != len(ex.input_ids):
        return None
    off = ec.offset_mapping

    def toks(span):
        a, b = span
        return [i for i, (s, e) in enumerate(off) if s < b and e > a]
    vpos, epos = [], []
    for vs, es in zip(it.value_spans, it.entry_spans):
        v = toks(vs)
        if len(v) != 1:
            return None
        vpos.append(v[0]); epos.append(toks(es))
    diff = {i for i, (x, y) in enumerate(zip(ec.input_ids, ex.input_ids)) if x != y}
    if not diff or not diff <= set(epos[it.target_idx]) | set(epos[it.distract_idx]):
        return None
    ans = tok(it.answer).input_ids, tok(it.answer_corr).input_ids
    if len(ans[0]) != 1 or len(ans[1]) != 1:
        return None
    return dict(n=len(ec.input_ids), final=len(ec.input_ids) - 1, value_pos=vpos, entry_pos=epos,
                ids=ec.input_ids, ids_c=ex.input_ids, aid=ans[0][0], cid=ans[1][0])


class Batch:
    """Items sharing length and all positions, run together."""

    def __init__(self, items, poss, device):
        p = poss[0]
        self.items, self.pos = items, p
        self.ti = items[0].target_idx
        self.clean_ids = torch.tensor([q["ids"] for q in poss], device=device)
        self.corr_ids = torch.tensor([q["ids_c"] for q in poss], device=device)
        self.aid = torch.tensor([q["aid"] for q in poss], device=device)
        self.cid = torch.tensor([q["cid"] for q in poss], device=device)

    def D(self, logits):
        lg = logits.float()
        return (lg.gather(1, self.aid[:, None]) - lg.gather(1, self.cid[:, None])).squeeze(1)


def batches(tok, items, device, max_bs=25):
    buckets = {}
    for it in items:
        p = positions(tok, it)
        if p is None:
            continue
        key = (p["n"], tuple(p["value_pos"]), tuple(map(tuple, p["entry_pos"])), it.target_idx)
        buckets.setdefault(key, []).append((it, p))
    out = []
    for v in buckets.values():
        for i in range(0, len(v), max_bs):
            ch = v[i:i + max_bs]
            out.append(Batch([c[0] for c in ch], [c[1] for c in ch], device))
    return out
