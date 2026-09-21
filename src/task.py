"""Synthetic dictionary key->value retrieval task with clean/corrupted pairs."""
import random
from dataclasses import dataclass

MODEL_PATH = "/user/yac/LinearSwap/models/Qwen3.5-0.8B"

# Candidate keys (appear as " key" in the list) and values (appear as "value" right after '=').
# Both lists were filtered so every word is a single token in the rendered context.
KEY_WORDS = ["apple", "banana", "cherry", "date", "elder", "fig", "grape", "lemon", "mango",
             "peach", "pear", "plum", "berry", "onion", "garlic", "ginger", "pepper", "tomato",
             "carrot", "potato", "walnut", "almond"]
VALUE_WORDS = ["red", "blue", "green", "gold", "gray", "black", "white", "orange", "yellow",
               "amber", "rose", "lime", "olive", "azure", "steel", "ice", "fire", "stone",
               "wood", "iron"]

CHAT_PRE = "<|im_start|>user\n"
CHAT_POST = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def render(pairs, query_key, template="chat"):
    """pairs: list[(key, value)]; returns the prompt ending right before the answer token."""
    kv_inline = ",".join(f" {k}={v}" for k, v in pairs)
    kv_lines = "".join(f"* {k}={v}\n" for k, v in pairs)
    if template == "chat":
        body = "Dictionary:" + kv_inline + ".\n" + f"Question: What is the value of {query_key}?"
        return CHAT_PRE + body + CHAT_POST + f"The value of {query_key} is **"
    if template == "chat_alt":
        body = "Lookup table:\n" + kv_lines + f"Which value does {query_key} map to?"
        return CHAT_PRE + body + CHAT_POST + f"The entry {query_key} maps to **"
    if template == "raw":
        body = "Dictionary:" + kv_inline + ".\n"
        return body + f"Question: What is the value of {query_key}?\nAnswer: "
    raise ValueError(template)


@dataclass
class Example:
    pairs: list           # clean (key, value) pairs, in presentation order
    pairs_corr: list      # corrupted pairs (two values swapped)
    query_key: str
    answer: str           # clean answer value
    answer_corr: str      # corrupted answer value (the swapped-in one)
    target_idx: int       # index in `pairs` of the queried entry
    distract_idx: int     # index of the entry whose value was swapped in
    template: str = "chat"

    def clean_prompt(self):
        return render(self.pairs, self.query_key, self.template)

    def corrupt_prompt(self):
        return render(self.pairs_corr, self.query_key, self.template)

    def answer_strs(self):
        """Exact surface forms of the two answer tokens (template-dependent spacing)."""
        return self.answer, self.answer_corr


def make_examples(n, n_pairs, seed=0, template="chat", target_pos=None, distract_pos=None):
    """target_pos / distract_pos fix the entry indices (otherwise random)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        keys = rng.sample(KEY_WORDS, n_pairs)
        vals = rng.sample(VALUE_WORDS, n_pairs)
        pairs = list(zip(keys, vals))
        ti = target_pos if target_pos is not None else rng.randrange(n_pairs)
        di = distract_pos if distract_pos is not None else rng.choice(
            [i for i in range(n_pairs) if i != ti])
        assert di != ti
        pc = list(pairs)
        pc[ti] = (keys[ti], vals[di])
        pc[di] = (keys[di], vals[ti])
        out.append(Example(pairs, pc, keys[ti], vals[ti], vals[di], ti, di, template))
    return out


def verify_single_token(tok, ex):
    """Clean/corrupt prompts must have equal token counts, differ at exactly the two value
    positions, and both answer values must be single tokens matching those positions."""
    a = tok(ex.clean_prompt()).input_ids
    b = tok(ex.corrupt_prompt()).input_ids
    if len(a) != len(b):
        return False, "length mismatch"
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(diff) != 2:
        return False, f"{len(diff)} differing positions"
    sa, sc = ex.answer_strs()
    ans, ansc = tok(sa).input_ids, tok(sc).input_ids
    if len(ans) != 1 or len(ansc) != 1:
        return False, "multi-token answer"
    if {a[diff[0]], a[diff[1]]} != {ans[0], ansc[0]}:
        return False, "value tokens not single/aligned"
    return True, diff
