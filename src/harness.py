"""Batched experiment helpers: groups of examples that share token positions."""
import torch
from .positions import token_positions


class Group:
    """A batch of examples that share `target_idx`/`distract_idx`, hence token positions."""

    def __init__(self, R, exs):
        self.R, self.exs = R, exs
        p0 = token_positions(R.tok, exs[0])
        for e in exs[1:]:
            p = token_positions(R.tok, e)
            assert p["target_value"] == p0["target_value"] and p["n_tokens"] == p0["n_tokens"], \
                "examples in a group must share token positions"
            assert p["distract_value"] == p0["distract_value"]
        self.pos = p0
        self.clean_ids = R.tok([e.clean_prompt() for e in exs], return_tensors="pt").input_ids.to(R.device)
        self.corr_ids = R.tok([e.corrupt_prompt() for e in exs], return_tensors="pt").input_ids.to(R.device)
        self.aid = torch.tensor([R.tok(e.answer).input_ids[0] for e in exs], device=R.device)
        self.cid = torch.tensor([R.tok(e.answer_corr).input_ids[0] for e in exs], device=R.device)

    def D(self, logits):
        """logits: [B, V] at the final position -> per-example logit difference."""
        lg = logits.float()
        return (lg.gather(1, self.aid[:, None]) - lg.gather(1, self.cid[:, None])).squeeze(1)


def group_examples(R, exs):
    """Split a list of examples into Groups keyed by (target_idx, distract_idx)."""
    buckets = {}
    for e in exs:
        buckets.setdefault((e.target_idx, e.distract_idx), []).append(e)
    return [Group(R, v) for v in buckets.values()]


def recovery(d_patch, d_corr, d_clean):
    return ((d_patch - d_corr) / (d_clean - d_corr)).cpu()
