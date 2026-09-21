import torch


def logit_diff(logits, ans_id, corr_id):
    """D = logit(clean answer) - logit(corrupted answer), at the final position."""
    return (logits[..., ans_id] - logits[..., corr_id]).float().item()


def recovery(d_patched, d_corr, d_clean):
    """Fraction of the clean-corrupted gap restored by the patch."""
    denom = d_clean - d_corr
    if abs(denom) < 1e-6:
        return float("nan")
    return (d_patched - d_corr) / denom
