"""Hooks that patch Q / K / V of a softmax-attention layer at selected positions/heads."""
import torch


class ProjPatch:
    """Patch the output of q_proj / k_proj / v_proj at given positions (optionally per head).

    Patching the projection output (before q_norm/k_norm and RoPE) keeps the rest of the
    attention computation intact, so the patched value propagates exactly as a real one.
    q_proj emits [query; gate] interleaved per head as 2*head_dim, so `which='q'` patches
    only the query half unless `include_gate` is set.
    """

    def __init__(self, attn, which, positions, donor, head_dim, heads=None,
                 offset=0, include_gate=False):
        self.attn, self.which, self.positions = attn, which, positions
        self.donor, self.head_dim, self.heads = donor, head_dim, heads
        self.offset, self.include_gate = offset, include_gate

    def _proj(self):
        return {"q": self.attn.q_proj, "k": self.attn.k_proj, "v": self.attn.v_proj}[self.which]

    def register(self):
        hd, gate = self.head_dim, self.include_gate
        which, heads = self.which, self.heads

        def fn(mod, args, output):
            t = output.clone()
            B, T, D = t.shape
            per = hd * 2 if which == "q" else hd
            nh = D // per
            hs = range(nh) if heads is None else heads
            for h in hs:
                base = h * per
                sl = slice(base, base + (per if (which != "q" or gate) else hd))
                for p in self.positions:
                    q = p - self.offset
                    if 0 <= q < T:
                        t[:, q, sl] = self.donor[:, p, sl].to(t.dtype)
            return t
        return self._proj().register_forward_hook(fn)


class ProjCapture:
    def __init__(self, attn, which):
        self.attn, self.which, self.value = attn, which, None

    def register(self):
        def fn(mod, args, output):
            self.value = output.detach().clone()
        return {"q": self.attn.q_proj, "k": self.attn.k_proj,
                "v": self.attn.v_proj}[self.which].register_forward_hook(fn)


class AttnWeights:
    """Record softmax attention probabilities (requires eager attention)."""

    def __init__(self, attn):
        self.attn, self.value = attn, None

    def register(self):
        def fn(mod, args, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                self.value = output[1].detach().clone()
        return self.attn.register_forward_hook(fn)
