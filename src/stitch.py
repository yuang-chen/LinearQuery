"""Cross-family stitching machinery shared by exp27-29: residual captures, block replacement,
closed-form ridge maps, and running a donor block on mapped hidden states."""
import numpy as np
import torch
from .runner import hook_ctx


class InCap:
    """residual stream entering layer L (all positions)"""
    def __init__(self, M, L): self.M, self.L, self.v = M, L, None
    def register(self):
        def fn(mod, args, kwargs):
            self.v = (kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]).detach().float()
        return self.M.layers[self.L].register_forward_pre_hook(fn, with_kwargs=True)


class OutCap:
    def __init__(self, M, L): self.M, self.L, self.v = M, L, None
    def register(self):
        def fn(mod, args, out):
            self.v = (out[0] if isinstance(out, tuple) else out).detach().float()
        return self.M.layers[self.L].register_forward_hook(fn)


class Replace:
    """Skip layers lo..hi-1 and make layer hi output h_in + g(h_in)."""
    def __init__(self, M, lo, hi, g): self.M, self.lo, self.hi, self.g = M, lo, hi, g
    def register(self):
        hs = []
        for L in range(self.lo, self.hi + 1):
            def fn(mod, args, kwargs, out, L=L):
                h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                if L < self.hi:
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                new = h + self.g(h.float()).to(h.dtype)
                return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
            hs.append(self.M.layers[L].register_forward_hook(fn, with_kwargs=True))

        class H:
            def remove(self_):
                for x in hs:
                    x.remove()
        return H()


class ZeroMix:
    def __init__(self, M, L): self.M, self.L = M, L
    def register(self):
        return self.M.mixer(self.L).register_forward_hook(
            lambda m, a_, o: (torch.zeros_like(o[0]),) + tuple(o[1:]) if isinstance(o, tuple)
            else torch.zeros_like(o))


@torch.no_grad()
def run_block(M, block, h, nomix=False):
    """Decoder layers `block` of model M applied to hidden states h [1, T, H]."""
    hooks = [ZeroMix(M, L) for L in block] if nomix else []
    with hook_ctx(hooks):
        x = h.to(next(M.model.parameters()).dtype)
        for L in block:
            o = M.layers[L](x, position_embeddings=None)
            x = o[0] if isinstance(o, tuple) else o
    return x.float()


class Ridge:
    """Centred ridge regression, lambda chosen on a random held-out fifth (<= 4000 rows)."""
    def __init__(self, X, T, n_val=None):
        n = len(X); perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        n_val = n_val or min(4000, n // 5)
        vi, ti = perm[:n_val], perm[n_val:]
        X = X.double().cuda(); T = T.double().cuda()
        self.mx, self.mt = X[ti].mean(0), T[ti].mean(0)
        Xc, Tc = X[ti] - self.mx, T[ti] - self.mt
        G = Xc.T @ Xc; C = Xc.T @ Tc
        ev, V = torch.linalg.eigh(G)
        VC = V.T @ C
        Xv, Tv = X[vi] - self.mx, T[vi] - self.mt
        best = None
        for p_ in np.arange(-6, 2.5, 0.5):
            lam = 10 ** p_ * float(ev[-1])
            W = V @ (VC / (ev + lam)[:, None])
            err = float(((Xv @ W - Tv) ** 2).sum() / (Tv ** 2).sum())
            if best is None or err < best[0]:
                best = (err, W)
        self.W, self.r2 = best[1], 1 - best[0]

    def __call__(self, X):
        return ((X.double() - self.mx) @ self.W + self.mt).float()


class Stitch:
    """g(h) = M_out(B_donor(M_in(h)) - M_in(h)) for a receiver residual h [B, T, H_recv]."""
    def __init__(self, D, block, nomix, m_in, m_out):
        self.D, self.block, self.nomix, self.m_in, self.m_out = D, block, nomix, m_in, m_out

    def __call__(self, h):
        out = []
        for b in range(h.shape[0]):
            x = self.m_in(h[b])[None]
            out.append(self.m_out(run_block(self.D, self.block, x, self.nomix)[0] - x[0]))
        return torch.stack(out)


def fit_stitch(D, block, nomix, items, yin_key, din_key, dw_target):
    """Fit M_in on aligned tokens (items[yin_key] -> items[din_key]) and M_out on every position
    (donor block write on the mapped input -> dw_target rows, same order as items)."""
    Xa = torch.cat([it[yin_key][[p for p, _ in it["al"]]] for it in items])
    Ta = torch.cat([it[din_key][[q for _, q in it["al"]]] for it in items])
    m_in = Ridge(Xa, Ta)
    dw = []
    for it in items:
        h = m_in(it[yin_key].cuda())[None]
        dw.append((run_block(D, block, h, nomix)[0] - h[0]).cpu())
    m_out = Ridge(torch.cat(dw), dw_target)
    return Stitch(D, block, nomix, m_in, m_out)
