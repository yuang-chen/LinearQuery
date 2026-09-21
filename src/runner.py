"""Model loading, split execution, and patching utilities for Qwen3.5-0.8B (GDN/softmax hybrid)."""
import contextlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from .task import MODEL_PATH


class Runner:
    def __init__(self, model_path=MODEL_PATH, device="cuda:0", dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        cfg = self.model.config.get_text_config()
        self.cfg = cfg
        self.layer_types = list(cfg.layer_types)
        self.gdn_layers = [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]
        self.attn_layers = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]
        self.n_layers = cfg.num_hidden_layers

    # ---------- module accessors ----------
    @property
    def layers(self):
        return self.model.model.layers if hasattr(self.model.model, "layers") else self.model.model.language_model.layers

    def mixer(self, i):
        L = self.layers[i]
        return L.linear_attn if self.layer_types[i] == "linear_attention" else L.self_attn

    def mlp(self, i):
        return self.layers[i].mlp

    # ---------- basic forward ----------
    def encode(self, text):
        return self.tok(text, return_tensors="pt").input_ids.to(self.device)

    @torch.no_grad()
    def forward(self, ids, hooks=(), use_cache=False):
        with hook_ctx(hooks):
            out = self.model(ids, use_cache=use_cache)
        return out

    # ---------- split execution ----------
    @torch.no_grad()
    def run_prefix(self, ids, upto, hooks=()):
        """Run tokens [0, upto] (inclusive) and return a fresh cache."""
        cache = DynamicCache(config=self.model.config.get_text_config())
        with hook_ctx(hooks):
            self.model(ids[:, : upto + 1], past_key_values=cache, use_cache=True)
        return cache

    @torch.no_grad()
    def run_suffix(self, ids, upto, cache, hooks=()):
        """Run tokens [upto+1, end] on top of `cache`; returns final-position logits."""
        n = ids.shape[1]
        pos = torch.arange(upto + 1, n, device=ids.device).view(1, -1)
        with hook_ctx(hooks):
            out = self.model(ids[:, upto + 1:], past_key_values=cache, use_cache=True, position_ids=pos)
        return out.logits[:, -1, :]

    @torch.no_grad()
    def split_forward(self, ids, upto, state_patch=None, prefix_hooks=(), suffix_hooks=()):
        """Run prefix, optionally patch GDN recurrent states, then run the suffix.

        state_patch: dict {layer_idx: tensor} of donor recurrent states.
        """
        cache = self.run_prefix(ids, upto, hooks=prefix_hooks)
        if state_patch:
            for li, st in state_patch.items():
                cache.layers[li].recurrent_states[0] = st.clone().to(
                    cache.layers[li].recurrent_states[0].dtype)
        return self.run_suffix(ids, upto, cache, hooks=suffix_hooks)

    @torch.no_grad()
    def recurrent_states(self, ids, upto, hooks=()):
        """GDN recurrent states after consuming tokens [0, upto]; {layer: tensor(clone)}."""
        cache = self.run_prefix(ids, upto, hooks=hooks)
        return {li: cache.layers[li].recurrent_states[0].detach().clone() for li in self.gdn_layers}


# ---------- hooks ----------
@contextlib.contextmanager
def hook_ctx(hooks):
    handles = []
    try:
        for h in hooks:
            handles.append(h.register())
        yield
    finally:
        for h in handles:
            h.remove()


class OutPatch:
    """Overwrite a module's output at selected sequence positions with donor values."""

    def __init__(self, module, positions, donor, out_index=None, offset=0):
        self.module, self.positions, self.donor = module, positions, donor
        self.out_index, self.offset = out_index, offset

    def register(self):
        def fn(mod, args, output):
            t = output[self.out_index] if self.out_index is not None else output
            t = t.clone()
            for p in self.positions:
                q = p - self.offset
                if 0 <= q < t.shape[1]:
                    t[:, q] = self.donor[:, p].to(t.dtype)
            if self.out_index is not None:
                output = list(output)
                output[self.out_index] = t
                return tuple(output)
            return t
        return self.module.register_forward_hook(fn)


class InPatch:
    """Overwrite a module's `hidden_states` input at selected positions with donor values."""

    def __init__(self, module, positions, donor, offset=0):
        self.module, self.positions, self.donor, self.offset = module, positions, donor, offset

    def register(self):
        def fn(mod, args, kwargs):
            if "hidden_states" in kwargs:
                t = kwargs["hidden_states"].clone()
            else:
                t = args[0].clone()
            for p in self.positions:
                q = p - self.offset
                if 0 <= q < t.shape[1]:
                    t[:, q] = self.donor[:, p].to(t.dtype)
            if "hidden_states" in kwargs:
                kwargs["hidden_states"] = t
                return args, kwargs
            return (t,) + tuple(args[1:]), kwargs
        return self.module.register_forward_pre_hook(fn, with_kwargs=True)


class Capture:
    """Record a module's output (or input)."""

    def __init__(self, module, out_index=None, mode="out"):
        self.module, self.out_index, self.mode = module, out_index, mode
        self.value = None

    def register(self):
        if self.mode == "out":
            def fn(mod, args, output):
                t = output[self.out_index] if self.out_index is not None else output
                self.value = t.detach().clone()
            return self.module.register_forward_hook(fn)

        def fn(mod, args, kwargs):
            t = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            self.value = t.detach().clone()
        return self.module.register_forward_pre_hook(fn, with_kwargs=True)


class HeadPatch:
    """Patch selected softmax-attention heads' contribution, before o_proj.

    Hooks the attention module's o_proj input (head-concat), replacing the slice of the
    given heads at the given positions with the donor's.
    """

    def __init__(self, attn_module, heads, positions, donor, head_dim, offset=0):
        self.m, self.heads, self.positions = attn_module, heads, positions
        self.donor, self.head_dim, self.offset = donor, head_dim, offset

    def register(self):
        def fn(mod, args, kwargs):
            t = (kwargs["input"] if "input" in kwargs else args[0]).clone()
            for h in self.heads:
                sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                for p in self.positions:
                    q = p - self.offset
                    if 0 <= q < t.shape[1]:
                        t[:, q, sl] = self.donor[:, p, sl].to(t.dtype)
            return (t,) + tuple(args[1:]), kwargs
        return self.m.o_proj.register_forward_pre_hook(fn, with_kwargs=True)
