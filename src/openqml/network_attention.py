"""Quantum Q/K/V features with Gaussian SDPA or positive-feature linear attention."""

import math

import numpy as np

from .errors import ValidationError


def quantum_attention(ctx):
    x, c, xp = ctx.inputs, ctx.spec.config, ctx.xp
    if x.ndim < 3:
        raise ValidationError("Attention requires a batch and at least one token dimension.")
    batch, width = x.shape[0], x.shape[-1]
    tokens, heads, key_size = math.prod(x.shape[1:-1]), c["heads"], c["key_features"]
    flat = x.reshape((batch, tokens, width))
    source = flat
    if c["positional"]:
        pos = np.arange(tokens)[:, None] / 10000 ** (2 * (np.arange(width) // 2)[None, :] / width)
        pos[:, 0::2], pos[:, 1::2] = np.sin(pos[:, 0::2]), np.cos(pos[:, 1::2])
        source = source + ctx.array(pos)[None]
    q = ctx.project("attention_q", ctx.kernel(source, "q"))
    k = ctx.project("attention_k", ctx.kernel(source, "k"))
    v = ctx.project("decode", ctx.kernel(source))

    def split(value, size):
        value = value.reshape((batch, tokens, heads, size))
        return (
            value.permute(0, 2, 1, 3)
            if ctx.backend == "torch"
            else xp.transpose(value, (0, 2, 1, 3))
        )

    q, k, v = split(q, key_size), split(k, key_size), split(v, width // heads)
    mask = ctx.context.get("node_mask", ctx.context.get("mask"))
    mask = ctx.array(
        np.ones((batch, tokens), dtype=bool) if mask is None else mask.reshape((batch, tokens)),
        boolean=True,
    )
    result = _linear(ctx, q, k, v, mask) if c["mode"] == "linear" else _gaussian(ctx, q, k, v, mask)
    result = (
        result.permute(0, 2, 1, 3) if ctx.backend == "torch" else xp.transpose(result, (0, 2, 1, 3))
    )
    return result.reshape(x.shape)


def _linear(ctx, q, k, v, mask):
    xp, c = ctx.xp, ctx.spec.config
    # Keep reductions in float32 under mixed precision.
    if ctx.backend == "torch" and q.dtype in (xp.float16, xp.bfloat16):
        q, k, v = q.float(), k.float(), v.float()

    def phi(value):
        return xp.logaddexp(ctx.array(0), value) + 1e-6

    q, k = phi(q), phi(k) * mask[:, None, :, None]
    v = v * mask[:, None, :, None]
    if not c["causal"]:
        ctx.check_workspace(q.shape[0] * q.shape[1] * q.shape[-1] * v.shape[-1])
        state = xp.einsum("bhnd,bhnv->bhdv", k, v)
        normalizer = ctx.sum(k, 2)
        numerator = xp.einsum("bhnd,bhdv->bhnv", q, state)
        denominator = xp.einsum("bhnd,bhd->bhn", q, normalizer)[..., None]
        return numerator / (denominator + 1e-6) * mask[:, None, :, None]
    outputs, state, normalizer = [], 0, 0
    for start in range(0, q.shape[2], c["chunk_size"]):
        stop = min(q.shape[2], start + c["chunk_size"])
        qc, kc, vc = q[:, :, start:stop], k[:, :, start:stop], v[:, :, start:stop]
        ctx.check_workspace(math.prod(kc.shape) * vc.shape[-1])
        outer = xp.einsum("bhnd,bhnv->bhndv", kc, vc)
        cumulative = (
            outer.cumsum(dim=2) if ctx.backend == "torch" else xp.cumsum(outer, axis=2)
        ) + (state if isinstance(state, int) else state[:, :, None])
        sums = (kc.cumsum(dim=2) if ctx.backend == "torch" else xp.cumsum(kc, axis=2)) + (
            normalizer if isinstance(normalizer, int) else normalizer[:, :, None]
        )
        numerator = xp.einsum("bhnd,bhndv->bhnv", qc, cumulative)
        denominator = xp.einsum("bhnd,bhnd->bhn", qc, sums)[..., None]
        outputs.append(numerator / (denominator + 1e-6))
        state, normalizer = cumulative[:, :, -1], sums[:, :, -1]
    return ctx.cat(outputs, axis=2) * mask[:, None, :, None]


def _gaussian(ctx, q, k, v, mask):
    xp, c, outputs = ctx.xp, ctx.spec.config, []
    tokens, temperature = q.shape[2], math.sqrt(q.shape[-1])
    key_norm = ctx.sum(k * k, -1, True)
    for start in range(0, tokens, c["chunk_size"]):
        stop = min(tokens, start + c["chunk_size"])
        qc = q[:, :, start:stop]
        ctx.check_workspace(q.shape[0] * q.shape[1] * (stop - start) * tokens * 4)
        allowed = mask[:, None, None, :] & mask[:, None, start:stop, None]
        if c["causal"]:
            allowed = (
                allowed
                & ctx.array(
                    np.arange(tokens)[None, :] <= np.arange(start, stop)[:, None], boolean=True
                )[None, None]
            )
        live = (
            allowed.any(dim=-1, keepdim=True)
            if ctx.backend == "torch"
            else xp.any(allowed, axis=-1, keepdims=True)
        )
        if ctx.backend == "torch":
            import torch.nn.functional as functional

            # Gaussian softmax equals dot-product softmax after augmenting Q/K.
            # Query norms cancel in row normalization. SDPA selects available
            # fused CUDA kernels; the sentinel makes old all-masked kernels safe.
            qa = ctx.cat((qc * math.sqrt(2 / temperature), xp.ones_like(qc[..., :1])))
            ka = ctx.cat((k * math.sqrt(2 / temperature), -key_norm / temperature))
            # Align Q/K/V widths for fused kernels without changing the kernel.
            value_width = v.shape[-1]
            aligned = 8 * math.ceil(max(qa.shape[-1], value_width) / 8)
            qa = functional.pad(qa, (0, aligned - qa.shape[-1]))
            ka = functional.pad(ka, (0, aligned - ka.shape[-1]))
            va = functional.pad(v, (0, aligned - value_width))
            sentinel = ctx.array(np.arange(tokens) == 0, boolean=True)[None, None, None]
            safe = allowed | (~live & sentinel)
            outputs.append(
                functional.scaled_dot_product_attention(
                    qa, ka, va, attn_mask=safe, dropout_p=0, scale=1
                )[..., :value_width]
                * live
            )
        else:
            scores = (
                2 * xp.einsum("bhid,bhjd->bhij", qc, k) - xp.swapaxes(key_norm, -1, -2)
            ) / temperature
            masked = xp.where(allowed, scores, -xp.inf)
            maximum = xp.where(live, xp.max(masked, axis=-1, keepdims=True), 0)
            weights = xp.exp(xp.where(allowed, scores - maximum, -xp.inf))
            weights = weights / xp.maximum(xp.sum(weights, axis=-1, keepdims=True), 1e-30)
            outputs.append(xp.einsum("bhij,bhjv->bhiv", weights, v))
    return ctx.cat(outputs, axis=2)
