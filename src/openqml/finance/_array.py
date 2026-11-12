"""Native real arrays, domain checks and bounded financial workspaces."""

import math

import numpy as np

from ..compute.kernel import RealOps, _array_backend, floating_array, get_compute_kernel
from ..errors import ValidationError


class Ops(RealOps):
    def __init__(self, *values):
        values = tuple(v for v in values if v is not None)
        if not values:
            raise ValidationError("Provide a financial array.")
        native = [v for v in values if _array_backend(v) != "numpy"]
        frameworks = {_array_backend(v) for v in native}
        if len(frameworks) > 1:
            raise ValidationError("Financial tensors must use one framework; convert explicitly.")
        try:
            like = native[0] if native else np.asarray(values[0])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("Financial arrays must contain real numeric values.") from exc
        if "torch" in frameworks:
            import torch

            if len({v.device for v in native}) > 1:
                raise ValidationError("Financial tensors must use one device; move explicitly.")
            if any(v.dtype == torch.bool or v.dtype.is_complex for v in native):
                raise ValidationError("Financial arrays must contain real numeric values.")
            dtype = like.dtype
            for value in native[1:]:
                dtype = torch.promote_types(dtype, value.dtype)
            like = like.to(dtype=dtype if dtype.is_floating_point else torch.get_default_dtype())
        elif "jax" in frameworks:
            if any(v.dtype.kind not in "fiu" and str(v.dtype) != "bfloat16" for v in native):
                raise ValidationError("Financial arrays must contain real numeric values.")
            import jax.numpy as jnp

            dtype = jnp.result_type(*(v.dtype for v in native))
            if dtype.kind in "iu":
                dtype = jnp.dtype("float32")
            if like.dtype != dtype:
                like = like.astype(dtype)
        else:
            if like.dtype.kind not in "fiu":
                raise ValidationError("Financial arrays must contain real numeric values.")
            if like.dtype.kind in "iu":
                like = like.astype(float)
        super().__init__(floating_array(like))

    def sum(self, value, axis=-1, keepdims=False):
        return (
            value.sum(dim=axis, keepdim=keepdims)
            if self.name == "torch"
            else value.sum(axis=axis, keepdims=keepdims)
        )

    def max(self, value, axis=-1, keepdims=False):
        return (
            value.amax(dim=axis, keepdim=keepdims)
            if self.name == "torch"
            else self.xp.max(value, axis=axis, keepdims=keepdims)
        )

    def min(self, value, axis=-1, keepdims=False):
        return (
            value.amin(dim=axis, keepdim=keepdims)
            if self.name == "torch"
            else self.xp.min(value, axis=axis, keepdims=keepdims)
        )

    def stack(self, values, axis=-1):
        return (
            self.xp.stack(tuple(values), dim=axis)
            if self.name == "torch"
            else self.xp.stack(tuple(values), axis=axis)
        )

    def cumsum(self, value, axis=-1):
        return (
            self.xp.cumsum(value, dim=axis)
            if self.name == "torch"
            else self.xp.cumsum(value, axis=axis)
        )

    def gather(self, value, indices):
        return (
            self.xp.take_along_dim(value, indices, dim=-1)
            if self.name == "torch"
            else self.xp.take_along_axis(value, indices, axis=-1)
        )

    def argsort(self, value):
        if self.name == "torch":
            return self.xp.argsort(value, dim=-1, stable=True)
        return self.xp.argsort(
            value, axis=-1, **({"kind": "stable"} if self.name == "numpy" else {"stable": True})
        )

    def clip(self, value, low, high):
        return self.xp.minimum(self.xp.maximum(value, self.array(low)), self.array(high))

    def broadcast(self, value, shape):
        return self.xp.broadcast_to(value, shape)

    def real(self, value, *, finite=True):
        if isinstance(value, (bool, complex, str, bytes)):
            raise ValidationError("Financial arrays must contain real numeric values.")
        dtype = getattr(value, "dtype", None)
        if dtype is None and not isinstance(value, (int, float)):
            try:
                dtype = np.asarray(value).dtype
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValidationError("Financial arrays must contain real numeric values.") from exc
        if getattr(dtype, "kind", None) in {"b", "c", "O", "U", "S", "V"} or str(dtype).startswith(
            ("torch.bool", "torch.complex")
        ):
            raise ValidationError("Financial arrays must contain real numeric values.")
        result = self.array(value)
        return (
            self.check(result, self.xp.isfinite(result), "Financial arrays must be finite.")
            if finite
            else result
        )

    def move(self, value, source, target):
        return (
            self.xp.movedim(value, source, target)
            if self.name == "torch"
            else self.xp.moveaxis(value, source, target)
        )

    def probabilities(self, value, size):
        if size < 1:
            raise ValidationError("Probability distributions need a nonempty scenario axis.")
        p = self.real(np.ones(size) if value is None else value)
        if p.shape[-1:] != (size,):
            raise ValidationError("Probability width differs from the scenario count.")
        p = self.check(p, p >= 0, "Probabilities must be nonnegative.")
        # A power-of-two scale is exact in binary floating point, including
        # ordinary unit-mass inputs, and bounds the positive reduction by 2*S.
        _, exponent = self.xp.frexp(self.max(p, keepdims=True))
        scale = self.stop(self.xp.ldexp(self.xp.ones_like(p[..., :1]), exponent - 1))
        p = p / scale
        total = self.sum(p, keepdims=True)
        total = self.check(total, total > 0, "Probability mass must be positive.")
        return p / total


def admission(elements, max_memory_mb=256, *, work=0, max_work=1_000_000_000):
    get_compute_kernel().admit(elements, max_memory_mb, work=work, max_work=max_work)


def rows(shape):
    return math.prod(shape)
