"""Native real arrays, domain checks and bounded financial workspaces."""

import math

import numpy as np

from .._norms import vector_norm
from ..errors import ResourceError, ValidationError
from ..parameters import real
from ..rl import _floating, _Ops


class Ops(_Ops):
    def __init__(self, *values):
        values = tuple(v for v in values if v is not None)
        if not values:
            raise ValidationError("Provide a financial array.")
        native = [v for v in values if type(v).__module__.startswith(("torch", "jax", "jaxlib"))]
        frameworks = {"torch" if type(v).__module__.startswith("torch") else "jax" for v in native}
        if len(frameworks) > 1:
            raise ValidationError("Financial tensors must use one framework; convert explicitly.")
        like = native[0] if native else np.asarray(values[0])
        if "torch" in frameworks:
            import torch

            if len({v.device for v in native}) > 1:
                raise ValidationError("Financial tensors must use one device; move explicitly.")
            if any(v.dtype == torch.bool or v.is_complex() for v in native):
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
        super().__init__(_floating(like))

    def array(self, value, *, boolean=False):
        if self.name == "jax":
            import jax

            if not isinstance(self.like, jax.core.Tracer) and isinstance(
                getattr(self.like, "sharding", None), jax.sharding.SingleDeviceSharding
            ):
                return self.xp.asarray(
                    value, dtype=bool if boolean else self.like.dtype, device=self.like.sharding
                )
        if (
            self.name == "torch"
            and isinstance(value, np.ndarray)
            and (not value.flags.writeable or any(s < 0 for s in value.strides))
        ):
            value = value.copy()
        return super().array(value, boolean=boolean)

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

    def check(self, value, valid, message, *, resource=False):
        traced = False
        if self.name == "torch":
            traced = getattr(self.xp.compiler, "is_compiling", lambda: False)()
        elif self.name == "jax":
            import jax

            traced = isinstance(valid, jax.core.Tracer)
        if not traced:
            if not bool(self.xp.all(valid)):
                raise (ResourceError if resource else ValidationError)(message)
            return value
        return self.xp.where(valid, value, float("nan"))

    def real(self, value, *, finite=True):
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            dtype = np.asarray(value).dtype
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

    def norm(self, value, axis=-1):
        return vector_norm(value, self.xp, self.name, axis=axis)

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
    budget = real(max_memory_mb)
    if budget <= 0:
        raise ValidationError("max_memory_mb must be positive.")
    if 16 * elements > budget * 2**20 or work > max_work:
        raise ResourceError("Financial primal workspace or arithmetic estimate exceeds the budget.")


def rows(shape):
    return math.prod(shape)
