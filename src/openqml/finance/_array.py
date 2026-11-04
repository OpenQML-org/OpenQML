"""Native real arrays, domain checks and bounded financial workspaces."""

import math

import numpy as np

from ..errors import ResourceError, ValidationError
from ..parameters import real
from ..rl import _floating, _Ops


class Ops(_Ops):
    def __init__(self, like):
        super().__init__(_floating(like))

    def array(self, value, *, boolean=False):
        if self.name == "torch" and isinstance(value, np.ndarray) and not value.flags.writeable:
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
        if not traced and not bool(self.xp.all(valid)):
            raise (ResourceError if resource else ValidationError)(message)
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
        squared = self.sum(value * value, axis)
        return self.xp.where(squared > 0, self.xp.sqrt(self.xp.where(squared > 0, squared, 1)), 0)

    def probabilities(self, value, size):
        p = self.real(np.ones(size) if value is None else value)
        if p.shape[-1:] != (size,):
            raise ValidationError("Probability width differs from the scenario count.")
        p = self.check(p, p >= 0, "Probabilities must be nonnegative.")
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
