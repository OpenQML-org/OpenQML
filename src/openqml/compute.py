"""Shared native compute primitives, numerical contracts and bounded plan reuse.

The kernel owns policies and immutable metadata, never model tensors or device
contexts. PyTorch and JAX continue to own their compiled graphs and gradients.
"""

from __future__ import annotations

import math
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from numbers import Integral, Real
from threading import RLock

import numpy as np

from .errors import NumericalError, ResourceError, ValidationError


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a finite real number.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be a finite real number.") from exc
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValidationError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}.")
    return value


@dataclass(frozen=True)
class ComputePolicy:
    """Process-wide ceilings; operation-specific budgets can always be tighter.

    max_memory_mb limits modeled primal workspaces, not process RSS, allocator
    pools, autodiff tapes or vendor workspaces. None preserves caller budgets.
    Numerical tolerances apply to provider validation, not approximation of
    circuits or stopping criteria selected by individual solvers.
    """

    max_memory_mb: float | None = None
    cache_entries: int = 256
    cache_bytes: int = 8 * 2**20
    validation_atol: float = 1e-6
    validation_rtol: float = 1e-7

    def __post_init__(self):
        if self.max_memory_mb is not None:
            object.__setattr__(
                self, "max_memory_mb", _positive(self.max_memory_mb, "max_memory_mb")
            )
        for name in ("cache_entries", "cache_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValidationError(f"{name} must be a nonnegative integer.")
            object.__setattr__(self, name, int(value))
        for name in ("validation_atol", "validation_rtol"):
            object.__setattr__(self, name, _positive(getattr(self, name), name, zero=True))


def _metadata_size(value):
    """Count retained Python metadata and reject tensor/autograd ownership."""
    size = sys.getsizeof(value)
    if value is None or type(value) in {bool, int, float, complex, str, bytes}:
        return size
    if isinstance(value, (tuple, frozenset)):
        return size + sum(_metadata_size(v) for v in value)
    if is_dataclass(value) and value.__dataclass_params__.frozen:
        storage = sys.getsizeof(value.__dict__) if hasattr(value, "__dict__") else 0
        return size + storage + _metadata_size(tuple(getattr(value, f.name) for f in fields(value)))
    raise ValidationError("Shared compute caches accept immutable metadata, not arrays or tensors.")


def _broadcast_shape(shapes):
    result = ()
    for shape in shapes:
        width = max(len(result), len(shape))
        left, right = (1,) * (width - len(result)) + result, (1,) * (width - len(shape)) + shape
        if any(a != b and a != 1 and b != 1 for a, b in zip(left, right, strict=True)):
            raise ValidationError("Compute batch dimensions cannot be broadcast.")
        result = tuple(b if a == 1 else a for a, b in zip(left, right, strict=True))
    return result


def _einsum_shape(equation, shapes):
    source, arrow, output = equation.replace(" ", "").partition("->")
    terms = source.split(",")
    if len(terms) != len(shapes):
        raise ValidationError("Contraction operands do not match the equation.")
    ellipses = []
    sizes, counts = {}, Counter(source.replace("...", "").replace(",", ""))
    for term, shape in zip(terms, shapes, strict=True):
        before, ellipsis, after = term.partition("...")
        labels = before + after
        width = len(shape) - len(labels)
        if width < 0 or (not ellipsis and width):
            raise ValidationError("Contraction dimensions do not match the equation.")
        ellipses.append(shape[len(before) : len(before) + width] if ellipsis else ())
        selected = shape[: len(before)] + (shape[-len(after) :] if after else ())
        local = {}
        for label, extent in zip(labels, selected, strict=True):
            if label in local and local[label] != extent:
                raise ValidationError("Repeated contraction axes must have equal extents.")
            if label in sizes and sizes[label] != extent and sizes[label] != 1 and extent != 1:
                raise ValidationError("Shared contraction axes cannot be broadcast.")
            local[label] = extent
            sizes[label] = extent if sizes.get(label, 1) == 1 else sizes[label]
    batch = _broadcast_shape(ellipses)
    if not arrow:
        return batch + tuple(sizes[label] for label in sorted(sizes) if counts[label] == 1)
    before, ellipsis, after = output.partition("...")
    if any(label not in sizes for label in before + after):
        raise ValidationError("Contraction output references an unknown axis.")
    return (
        tuple(sizes[label] for label in before)
        + (batch if ellipsis else ())
        + tuple(sizes[label] for label in after)
    )


class _PlanCache:
    def __init__(self, policy):
        self.policy, self.lock = policy, RLock()
        self.values = OrderedDict()
        self.bytes = self.hits = self.misses = self.evictions = 0

    def get(self, key, build):
        # Compilation under one reentrant lock prevents duplicate same-key
        # planning and publishes only complete immutable plans.
        with self.lock:
            if key in self.values:
                self.hits += 1
                self.values.move_to_end(key)
                return self.values[key][0]
            self.misses += 1
            value = build()
            size = 256 + _metadata_size(key) + _metadata_size(value)
            if not self.policy.cache_entries or size > self.policy.cache_bytes:
                return value
            while self.values and (
                len(self.values) >= self.policy.cache_entries
                or self.bytes + size > self.policy.cache_bytes
            ):
                _, (_, removed) = self.values.popitem(last=False)
                self.bytes -= removed
                self.evictions += 1
            self.values[key] = value, size
            self.bytes += size
            return value

    def info(self):
        with self.lock:
            return {
                "entries": len(self.values),
                "bytes": self.bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "max_entries": self.policy.cache_entries,
                "max_bytes": self.policy.cache_bytes,
            }

    def clear(self):
        with self.lock:
            self.values.clear()
            self.bytes = self.hits = self.misses = self.evictions = 0


class ComputeKernel:
    """One compute service for simulation, tensor networks, AI and finance."""

    def __init__(self, policy=None):
        self.policy = ComputePolicy() if policy is None else policy
        if not isinstance(self.policy, ComputePolicy):
            raise ValidationError("policy must be ComputePolicy.")
        self._plans = _PlanCache(self.policy)

    def memory_budget(self, requested):
        requested = _positive(requested, "max_memory_mb")
        cap = self.policy.max_memory_mb
        return requested if cap is None else min(requested, cap)

    def admit(self, elements, max_memory_mb=256, *, itemsize=16, work=0, max_work=None):
        for name, value in (("elements", elements), ("itemsize", itemsize), ("work", work)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValidationError(f"{name} must be a nonnegative integer.")
        if itemsize == 0:
            raise ValidationError("itemsize must be positive.")
        if max_work is not None and (
            isinstance(max_work, bool) or not isinstance(max_work, Integral) or max_work < 0
        ):
            raise ValidationError("max_work must be a nonnegative integer.")
        budget = self.memory_budget(max_memory_mb)
        numerator, denominator = budget.as_integer_ratio()
        if elements * itemsize > (numerator << 20) // denominator or (
            max_work is not None and work > max_work
        ):
            raise ResourceError("Compute workspace or arithmetic estimate exceeds the budget.")
        return budget

    def cache_info(self):
        return self._plans.info()

    def clear_cache(self):
        self._plans.clear()

    def plan(self, namespace, key, build):
        """Intern immutable structural metadata; bound values stay in each forward."""
        return self._plans.get((namespace, key), build)

    def _workspace(self, values, output_shape, *, buffers=1):
        cap = self.policy.max_memory_mb
        if cap is not None:
            sizes = tuple(
                v.element_size() if hasattr(v, "element_size") else v.dtype.itemsize for v in values
            )
            itemsize = max(sizes)
            # Mixed real/complex promotion can double the largest input item.
            dtypes = tuple(str(v.dtype) for v in values)
            if len(set(dtypes)) > 1 and any("complex" in dtype for dtype in dtypes):
                itemsize *= 2
            elements = sum(math.prod(v.shape) for v in values) + buffers * math.prod(output_shape)
            self.admit(elements, cap, itemsize=itemsize)

    def asarray(self, value, *, xp=np, backend="numpy", dtype=None, like=None, device=None):
        """Convert once, preserving native device placement and differentiation."""
        if backend == "torch":
            if isinstance(value, np.ndarray):
                compiling = getattr(xp.compiler, "is_compiling", None) or xp._dynamo.is_compiling
                if not compiling() and (
                    not value.flags.writeable or any(s < 0 for s in value.strides)
                ):
                    value = value.copy()
            if device is None and like is not None:
                device = like.device
            return xp.as_tensor(value, dtype=dtype, device=device)
        if backend == "jax" and like is not None:
            import jax

            traced = isinstance(value, jax.core.Tracer) or (
                isinstance(value, (list, tuple))
                and any(isinstance(v, jax.core.Tracer) for v in jax.tree_util.tree_leaves(value))
            )
            if (
                not traced
                and not isinstance(like, jax.core.Tracer)
                and isinstance(getattr(like, "sharding", None), jax.sharding.SingleDeviceSharding)
            ):
                return xp.asarray(value, dtype=dtype, device=like.sharding)
        return xp.asarray(value, dtype=dtype)

    def matmul(self, left, right, *, xp=np, out=None):
        if xp is np:
            left, right = np.asarray(left), np.asarray(right)
        if self.policy.max_memory_mb is not None and left.ndim and right.ndim:
            batch = _broadcast_shape((tuple(left.shape[:-2]), tuple(right.shape[:-2])))
            shape = (
                batch
                + ((left.shape[-2],) if left.ndim > 1 else ())
                + ((right.shape[-1],) if right.ndim > 1 else ())
            )
            self._workspace((left, right), shape)
        return xp.matmul(left, right) if out is None else xp.matmul(left, right, out=out)

    def einsum(self, equation, *operands, xp=np, backend="numpy"):
        if backend == "numpy":
            operands = tuple(np.asarray(v) for v in operands)
        if self.policy.max_memory_mb is not None:
            shape = _einsum_shape(equation, tuple(tuple(v.shape) for v in operands))
            self._workspace(operands, shape)
        if backend != "numpy" or len(operands) < 2:
            return xp.einsum(equation, *operands)
        if len(operands) == 2 and all(v.size < 2048 for v in operands):
            # Tiny contractions are faster than Python planning/BLAS dispatch.
            return np.einsum(equation, *operands)
        # Search using scalar-backed shape views: no real operand, device value
        # or differentiation graph is retained by the process-wide cache.
        shapes = tuple(tuple(int(n) for n in v.shape) for v in operands)
        itemsize = max(np.dtype(v.dtype).itemsize for v in operands)
        cap = self.policy.max_memory_mb
        limit = None if cap is None else int(cap * 2**20) // itemsize

        def build():
            views = tuple(np.broadcast_to(np.zeros((), dtype=np.uint8), s) for s in shapes)
            optimize = "greedy" if limit is None else ("greedy", limit)
            return tuple(np.einsum_path(equation, *views, optimize=optimize)[0])

        path = self.plan("einsum", (equation, shapes, limit), build)
        return np.einsum(equation, *operands, optimize=list(path))

    def check(self, value, valid, message, *, xp=np, backend="numpy", error=ValidationError):
        """Raise eagerly; compiled invalid lanes remain visibly non-finite."""
        traced = False
        if backend == "torch":
            traced = getattr(xp.compiler, "is_compiling", lambda: False)()
        elif backend == "jax":
            import jax

            traced = isinstance(valid, jax.core.Tracer)
        if not traced:
            if not bool(xp.all(valid)):
                raise error(message)
            return value
        return xp.where(valid, value, float("nan"))

    def finite(self, value, *, xp=np, backend="numpy", error=NumericalError):
        return self.check(
            value,
            xp.isfinite(value),
            "Compute input or result is not finite.",
            xp=xp,
            backend=backend,
            error=error,
        )

    def solve(self, matrix, rhs, *, xp=np, backend="numpy"):
        if backend == "numpy":
            matrix, rhs = np.asarray(matrix), np.asarray(rhs)
        self._workspace((matrix, rhs), tuple(rhs.shape), buffers=2)
        matrix = self.finite(matrix, xp=xp, backend=backend)
        rhs = self.finite(rhs, xp=xp, backend=backend)
        try:
            result = xp.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as exc:
            raise NumericalError("Linear system is singular or its factorization failed.") from exc
        except RuntimeError as exc:
            if backend != "torch" or type(exc).__name__ not in {"_LinAlgError", "LinAlgError"}:
                raise
            raise NumericalError("Linear system is singular or its factorization failed.") from exc
        return self.finite(result, xp=xp, backend=backend)

    def svd(self, matrix, *, xp=np, backend="numpy", full_matrices=False):
        if backend == "numpy":
            matrix = np.asarray(matrix)
        rows, columns = matrix.shape[-2:]
        rank = min(rows, columns)
        outputs = (
            rows * rows + columns * columns + rank if full_matrices else (rows + columns + 1) * rank
        )
        self._workspace((matrix,), matrix.shape[:-2] + (outputs,))
        matrix = self.finite(matrix, xp=xp, backend=backend)
        try:
            values = xp.linalg.svd(matrix, full_matrices=full_matrices)
        except np.linalg.LinAlgError as exc:
            raise NumericalError("Singular-value decomposition did not converge.") from exc
        except RuntimeError as exc:
            if backend != "torch" or type(exc).__name__ not in {"_LinAlgError", "LinAlgError"}:
                raise
            raise NumericalError("Singular-value decomposition did not converge.") from exc
        return tuple(self.finite(v, xp=xp, backend=backend) for v in values)

    def norm(self, value, *, xp=np, backend="numpy", axis=None, keepdims=False):
        from ._norms import vector_norm

        return vector_norm(value, xp, backend, axis=axis, keepdims=keepdims)


_KERNEL = ComputeKernel()


def get_compute_kernel():
    """Return the shared process-local kernel without discovering devices."""
    return _KERNEL


def configure_compute(*, policy=None):
    """Atomically install a new policy and empty shared metadata cache.

    Configure before starting worker threads or compiling native graphs. Existing
    Torch/JAX compiled graphs capture their numerical policy at trace time and
    must be retraced after reconfiguration. No hardware or jobs are accessed.
    """
    global _KERNEL
    kernel = ComputeKernel(policy)
    _KERNEL = kernel
    return kernel


def memory_budget(requested):
    """Apply the shared ceiling to a caller's operation-specific workspace."""
    return _KERNEL.memory_budget(requested)


class KernelOps:
    """Native-array bridge shared by quantum, reinforcement and financial code."""

    @property
    def kernel(self):
        return _KERNEL

    def matmul(self, left, right, *, out=None):
        return self.kernel.matmul(left, right, xp=self.xp, out=out)

    def einsum(self, equation, *operands):
        return self.kernel.einsum(equation, *operands, xp=self.xp, backend=self.backend)

    def solve(self, matrix, rhs):
        return self.kernel.solve(matrix, rhs, xp=self.xp, backend=self.backend)

    def check(self, value, valid, message, *, resource=False):
        return self.kernel.check(
            value,
            valid,
            message,
            xp=self.xp,
            backend=self.backend,
            error=ResourceError if resource else ValidationError,
        )

    def norm(self, value, axis=-1):
        return self.kernel.norm(value, xp=self.xp, backend=self.backend, axis=axis)
