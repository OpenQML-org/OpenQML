"""Shared native compute primitives, numerical contracts and bounded plan reuse.

The kernel owns policies and immutable metadata, never model tensors or device
contexts. PyTorch and JAX continue to own their compiled graphs and gradients.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache, wraps
from numbers import Integral, Real
from threading import RLock
from typing import NamedTuple

import numpy as np

from ..errors import NumericalError, ResourceError, ValidationError
from ._cache import SingleFlightCache


def _torch_compiling():
    torch = sys.modules.get("torch")
    return torch is not None and torch._dynamo.is_compiling()


def _array_backend(value):
    """Recognize real native arrays without importing a framework for user data."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, getattr(torch, "Tensor", ())):
        return "torch"
    if hasattr(value, "dtype") and type(value).__module__.startswith(("jax.", "jaxlib.")):
        jax = sys.modules.get("jax")
        if jax is not None and isinstance(value, (jax.Array, jax.core.Tracer)):
            return "jax"
    return "numpy"


@lru_cache(maxsize=1)
def _jax_asarray_device():
    """Observe the installed converter instead of assuming a newer JAX API."""
    import inspect

    import jax.numpy as jnp

    return "device" in inspect.signature(jnp.asarray).parameters


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValidationError(f"{name} must be a finite real number.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be a finite real number.") from exc
    if not -float("inf") < value < float("inf") or value < 0 or (value == 0 and not zero):
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
    pending, seen, size = [value], {}, 0
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        # Retain visited objects until counting ends, including values returned
        # by descriptors, so temporary object IDs cannot be recycled.
        seen[identity] = current
        size += sys.getsizeof(current)
        if current is None or type(current) in {bool, int, float, complex, str, bytes}:
            continue
        if isinstance(current, (tuple, frozenset)):
            pending.extend(current)
        elif is_dataclass(current) and current.__dataclass_params__.frozen:
            if hasattr(current, "__dict__"):
                size += sys.getsizeof(current.__dict__)
            pending.extend(getattr(current, f.name) for f in fields(current))
        else:
            raise ValidationError(
                "Shared compute caches accept immutable metadata, not arrays or tensors."
            )
    return size


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
    sizes = {}
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
        return batch + tuple(sizes[label] for label in sorted(sizes) if source.count(label) == 1)
    before, ellipsis, after = output.partition("...")
    if any(label not in sizes for label in before + after):
        raise ValidationError("Contraction output references an unknown axis.")
    return (
        tuple(sizes[label] for label in before)
        + (batch if ellipsis else ())
        + tuple(sizes[label] for label in after)
    )


class _PlanCache(SingleFlightCache):
    def __init__(self, policy):
        super().__init__(
            policy.cache_entries,
            policy.cache_bytes,
            lambda key, value: 256 + _metadata_size(key) + _metadata_size(value),
        )


class _ByteBudget(NamedTuple):
    memory_mb: float
    max_bytes: int


@dataclass(frozen=True)
class _BinaryGEMM:
    left_axes: tuple
    right_axes: tuple
    left_expand: tuple
    right_expand: tuple
    left_matrix: tuple
    right_matrix: tuple
    natural_shape: tuple
    output_axes: tuple
    output_shape: tuple

    @classmethod
    def compile(cls, equation, shapes):
        source, arrow, target = equation.replace(" ", "").partition("->")
        terms = source.split(",")
        if len(terms) != 2:
            return None
        widths = [len(s) - len(t.replace("...", "")) for t, s in zip(terms, shapes, strict=True)]
        if min(widths) < 0:
            return None
        ellipsis = tuple(f"#{i}" for i in range(max(widths)))
        labels = []
        for term, width in zip(terms, widths, strict=True):
            before, dot, after = term.partition("...")
            if width and not dot:
                return None
            row = tuple(before) + (ellipsis[-width:] if width else ()) + tuple(after)
            if len(set(row)) != len(row):
                return None
            labels.append(row)
        if arrow:
            before, dot, after = target.partition("...")
            output = tuple(before) + (ellipsis if dot else ()) + tuple(after)
        else:
            counts = Counter(x for row in labels for x in row)
            output = ellipsis + tuple(
                sorted(x for x, n in counts.items() if n == 1 and x[0] != "#")
            )
        left, right = labels
        if len(set(output)) != len(output) or set(output) - set(left + right):
            return None
        sizes = {}
        for row, shape in zip(labels, shapes, strict=True):
            for label, extent in zip(row, shape, strict=True):
                if label in sizes and sizes[label] != extent and sizes[label] != 1 and extent != 1:
                    return None
                sizes[label] = extent if sizes.get(label, 1) == 1 else sizes[label]
        batch = tuple(x for x in output if x in left and x in right)
        contracted = tuple(x for x in left if x in right and x not in output)
        a = tuple(x for x in left if x not in batch + contracted)
        b = tuple(x for x in right if x not in batch + contracted)
        if set(a + b) - set(output):
            return None
        natural = batch + a + b
        if set(natural) != set(output):
            return None
        k = math.prod(sizes[x] for x in contracted)
        expand = tuple(
            tuple(sizes[x] if x in contracted else n for x, n in zip(row, shape, strict=True))
            for row, shape in zip(labels, shapes, strict=True)
        )
        batches = tuple(
            tuple(shape[row.index(x)] for x in batch)
            for row, shape in zip(labels, shapes, strict=True)
        )
        return cls(
            tuple(left.index(x) for x in batch + a + contracted),
            tuple(right.index(x) for x in batch + contracted + b),
            *expand,
            batches[0] + (math.prod(sizes[x] for x in a), k),
            batches[1] + (k, math.prod(sizes[x] for x in b)),
            tuple(sizes[x] for x in natural),
            tuple(natural.index(x) for x in output),
            tuple(sizes[x] for x in output),
        )

    def apply(self, left, right):
        if left.shape != self.left_expand:
            left = np.broadcast_to(left, self.left_expand)
        if right.shape != self.right_expand:
            right = np.broadcast_to(right, self.right_expand)
        left = left.transpose(self.left_axes).reshape(self.left_matrix)
        right = right.transpose(self.right_axes).reshape(self.right_matrix)
        result = np.matmul(left, right).reshape(self.natural_shape)
        return result.transpose(self.output_axes)


class ComputeKernel:
    """One compute service for simulation, tensor networks, AI and finance."""

    def __init__(self, policy=None):
        self._source_id = id(self)
        self._policy = ComputePolicy() if policy is None else policy
        if not isinstance(self.policy, ComputePolicy):
            raise ValidationError("policy must be ComputePolicy.")
        if self.policy.max_memory_mb is None:
            self._memory_bytes = None
        else:
            numerator, denominator = self.policy.max_memory_mb.as_integer_ratio()
            self._memory_bytes = (numerator << 20) // denominator
        self._default_budget = _ByteBudget(
            min(256.0, self.policy.max_memory_mb)
            if self.policy.max_memory_mb is not None
            else 256.0,
            min(256 << 20, self._memory_bytes) if self._memory_bytes is not None else 256 << 20,
        )
        self._plans = _PlanCache(self.policy)

    @property
    def policy(self):
        return self._policy

    def memory_budget(self, requested):
        requested = _positive(requested, "max_memory_mb")
        cap = self.policy.max_memory_mb
        return requested if cap is None else min(requested, cap)

    def _prepare_budget(self, requested=256):
        if type(requested) in (int, float) and requested == 256:
            return self._default_budget
        budget = self.memory_budget(requested)
        numerator, denominator = budget.as_integer_ratio()
        return _ByteBudget(budget, (numerator << 20) // denominator)

    def admit(self, elements, max_memory_mb=256, *, itemsize=16, work=0, max_work=None):
        # Common eager/static dimensions need neither ABC dispatch nor repeated
        # float-to-byte conversion. Symbolic/native integers use the checked path.
        if (
            type(elements) is int
            and type(itemsize) is int
            and type(work) is int
            and elements >= 0
            and itemsize > 0
            and work >= 0
            and (max_work is None or (type(max_work) is int and max_work >= 0))
            and (
                (type(max_memory_mb) in (int, float) and max_memory_mb == 256)
                or not _torch_compiling()
            )
        ):
            budget = self._prepare_budget(max_memory_mb)
            if elements * itemsize > budget.max_bytes or (max_work is not None and work > max_work):
                raise ResourceError("Compute workspace or arithmetic estimate exceeds the budget.")
            return budget.memory_mb
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
        torch = sys.modules.get("torch")
        if _torch_compiling():
            valid = elements * itemsize <= budget * 2**20
            if max_work is not None:
                valid = valid & (work <= max_work)
            torch._assert(valid, "[OpenQML:ResourceError] Compute budget exceeded.")
            return budget
        numerator, denominator = budget.as_integer_ratio()
        if elements * itemsize > (numerator << 20) // denominator or (
            max_work is not None and work > max_work
        ):
            raise ResourceError("Compute workspace or arithmetic estimate exceeds the budget.")
        return budget

    def cache_info(self):
        return self._plans.info()

    def compile(self, function, *, backend, compiler=None, check_outputs=True, **options):
        from ..autodiff.compilation import compile_compute

        return compile_compute(
            function,
            backend=backend,
            compiler=compiler,
            kernel=self,
            check_outputs=check_outputs,
            **options,
        )

    def clear_cache(self):
        self._plans.clear()

    def plan(self, namespace, key, build):
        """Intern immutable structural metadata; bound values stay in each forward."""
        return self._plans.get((namespace, key), build)

    def _workspace(self, values, output_shape, *, buffers=1, scratch_shapes=()):
        cap = self.policy.max_memory_mb
        if cap is not None:
            sizes = tuple(v.dtype.itemsize for v in values)
            itemsize = max(sizes)
            # Mixed real/complex promotion can double the largest input item.
            dtypes = tuple(str(v.dtype) for v in values)
            if len(set(dtypes)) > 1 and any("complex" in dtype for dtype in dtypes):
                itemsize *= 2
            elements = (
                sum(math.prod(v.shape) for v in values)
                + sum(math.prod(shape) for shape in scratch_shapes)
                + buffers * math.prod(output_shape)
            )
            required = elements * itemsize
            torch = sys.modules.get("torch")
            if _torch_compiling():
                torch._assert(
                    required <= self._memory_bytes,
                    "[OpenQML:ResourceError] Compute workspace exceeds the budget.",
                )
            elif required > self._memory_bytes:
                raise ResourceError("Compute workspace exceeds the budget.")

    def asarray(self, value, *, xp=None, backend="numpy", dtype=None, like=None, device=None):
        """Convert once, preserving native device placement and differentiation."""
        xp = np if xp is None else xp
        if backend == "torch":
            if isinstance(value, np.ndarray):
                if not _torch_compiling() and (
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
                if _jax_asarray_device():
                    return xp.asarray(value, dtype=dtype, device=like.sharding)
                return jax.device_put(xp.asarray(value, dtype=dtype), like.sharding)
        return xp.asarray(value, dtype=dtype)

    def matmul(self, left, right, *, xp=None, out=None):
        xp = np if xp is None else xp
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

    def einsum(self, equation, *operands, xp=None, backend="numpy"):
        xp = np if xp is None else xp
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
        if cap is None:
            limit = None
        else:
            numerator, denominator = cap.as_integer_ratio()
            limit = (numerator << 20) // (denominator * itemsize)

        def build():
            if len(operands) == 2 and all(v.dtype.kind in "fciub" for v in operands):
                gemm = _BinaryGEMM.compile(equation, shapes)
                if gemm is not None:
                    return gemm
            views = tuple(np.broadcast_to(np.zeros((), dtype=np.uint8), s) for s in shapes)
            optimize = "greedy" if limit is None else ("greedy", limit)
            return tuple(np.einsum_path(equation, *views, optimize=optimize)[0])

        path = self.plan("einsum", (equation, shapes, limit), build)
        if isinstance(path, _BinaryGEMM):
            if self.policy.max_memory_mb is not None:
                # Broadcasting can expand contracted axes before reshape makes
                # copies. Admit those expanded shapes before entering GEMM.
                try:
                    self._workspace(
                        operands,
                        path.output_shape,
                        scratch_shapes=(path.left_expand, path.right_expand),
                    )
                except ResourceError:
                    # The inputs/output were already admitted. Direct einsum
                    # traverses broadcast strides without expanded copies.
                    return np.einsum(equation, *operands)
            return path.apply(*operands)
        return np.einsum(equation, *operands, optimize=list(path))

    def check(self, value, valid, message, *, xp=None, backend="numpy", error=ValidationError):
        """Raise eagerly; compiled invalid lanes remain visibly non-finite."""
        xp = np if xp is None else xp
        traced = False
        if backend == "torch":
            traced = _torch_compiling()
        elif backend == "jax":
            import jax

            traced = isinstance(valid, jax.core.Tracer)
        if not traced:
            if not bool(xp.all(valid)):
                raise error(message)
            return value
        label = f"[OpenQML:{error.__name__}] {message}"
        if backend == "torch":
            xp._assert_async(xp.all(valid), label)
        else:
            if _CHECKED_JAX_CONTEXT.get():
                from jax.experimental import checkify

                checkify.debug_check(xp.all(valid), label)
        return xp.where(valid, value, float("nan"))

    def finite(self, value, *, xp=None, backend="numpy", error=NumericalError):
        xp = np if xp is None else xp
        return self.check(
            value,
            xp.isfinite(value),
            "Compute input or result is not finite.",
            xp=xp,
            backend=backend,
            error=error,
        )

    def solve(self, matrix, rhs, *, xp=None, backend="numpy"):
        xp = np if xp is None else xp
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

    def factorized_solve(self, matrix, *, xp=None, backend="numpy"):
        """Reuse a native pivoted LU within one forward, never in a plan cache."""
        xp = np if xp is None else xp
        if backend == "numpy":
            # NumPy exposes no reusable LU object. Keep its checked LAPACK solve.
            return lambda rhs: self.solve(matrix, rhs, xp=xp, backend=backend)
        if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
            raise ValidationError("Linear solves require square coefficient matrices.")
        self._workspace((matrix,), tuple(matrix.shape), buffers=2)
        matrix = self.finite(matrix, xp=xp, backend=backend)
        if backend == "torch":
            lu, pivots, info = xp.linalg.lu_factor_ex(matrix, check_errors=False)
            valid = (info == 0) & xp.isfinite(lu).all(dim=(-1, -2))
        elif backend == "jax":
            from jax.scipy.linalg import lu_factor

            lu, pivots = lu_factor(matrix)
            valid = xp.all(xp.diagonal(lu, axis1=-2, axis2=-1) != 0, axis=-1)
            valid = valid & xp.all(xp.isfinite(lu), axis=(-1, -2))
        else:
            raise ValidationError("Linear solve backend must be numpy, torch or jax.")
        lu = self.check(
            lu,
            valid[..., None, None],
            "Linear system is singular or its factorization failed.",
            xp=xp,
            backend=backend,
            error=NumericalError,
        )

        def apply(rhs):
            self._workspace((matrix, lu, pivots, rhs), tuple(rhs.shape), buffers=2)
            rhs = self.finite(rhs, xp=xp, backend=backend)
            if backend == "torch":
                result = xp.linalg.lu_solve(lu, pivots, rhs)
            else:
                from jax.scipy.linalg import lu_solve

                result = lu_solve((lu, pivots), rhs)
            return self.finite(result, xp=xp, backend=backend)

        return apply

    def svd(self, matrix, *, xp=None, backend="numpy", full_matrices=False):
        xp = np if xp is None else xp
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

    def norm(self, value, *, xp=None, backend="numpy", axis=None, keepdims=False):
        from ._norms import vector_norm

        xp = np if xp is None else xp
        return vector_norm(value, xp, backend, axis=axis, keepdims=keepdims)


_KERNEL = ComputeKernel()
_COMPUTE_CONTEXT = ContextVar("openqml_compute_kernel", default=None)
_CHECKED_JAX_CONTEXT = ContextVar("openqml_checked_jax", default=False)
_CONFIG_LOCK = RLock()


def get_compute_kernel() -> ComputeKernel:
    """Return the shared process-local kernel without discovering devices."""
    if _torch_compiling():
        from ._torch import captured_kernel

        return captured_kernel()
    return _COMPUTE_CONTEXT.get() or _KERNEL


@contextmanager
def _checked_jax_context():
    token = _CHECKED_JAX_CONTEXT.set(True)
    try:
        yield
    finally:
        _CHECKED_JAX_CONTEXT.reset(token)


def configure_compute(*, policy=None) -> ComputeKernel:
    """Atomically install a new policy and empty shared metadata cache.

    Configure before starting worker threads or compiling native graphs. Existing
    Torch/JAX compiled graphs capture their numerical policy at trace time and
    must be retraced after reconfiguration. No hardware or jobs are accessed.
    """
    global _KERNEL
    kernel = ComputeKernel(policy)
    with _CONFIG_LOCK:
        _KERNEL = kernel
    return kernel


@contextmanager
def compute_context(*, policy=None, kernel=None):
    """Isolate a kernel in the current thread/async task and restore it on exit."""
    if policy is not None and kernel is not None:
        raise ValidationError("Select a kernel or a policy, not both.")
    selected = ComputeKernel(policy) if policy is not None else kernel or get_compute_kernel()
    if not isinstance(selected, ComputeKernel):
        raise ValidationError("kernel must be ComputeKernel.")
    token = _COMPUTE_CONTEXT.set(selected)
    try:
        yield selected
    finally:
        _COMPUTE_CONTEXT.reset(token)


def compute_operation(function):
    """Pin one policy snapshot across eager execution and nested numerical calls."""

    @wraps(function)
    def call(*args, **kwargs):
        if _torch_compiling():
            return function(*args, **kwargs)
        if _COMPUTE_CONTEXT.get() is not None:
            return function(*args, **kwargs)
        with compute_context():
            return function(*args, **kwargs)

    return call


def memory_budget(requested):
    """Apply the shared ceiling to a caller's operation-specific workspace."""
    return get_compute_kernel().memory_budget(requested)


class KernelOps:
    """Native-array bridge shared by quantum, hybrid-model and financial code."""

    @property
    def kernel(self):
        return self._kernel

    def matmul(self, left, right, *, out=None):
        return self.kernel.matmul(left, right, xp=self.xp, out=out)

    def einsum(self, equation, *operands):
        return self.kernel.einsum(equation, *operands, xp=self.xp, backend=self.backend)

    def solve(self, matrix, rhs):
        return self.kernel.solve(matrix, rhs, xp=self.xp, backend=self.backend)

    def broadcast_shape(self, *shapes):
        return _broadcast_shape(shapes)

    def check(self, value, valid, message, *, error=ValidationError):
        return self.kernel.check(
            value,
            valid,
            message,
            xp=self.xp,
            backend=self.backend,
            error=error,
        )

    def norm(self, value, axis=-1):
        return self.kernel.norm(value, xp=self.xp, backend=self.backend, axis=axis)


class RealOps(KernelOps):
    def __init__(self, like):
        self._kernel = get_compute_kernel()
        backend = _array_backend(like)
        self.like = like
        if backend == "torch":
            import torch

            self.name, self.xp = "torch", torch
        elif backend == "jax":
            import jax.numpy as jnp

            self.name, self.xp = "jax", jnp
        else:
            self.name, self.xp = "numpy", np
        self.backend = self.name

    def array(self, value, *, boolean=False):
        return self.kernel.asarray(
            value,
            xp=self.xp,
            backend=self.backend,
            like=self.like,
            dtype=(self.xp.bool if self.name == "torch" else bool) if boolean else self.like.dtype,
        )

    def stop(self, value):
        if self.name == "torch":
            return value.detach()
        if self.name == "jax":
            import jax

            return jax.lax.stop_gradient(value)
        return value

    def cat(self, values, axis=0):
        if self.name == "torch":
            return self.xp.cat(values, dim=axis)
        return self.xp.concatenate(values, axis=axis)

    def maximum(self, left, right):
        return self.xp.maximum(left, self.array(right))


def floating_array(value):
    backend = _array_backend(value)
    if backend == "numpy" and not isinstance(value, np.ndarray):
        try:
            value = np.asarray(value, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("Inputs must be real floating tensors.") from exc
    if str(value.dtype).removeprefix("torch.") not in {"float16", "bfloat16", "float32", "float64"}:
        raise ValidationError("Inputs must be real floating tensors.")
    # Accumulate losses and traces in at least float32 under mixed precision.
    if str(value.dtype).removeprefix("torch.") in {"float16", "bfloat16"}:
        return value.float() if backend == "torch" else value.astype("float32")
    return value
