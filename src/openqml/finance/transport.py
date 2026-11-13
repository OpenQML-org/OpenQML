"""Differentiable anisotropic Wasserstein norms with certified factor solves."""

from typing import NamedTuple

from ..errors import NumericalError, ValidationError, _require_instance
from ..ir.core import integer
from ..ir.parameters import real
from ._array import Ops, admission, rows
from ._linear import linear_solve


class WassersteinMetric(NamedTuple):
    """Ground cost ||delta||_M, M=diag(diagonal)+factors.T@factors.

    Diagonal entries must be strictly positive. Factors are optional [...,R,N]
    native arrays, so a contextual model can learn the metric within a forward.
    No covariance matrix or inverse is materialized or cached.
    """

    diagonal: object
    factors: object = None


def _metric_values(metric):
    if metric is None:
        return ()
    _require_instance(metric, WassersteinMetric, "transport_metric")
    return metric.diagonal, metric.factors


def transport_dual_norm(
    weights,
    metric=None,
    *,
    max_memory_mb=256,
    max_work=1_000_000_000,
    cg_iterations=128,
    tolerance=1e-6,
):
    """Return sqrt(w.T @ M^-1 @ w), or the Euclidean norm when metric=None.

    Use the shared implicit Torch/JAX factor solve and its true-residual check.
    The result retains device placement and gradients in weights and the metric.
    Working-memory admission excludes framework autodiff tapes and allocators.
    """
    values = _metric_values(metric)
    ops = Ops(weights, *values)
    w = ops.real(weights)
    if w.ndim < 1 or w.shape[-1] < 1:
        raise ValidationError("Transport norms require a nonempty asset axis.")
    admission(8 * rows(w.shape), max_memory_mb, work=16 * rows(w.shape), max_work=max_work)
    if metric is None:
        norm = ops.norm(w)
        return ops.check(
            norm,
            ops.xp.isfinite(norm),
            "Transport dual norm exceeds finite precision.",
            error=NumericalError,
        )
    iterations = integer(cg_iterations, "cg_iterations", 1)
    tolerance = real(tolerance)
    if tolerance <= 0:
        raise ValidationError("Transport solve tolerance must be positive.")
    diagonal = ops.real(metric.diagonal)
    n = w.shape[-1]
    if diagonal.shape[-1:] != (n,):
        raise ValidationError("Transport diagonal must match the asset axis.")
    factors = None if metric.factors is None else ops.real(metric.factors)
    if factors is not None and (factors.ndim < 2 or factors.shape[-1] != n):
        raise ValidationError("Transport factors must have shape [...,rank,assets].")
    rank = 0 if factors is None else factors.shape[-2]
    prefix = ops.broadcast_shape(
        w.shape[:-1], diagonal.shape[:-1], () if factors is None else factors.shape[:-2]
    )
    count = rows(prefix)
    small = rank if rank <= 64 and 4 * rank <= n else 0
    work = (
        n if rank == 0 else rank * rank * n + rank**3 + rank * n if small else iterations * rank * n
    )
    admission(
        count * (12 * rank * n + 16 * n + 8 * small * small),
        max_memory_mb,
        work=16 * count * work,
        max_work=max_work,
    )
    w = ops.broadcast(w, prefix + (n,))
    diagonal = ops.broadcast(diagonal, prefix + (n,))
    diagonal = ops.check(diagonal, diagonal > 0, "Transport diagonal must be strictly positive.")
    if rank == 0:
        norm = ops.norm(w / ops.xp.sqrt(diagonal))
        return ops.check(
            norm,
            ops.xp.isfinite(norm),
            "Transport dual norm exceeds finite precision.",
            error=NumericalError,
        )
    factors = ops.broadcast(factors, prefix + (rank, n))
    # Normalize the RHS before the solve. The norm of the whitened solution
    # avoids sqrt(w*y) cancellation and has the shared zero-vector derivative.
    scale = ops.stop(ops.max(ops.xp.abs(w), keepdims=True))
    scale = ops.xp.where(scale > 0, scale, ops.array(1))
    mass = ops.xp.ones_like(factors[..., :, 0])
    solution = linear_solve(
        diagonal, factors, mass, (w / scale)[..., None], iterations, tolerance, ops
    )[..., 0]
    whitened = ops.cat(
        (ops.xp.sqrt(diagonal) * solution, ops.einsum("...rn,...n->...r", factors, solution)),
        axis=-1,
    )
    norm = ops.norm(whitened) * scale[..., 0]
    return ops.check(
        norm,
        ops.xp.isfinite(norm),
        "Transport dual norm exceeds finite precision.",
        error=NumericalError,
    )
