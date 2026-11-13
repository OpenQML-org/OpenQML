"""Closed-form, differentiable CVaR budgeting with explicit cash and robust margins."""

from typing import NamedTuple

import numpy as np

from ..errors import ValidationError
from ..ir.parameters import real
from ._array import Ops, admission, rows
from .risk import expected_shortfall


class RiskBudgetResult(NamedTuple):
    weights: object
    cash: object
    scale: object
    cvar: object
    robust_cvar: object
    budget: object


def risk_budget_projection(
    weights,
    scenarios,
    *,
    probabilities=None,
    confidence=0.95,
    budget=0.02,
    radius=0,
    cash_return=0,
    max_memory_mb=256,
    transport_metric=None,
    metric_iterations=128,
    metric_tolerance=1e-6,
):
    """Scale a funded long-only proposal toward cash to meet its CVaR bound.

    The risk bound combines scenario CVaR with radius*dual_norm(w)/(1-confidence).
    The ground norm is Euclidean unless a WassersteinMetric is supplied.
    Positive homogeneity and cash translation give the maximal feasible scale
    in closed form. Inputs retain native probability, scenario and proposal
    gradients; callers supply causal scenarios and a calibrated robust radius.
    Cash has a deterministic arithmetic return for the same scenario horizon.
    """
    from .transport import _metric_values, transport_dual_norm

    confidence = real(confidence)
    if not 0 <= confidence < 1:
        raise ValidationError("Use confidence in [0,1).")
    ops = Ops(
        weights,
        scenarios,
        probabilities,
        budget,
        cash_return,
        radius,
        *_metric_values(transport_metric),
    )
    w, r = ops.real(weights), ops.real(scenarios)
    if w.ndim < 1 or r.ndim < 2 or w.shape[-1] != r.shape[-1] or min(r.shape[-2:]) < 1:
        raise ValidationError("Proposal and scenario asset axes must match and be nonempty.")
    bound, cash_rate, radius = (ops.real(v) for v in (budget, cash_return, radius))
    radius = ops.check(radius, radius >= 0, "Robust radii must be nonnegative.")
    w = ops.check(w, w >= 0, "Risk-budget proposals must be long-only.")
    total = ops.sum(w)
    eps = np.finfo(np.float64 if "64" in str(w.dtype) else np.float32).eps
    total = ops.check(
        total, total <= 1 + 16 * eps * w.shape[-1], "Risk-budget proposals must be fully funded."
    )
    w = w / ops.maximum(total, 1)[..., None]
    dual = transport_dual_norm(
        w,
        transport_metric,
        max_memory_mb=max_memory_mb,
        cg_iterations=metric_iterations,
        tolerance=metric_tolerance,
    )
    prefix = ops.broadcast_shape(
        w.shape[:-1], r.shape[:-2], bound.shape, cash_rate.shape, radius.shape, dual.shape
    )
    if probabilities is not None:
        p = ops.probabilities(probabilities, r.shape[-2])
        prefix = ops.broadcast_shape(prefix, p.shape[:-1])
    else:
        p = None
    admission(
        rows(prefix) * (4 * r.shape[-2] * r.shape[-1] + 12 * r.shape[-2] + 8 * w.shape[-1]),
        max_memory_mb,
    )
    w = ops.broadcast(w, prefix + (w.shape[-1],))
    r = ops.broadcast(r, prefix + r.shape[-2:])
    bound, cash_rate = ops.broadcast(bound, prefix), ops.broadcast(cash_rate, prefix)
    total = ops.sum(w)
    target = ops.check(
        bound + cash_rate, bound + cash_rate >= 0, "The risk budget must admit the cash position."
    )
    incremental = cash_rate[..., None] * total[..., None] - ops.einsum("...sn,...n->...s", r, w)
    base_cvar = expected_shortfall(incremental, confidence=confidence, probabilities=p)
    robust = base_cvar + radius * dual / (1 - confidence)
    safe_denominator = ops.xp.where(robust > target, robust, ops.array(1))
    scale = ops.xp.where(robust > target, target / safe_denominator, ops.array(1))
    scale = ops.check(
        scale,
        ops.xp.isfinite(target) & ops.xp.isfinite(robust),
        "Risk-budget inputs must define finite admissible risk and cash bounds.",
    )
    projected = w * scale[..., None]
    return RiskBudgetResult(
        projected,
        ops.xp.maximum(1 - ops.sum(projected), ops.array(0)),
        scale,
        -cash_rate + scale * base_cvar,
        -cash_rate + scale * robust,
        bound,
    )
