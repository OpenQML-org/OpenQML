"""Exact funded cash-ray risk budgets including proportional and quadratic costs."""

from typing import NamedTuple

from ..errors import ConvergenceError, ValidationError
from ..ir.parameters import real
from ._array import Ops, admission, rows
from .expectiles import _expectile_root
from .risk import expected_shortfall
from .transport import _metric_values, transport_dual_norm


class ExecutionRiskResult(NamedTuple):
    weights: object
    positions: object
    cash: object
    scale: object
    costs: object
    nominal_risk: object
    robust_risk: object
    budget: object


def _upper_root(a, b, c, endpoint, ops):
    """Upper feasible quadratic root, evaluated only for an active upper bound."""
    xp = ops.xp
    active = a * endpoint * endpoint + b * endpoint + c > 0
    a, b, c = xp.where(active, a, 0), xp.where(active, b, 1), xp.where(active, c, -endpoint)
    magnitude = ops.stop(xp.maximum(xp.maximum(xp.abs(a), xp.abs(b)), xp.abs(c)))
    magnitude = xp.where(magnitude > 0, magnitude, 1)
    a, b, c = a / magnitude, b / magnitude, c / magnitude
    discriminant = xp.maximum(b * b - 4 * a * c, ops.array(0))
    root = xp.where(discriminant > 0, xp.sqrt(xp.where(discriminant > 0, discriminant, 1)), 0)
    positive = -2 * c / xp.where(b + root > 0, b + root, 1)
    negative = (-b + root) / xp.where(a > 0, 2 * a, 1)
    # The rational root also covers a=0 when b>=0, retaining the correct
    # derivative with respect to zero initial market impact.
    return ops.clip(xp.where(b >= 0, positive, negative), 0, endpoint)


def execution_risk_projection(
    weights,
    scenarios,
    *,
    previous=0,
    fees=0.001,
    impact=0,
    probabilities=None,
    risk_measure="cvar",
    confidence=0.95,
    budget=0.02,
    radius=0,
    cash_return=0,
    transport_metric=None,
    max_memory_mb=256,
    metric_iterations=128,
    metric_tolerance=1e-6,
):
    """Maximal cost-funded cash-ray scale meeting a CVaR or expectile budget.

    Proposal/previous positions are fractions of pre-trade NAV. Costs use
    sum(fees*abs(trades)+impact*trades**2) in the same units. Returned positions,
    cash and costs sum to one. Returned weights are post-cost targets accepted
    by SelfFinancingPlan; they differ from pre-trade positions when costs>0.
    Liquidation to cash must itself be feasible. No market order is submitted.
    """
    if not isinstance(risk_measure, str) or risk_measure not in {"cvar", "expectile"}:
        raise ValidationError("risk_measure must be cvar or expectile.")
    confidence = real(confidence)
    if not (0.5 if risk_measure == "expectile" else 0) <= confidence < 1:
        raise ValidationError("Use a valid CVaR confidence or coherent expectile level.")
    ops = Ops(
        weights,
        scenarios,
        previous,
        fees,
        impact,
        probabilities,
        budget,
        radius,
        cash_return,
        *_metric_values(transport_metric),
    )
    xp = ops.xp
    w, r, held, fee, quadratic = (ops.real(v) for v in (weights, scenarios, previous, fees, impact))
    if w.ndim < 1 or r.ndim < 2 or r.shape[-1] != w.shape[-1] or min(r.shape[-2:]) < 1:
        raise ValidationError("Execution risk requires matching nonempty scenario and asset axes.")
    r = ops.check(r, r >= -1, "Execution scenarios require arithmetic returns at least -1.")
    n = w.shape[-1]
    shape = ops.broadcast_shape(w.shape, held.shape, fee.shape, quadratic.shape)
    if shape[-1] != n:
        raise ValidationError("Execution cost and holding widths must match the assets.")
    w, held, fee, quadratic = (ops.broadcast(v, shape) for v in (w, held, fee, quadratic))
    # Ops accumulates in float32 or float64; keep static IEEE spacing out of
    # NumPy finfo reflection, which older Dynamo cannot trace in fullgraph.
    eps = 2.0 ** (-52 if "64" in str(w.dtype) else -23)
    w = ops.check(
        w,
        (w >= 0) & (ops.sum(w) <= 1 + 16 * n * eps)[..., None],
        "Proposals must be funded and long-only.",
    )
    held = ops.check(
        held,
        (held >= 0) & (ops.sum(held) <= 1 + 16 * n * eps)[..., None],
        "Previous holdings must be funded and long-only.",
    )
    w = w / ops.maximum(ops.sum(w, keepdims=True), 1)
    held = held / ops.maximum(ops.sum(held, keepdims=True), 1)
    fee = ops.check(fee, fee >= 0, "Trading fees must be nonnegative.")
    quadratic = ops.check(quadratic, quadratic >= 0, "Market impact must be nonnegative.")
    fee = ops.check(
        fee,
        (ops.max(fee) + 2 * ops.max(quadratic) < 1)[..., None],
        "Unique financing requires max(fees)+2*max(impact)<1.",
    )
    bound, rad, cash_rate = (ops.real(v) for v in (budget, radius, cash_return))
    rad = ops.check(rad, rad >= 0, "Wasserstein radii must be nonnegative.")
    cash_rate = ops.check(
        cash_rate, cash_rate >= -1, "Arithmetic cash returns must be at least -1."
    )
    p = ops.probabilities(probabilities, r.shape[-2])
    dual = transport_dual_norm(
        w,
        transport_metric,
        max_memory_mb=max_memory_mb,
        cg_iterations=metric_iterations,
        tolerance=metric_tolerance,
    )
    prefix = ops.broadcast_shape(
        shape[:-1], r.shape[:-2], p.shape[:-1], bound.shape, rad.shape, cash_rate.shape, dual.shape
    )
    admission(rows(prefix) * (100 * n + 40 * r.shape[-2] + 4 * rows(r.shape[-2:])), max_memory_mb)
    w, held, fee, quadratic = (ops.broadcast(v, prefix + (n,)) for v in (w, held, fee, quadratic))
    r = ops.broadcast(r, prefix + r.shape[-2:])
    p = ops.broadcast(p, prefix + (r.shape[-2],))
    bound, rad, cash_rate, dual = (ops.broadcast(v, prefix) for v in (bound, rad, cash_rate, dual))
    total = ops.sum(w)
    losses = cash_rate[..., None] * total[..., None] - ops.einsum("...sn,...n->...s", r, w)
    if risk_measure == "expectile":
        nominal = _expectile_root(losses, confidence, p, 0, max_memory_mb)
        robust = _expectile_root(losses, confidence, p, rad * dual, max_memory_mb)
    else:
        nominal = expected_shortfall(losses, confidence=confidence, probabilities=p)
        robust = nominal + rad * dual / (1 - confidence)
    coefficient = xp.ones_like(cash_rate) + cash_rate
    a = ops.sum(quadratic * w * w)
    b0 = -ops.sum(fee * w + 2 * quadratic * w * held)
    c0 = ops.sum(fee * held + quadratic * held * held)
    bound = ops.check(
        bound,
        (coefficient * c0 - cash_rate <= bound) & (c0 < 1),
        "The risk budget must admit fully funded liquidation costs.",
    )
    ratio = held / xp.where(w > 0, w, 1)
    knots = xp.where(w > 0, ops.clip(ratio, 0, 1), 1)
    flips = (w > 0) & (ratio < 1)
    order = ops.argsort(knots)
    knots = ops.gather(knots, order)
    zero = xp.zeros_like(knots[..., :1])
    slopes = (
        ops.cat((zero, ops.cumsum(ops.gather(xp.where(flips, 2 * fee * w, 0), order))), axis=-1)
        + b0[..., None]
    )
    intercepts = (
        ops.cat((zero, ops.cumsum(ops.gather(xp.where(flips, -2 * fee * held, 0), order))), axis=-1)
        + c0[..., None]
    )
    ends = ops.cat((knots, xp.ones_like(zero)), axis=-1)
    cost_at_end = a[..., None] * ends * ends + slopes * ends + intercepts
    funding = total[..., None] * ends + cost_at_end
    risk = robust[..., None] * ends - cash_rate[..., None] + coefficient[..., None] * cost_at_end
    violation = (funding > 1) | (risk > bound[..., None])
    index = (
        violation.to(xp.int64).argmax(dim=-1, keepdim=True)
        if ops.name == "torch"
        else xp.argmax(violation, axis=-1)[..., None]
    )
    endpoint = ops.gather(ends, index)[..., 0]
    b, c = (ops.gather(v, index)[..., 0] for v in (slopes, intercepts))
    finance_root = _upper_root(a, b + total, c - 1, endpoint, ops)
    risk_root = _upper_root(
        coefficient * a,
        coefficient * b + robust,
        coefficient * c - cash_rate - bound,
        endpoint,
        ops,
    )
    scale = xp.where(ops.sum(violation) > 0, xp.minimum(finance_root, risk_root), ops.array(1))
    positions = scale[..., None] * w
    trades = positions - held
    costs = ops.sum(fee * xp.abs(trades) + quadratic * trades * trades)
    cash = 1 - ops.sum(positions) - costs
    nominal_risk = scale * nominal - cash_rate + coefficient * costs
    robust_risk = scale * robust - cash_rate + coefficient * costs
    tolerance = 128 * n * eps * (1 + xp.abs(bound) + xp.abs(robust_risk))
    positions = ops.check(
        positions,
        (
            xp.isfinite(robust_risk)
            & (robust_risk <= bound + tolerance)
            & (cash >= -tolerance)
            & (costs < 1)
        )[..., None],
        "Execution projection exceeded its risk or funding tolerance.",
        error=ConvergenceError,
    )
    cash = xp.maximum(cash, ops.array(0))
    post_cost = positions / (1 - costs)[..., None]
    return ExecutionRiskResult(
        post_cost, positions, cash, scale, costs, nominal_risk, robust_risk, bound
    )
