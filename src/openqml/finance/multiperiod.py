"""Differentiable convex horizon allocation with coupled rebalancing decisions."""

from dataclasses import dataclass
from typing import NamedTuple

from ..compute.kernel import _torch_compiling, compute_operation
from ..errors import ConvergenceError, ValidationError, _require_instance
from ..ir.core import integer
from ..ir.parameters import real
from ._array import Ops, admission, machine_epsilon, rows
from ._temporal import hessian, project, solve
from .projection import _prox
from .risk import FactorRisk, prepare_risk


class MultiPeriodResult(NamedTuple):
    weights: object
    objective: object
    residual: object
    converged: object

    @property
    def first_weights(self):
        return self.weights[..., 0, :]


@dataclass(frozen=True)
class MultiPeriodPlan:
    """Minimize horizon risk/return and quadratic changes, with per-step caps/budgets.

    Each mean has shape [...,horizon,assets]. FactorRisk may be stationary or
    vary by period. Future decisions are open-loop forecasts, to be replanned
    after observations. Quadratic turnover is a regularizer; actual fees and
    price impact require the funded execution layer before sending a trade.
    """

    risk_aversion: float = 1
    ridge: float = 0.001
    turnover: float = 0.01
    max_iterations: int = 512
    cg_iterations: int = 128
    tolerance: float = 1e-6
    max_memory_mb: float = 256
    max_work: int = 10_000_000_000

    def __post_init__(self):
        for name in ("risk_aversion", "ridge", "turnover", "tolerance", "max_memory_mb"):
            object.__setattr__(self, name, real(getattr(self, name)))
        for name in ("max_iterations", "cg_iterations", "max_work"):
            object.__setattr__(self, name, integer(getattr(self, name), name, 1))
        if (
            self.risk_aversion < 0
            or self.turnover < 0
            or min(self.ridge, self.tolerance, self.max_memory_mb) <= 0
        ):
            raise ValidationError(
                "Use nonnegative risk/turnover and positive ridge/tolerance/budget."
            )

    def resource_profile(self, horizon, assets, rank, *, batch_size=1):
        h, n, r, b = (
            integer(horizon, "horizon", 1),
            integer(assets, "assets", 1),
            integer(rank, "rank", 0),
            integer(batch_size, "batch_size", 1),
        )
        elements = b * h * n * (12 * r + 160)
        levels, remaining = 0, n
        while remaining:
            levels, remaining = levels + 1, remaining // 2
        work = (
            8
            * b
            * h
            * n
            * (
                (self.max_iterations + 2) * (8 * (levels + 3) + 4 * max(1, r))
                + 16 * self.cg_iterations * max(1, r)
            )
        )
        admission(elements, self.max_memory_mb, work=work, max_work=self.max_work)
        return dict(
            horizon=h,
            assets=n,
            rank=r,
            batch_size=b,
            primal_workspace_bytes=16 * elements,
            arithmetic_work_estimate=work,
            solver="projected_factor_temporal_CG; no_dense_horizon_KKT",
            derivative="implicit_on_fixed_active_face",
            excludes="autodiff_tapes_allocator_and_framework_reserves",
        )

    @compute_operation
    def solve(self, mean, risk, *, previous=None, caps=1, budget=1, turnover=None, warm_start=None):
        _require_instance(risk, FactorRisk, "risk")
        ops = Ops(mean, *risk, previous, caps, budget, turnover)
        xp = ops.xp
        mean = ops.real(mean)
        if mean.ndim < 2 or min(mean.shape[-2:]) < 1:
            raise ValidationError("Expected returns require nonempty horizon and asset axes.")
        risk = prepare_risk(risk, ops, mean.shape)
        caps, budget = ops.real(caps), ops.real(budget)
        previous = ops.real(0 if previous is None else previous)
        turnover = ops.real(self.turnover if turnover is None else turnover)
        shape = ops.broadcast_shape(mean.shape, risk.mean.shape, caps.shape)
        h, n = shape[-2:]
        prefix = ops.broadcast_shape(
            shape[:-2],
            previous.shape[:-1] if previous.ndim else (),
            turnover.shape,
            budget.shape[:-1] if budget.ndim else (),
        )
        shape = prefix + (h, n)
        rank = risk.factors.shape[-2]
        self.resource_profile(h, n, rank, batch_size=rows(prefix))
        mean, caps = (ops.broadcast(value, shape) for value in (mean, caps))
        previous = ops.broadcast(previous, prefix + (n,))
        budget = ops.broadcast(budget, prefix + (h,))
        turnover = ops.broadcast(turnover, prefix)
        factors = ops.broadcast(risk.factors, prefix + (h, rank, n))
        mass = self.risk_aversion * ops.broadcast(risk.factor_weights, prefix + (h, rank))
        diagonal = self.risk_aversion * ops.broadcast(risk.diagonal, shape) + self.ridge
        caps = ops.check(caps, caps >= 0, "Position caps must be nonnegative.")
        previous = ops.check(previous, previous >= 0, "Previous positions must be nonnegative.")
        turnover = ops.check(
            turnover, turnover >= 0, "Turnover regularization must be nonnegative."
        )
        eps = machine_epsilon(mean, ops)
        tolerance = max(self.tolerance, 64 * eps)
        capacity = ops.sum(caps)
        budget = ops.check(
            budget,
            (budget >= 0) & (budget <= capacity + 8 * eps * n),
            "Period budgets exceed available position capacity.",
        )
        budget = xp.minimum(budget, capacity)
        first = (turnover[..., None] * previous)[..., None, :]
        linear = mean + ops.cat((first, xp.zeros_like(mean[..., 1:, :])), axis=-2)
        constants = tuple(
            ops.stop(v) for v in (diagonal, factors, mass, turnover, linear, caps, budget)
        )
        seed = None if warm_start is None else ops.stop(ops.broadcast(ops.real(warm_start), shape))
        approximate = self._identify(*constants, ops, seed)
        threshold = xp.minimum(ops.array(max(1e-8, 16 * eps)), caps * 0.25)
        upper = (approximate > threshold) & (caps - approximate <= threshold)
        free = (approximate > threshold) & (caps - approximate > threshold)
        free = ops.stop(free.to(mean.dtype) if ops.name == "torch" else free.astype(mean.dtype))
        fixed = xp.where(upper, caps, 0)
        count = ops.sum(free)
        anchor = (
            fixed + free * ((budget - ops.sum(fixed)) / xp.where(count > 0, count, 1))[..., None]
        )
        rhs = project(linear - hessian(diagonal, factors, mass, turnover, anchor, ops), free, ops)
        tangent = solve(
            diagonal,
            factors,
            mass,
            turnover,
            free,
            rhs,
            self.cg_iterations,
            max(self.tolerance * 0.1, 16 * eps),
            ops,
        )
        weights = anchor + project(tangent, free, ops)
        gradient = hessian(diagonal, factors, mass, turnover, weights, ops) - linear
        projected = _prox(
            weights - gradient, xp.zeros_like(weights), caps, xp.zeros_like(weights), budget, ops
        )
        residual = ops.max(xp.abs(projected - weights), axis=(-2, -1))
        feasibility = ops.max(xp.abs(ops.sum(weights) - budget), axis=-1)
        bounds = ops.max(xp.maximum(-weights, weights - caps), axis=(-2, -1))
        scale = xp.maximum(ops.array(1), ops.max(xp.abs(linear), axis=(-2, -1)))
        converged = (
            (residual <= tolerance * scale) & (feasibility <= tolerance) & (bounds <= tolerance)
        )
        weights = ops.check(
            weights,
            converged[..., None, None],
            "Horizon allocation failed its feasibility/KKT tolerance; "
            "increase iterations or precision.",
            error=ConvergenceError,
        )
        applied = hessian(diagonal, factors, mass, turnover, weights, ops)
        objective = ops.sum(0.5 * weights * applied - linear * weights, axis=(-2, -1))
        objective = objective + 0.5 * turnover * ops.sum(previous * previous)
        return MultiPeriodResult(weights, objective, residual, converged)

    def _identify(self, diagonal, factors, mass, turnover, linear, caps, budget, ops, seed):
        xp = ops.xp
        bound = ops.max(diagonal, axis=-1) + ops.sum(
            mass * ops.sum(factors * factors, axis=-1), axis=-1
        )
        lipschitz = ops.max(bound, axis=-1) + 4 * turnover
        lower = ops.min(diagonal, axis=(-2, -1))
        beta = (xp.sqrt(lipschitz) - xp.sqrt(lower)) / (xp.sqrt(lipschitz) + xp.sqrt(lower))
        step_size = 1 / lipschitz[..., None, None]
        zeros = xp.zeros_like(linear)
        initial = (
            caps
            * budget[..., None]
            / xp.where(ops.sum(caps, keepdims=True) > 0, ops.sum(caps, keepdims=True), 1)
        )
        if seed is not None:
            initial = _prox(seed, zeros, caps, zeros, budget, ops)

        def advance(state):
            current, extrapolated = state
            gradient = hessian(diagonal, factors, mass, turnover, extrapolated, ops) - linear
            next_value = _prox(extrapolated - step_size * gradient, zeros, caps, zeros, budget, ops)
            return next_value, next_value + beta[..., None, None] * (next_value - current)

        if ops.name == "jax":
            import jax

            return jax.lax.fori_loop(
                0, self.max_iterations, lambda _, state: advance(state), (initial, initial)
            )[0]
        if ops.name == "torch" and _torch_compiling() and hasattr(xp, "while_loop"):

            def condition(index, current, extrapolated, delta):
                return (index < self.max_iterations) & xp.any(delta > self.tolerance * 0.02)

            def step(index, current, extrapolated, delta):
                value, next_extrapolated = advance((current, extrapolated))
                change = ops.max(xp.abs(value - current), axis=(-2, -1))
                return index + 1, value, next_extrapolated, change

            counter = xp.zeros((), dtype=xp.int64, device=linear.device)
            delta = xp.full(
                linear.shape[:-2], float("inf"), device=linear.device, dtype=linear.dtype
            )
            return ops.stop(
                xp.while_loop(condition, step, (counter, initial, initial.clone(), delta))[1]
            )
        state = initial, initial
        for _ in range(self.max_iterations):
            updated = advance(state)
            if not (ops.name == "torch" and _torch_compiling()):
                if bool(
                    xp.all(
                        ops.max(xp.abs(updated[0] - state[0]), axis=(-2, -1))
                        < self.tolerance * 0.02
                    )
                ):
                    state = updated
                    break
            state = updated
        return ops.stop(state[0])
