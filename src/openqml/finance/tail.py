"""Joint-path, distributionally robust CVaR decisions with implicit derivatives."""

from dataclasses import dataclass
from typing import NamedTuple

from ..compute.kernel import compute_operation
from ..errors import ConvergenceError, ValidationError
from ..ir.core import integer
from ..ir.parameters import real
from ._array import Ops, admission, machine_epsilon, rows
from ._tail_linear import cg, project, solve
from ._temporal import temporal
from .risk import expected_shortfall


class TailAllocationResult(NamedTuple):
    weights: object
    threshold: object
    cvar: object
    robust_cvar: object
    smooth_risk: object
    objective: object
    residual: object
    converged: object

    @property
    def first_weights(self):
        return self.weights[..., 0, :]


@dataclass(frozen=True)
class TailAllocationPlan:
    """Optimize incremental dollar P&L on joint return paths [...,S,H,N].

    The strictly convex objective combines softplus CVaR, a Euclidean W1
    margin, expected return, ridge, quadratic rebalancing and a log barrier.
    Per-period budgets and caps are strict interior constraints. Future
    positions are open-loop; fees, funding and compound wealth use the
    execution/accounting APIs. The implicit derivative excludes solver tapes.
    """

    confidence: float = 0.95
    risk_aversion: float = 1
    smoothing: float = 0.02
    barrier: float = 0.001
    ridge: float = 0.01
    turnover: float = 0.01
    max_iterations: int = 48
    cg_iterations: int = 128
    tolerance: float = 1e-7
    max_memory_mb: float = 256
    max_work: int = 10_000_000_000

    def __post_init__(self):
        for name in (
            "confidence",
            "risk_aversion",
            "smoothing",
            "barrier",
            "ridge",
            "turnover",
            "tolerance",
            "max_memory_mb",
        ):
            object.__setattr__(self, name, real(getattr(self, name)))
        for name in ("max_iterations", "cg_iterations", "max_work"):
            object.__setattr__(self, name, integer(getattr(self, name), name, 1))
        if (
            not 0 < self.confidence < 1
            or self.turnover < 0
            or min(
                self.risk_aversion,
                self.smoothing,
                self.barrier,
                self.ridge,
                self.tolerance,
                self.max_memory_mb,
            )
            <= 0
        ):
            raise ValidationError(
                "Use confidence in (0,1), positive scales and nonnegative turnover."
            )

    def resource_profile(self, scenarios, horizon, assets, *, batch_size=1):
        s, h, n, b = (
            integer(scenarios, "scenarios", 1),
            integer(horizon, "horizon", 1),
            integer(assets, "assets", 1),
            integer(batch_size, "batch_size", 1),
        )
        elements = b * (32 * s * h * n + 128 * h * n + 32 * s)
        work = 16 * b * s * h * n * self.max_iterations * (self.cg_iterations + 32)
        admission(elements, self.max_memory_mb, work=work, max_work=self.max_work)
        return dict(
            scenarios=s,
            horizon=h,
            assets=n,
            batch_size=b,
            primal_workspace_bytes=16 * elements,
            arithmetic_work_estimate=work,
            solver="interior_Newton_CG; joint_path_factors; no_dense_KKT",
            derivative="implicit_strict_interior_root",
            excludes="autodiff_tapes_compiler_workspaces_and_allocator_reserves",
        )

    @compute_operation
    def solve(
        self,
        scenarios,
        *,
        probabilities=None,
        caps=1,
        budget=1,
        previous=None,
        turnover=None,
        radius=0,
        smoothing=None,
    ):
        ops = Ops(scenarios, probabilities, caps, budget, previous, turnover, radius, smoothing)
        xp = ops.xp
        returns = ops.real(scenarios)
        if returns.ndim < 3 or min(returns.shape[-3:]) < 1:
            raise ValidationError("Return paths require nonempty scenario, horizon and asset axes.")
        s, h, n = returns.shape[-3:]
        probability = ops.probabilities(probabilities, s)
        caps, budget = ops.real(caps), ops.real(budget)
        previous = ops.real(0 if previous is None else previous)
        turnover = ops.real(self.turnover if turnover is None else turnover)
        radius = ops.real(radius)
        smoothing = ops.real(self.smoothing if smoothing is None else smoothing)
        capshape = ops.broadcast_shape(caps.shape, (h, n))
        if capshape[-2:] != (h, n):
            raise ValidationError("Caps must broadcast to the declared horizon and assets.")
        prefix = ops.broadcast_shape(
            returns.shape[:-3],
            probability.shape[:-1],
            capshape[:-2],
            budget.shape[:-1] if budget.ndim else (),
            previous.shape[:-1] if previous.ndim else (),
            turnover.shape,
            radius.shape,
            smoothing.shape,
        )
        self.resource_profile(s, h, n, batch_size=rows(prefix))
        parameters = (
            ops.broadcast(returns, prefix + (s, h, n)),
            ops.broadcast(probability, prefix + (s,)),
            ops.broadcast(caps, prefix + (h, n)),
            ops.broadcast(budget, prefix + (h,)),
            ops.broadcast(previous, prefix + (n,)),
            ops.broadcast(turnover, prefix),
            ops.broadcast(radius, prefix),
            ops.broadcast(smoothing, prefix),
        )
        returns, probability, caps, budget, previous, turnover, radius, smoothing = parameters
        caps = ops.check(caps, caps > 0, "Path-tail caps must be strictly positive.")
        budget = ops.check(
            budget,
            (budget > 0) & (budget < ops.sum(caps)),
            "Path-tail budgets must lie strictly between zero and period capacity.",
        )
        turnover = ops.check(turnover, turnover >= 0, "Turnover must be nonnegative.")
        radius = ops.check(radius, radius >= 0, "Wasserstein radii must be nonnegative.")
        smoothing = ops.check(smoothing, smoothing > 0, "CVaR smoothing must be positive.")
        parameters = returns, probability, caps, budget, previous, turnover, radius, smoothing
        if ops.name == "torch":
            from ._torch_tail import root

            value = root(parameters, self)
        elif ops.name == "jax":
            import jax

            def tangent_solve(operator, rhs):
                return jax.lax.custom_linear_solve(
                    operator,
                    rhs,
                    solve=lambda _, x: cg(operator, x, self.cg_iterations, self.tolerance, ops),
                    symmetric=True,
                )

            initial = xp.zeros(prefix + (h * n + 1,), dtype=returns.dtype)
            value = jax.lax.custom_root(
                lambda x: _stationarity(x, parameters, self, ops),
                initial,
                solve=lambda _, x: _newton(parameters, self, ops),
                tangent_solve=tangent_solve,
            )
        else:
            value = _newton(parameters, self, ops)
        weights, threshold = _positions(value, parameters, ops)
        losses = -ops.einsum("...shn,...hn->...s", returns, weights)
        empirical = expected_shortfall(
            losses, confidence=self.confidence, probabilities=probability
        )
        margin = radius * ops.norm(weights, axis=(-2, -1)) / (1 - self.confidence)
        smooth = _smooth(losses, threshold, probability, smoothing, self, ops) + margin
        residual = ops.max(xp.abs(_stationarity(value, parameters, self, ops)))
        tolerance = max(self.tolerance, 128 * machine_epsilon(value, ops))
        weights = ops.check(
            weights,
            (residual <= tolerance)[..., None, None],
            "Path-tail optimizer failed its KKT residual; increase iterations or smoothing.",
            error=ConvergenceError,
        )
        return TailAllocationResult(
            weights,
            threshold,
            empirical,
            empirical + margin,
            smooth,
            _objective(value, parameters, self, ops),
            residual,
            residual <= tolerance,
        )


def _positions(value, parameters, ops):
    returns, _, caps, budget, *_ = parameters
    h, n = returns.shape[-2:]
    anchor = caps * (budget / ops.sum(caps))[..., None]
    weights = project(value, h, n, ops)[..., :-1].reshape(anchor.shape)
    return anchor + weights, value[..., -1]


def _sigmoid(value, ops):
    if ops.name == "torch":
        return ops.xp.sigmoid(value)
    if ops.name == "jax":
        from jax.nn import sigmoid

        return sigmoid(value)
    return ops.xp.exp(-ops.xp.logaddexp(0, -value))


def _smooth(losses, threshold, probability, smoothing, plan, ops):
    scaled = (losses - threshold[..., None]) / smoothing[..., None]
    soft = ops.xp.logaddexp(ops.xp.zeros_like(scaled), scaled)
    return threshold + smoothing * ops.sum(probability * soft) / (1 - plan.confidence)


def _objective(value, parameters, plan, ops):
    returns, probability, caps, _, previous, turnover, radius, smoothing = parameters
    weights, threshold = _positions(value, parameters, ops)
    losses = -ops.einsum("...shn,...hn->...s", returns, weights)
    changes = weights - ops.cat((previous[..., None, :], weights[..., :-1, :]), axis=-2)
    norm = ops.norm(weights, axis=(-2, -1))
    risk = _smooth(losses, threshold, probability, smoothing, plan, ops)
    risk = risk + radius * norm / (1 - plan.confidence)
    barrier = ops.sum(ops.xp.log(weights) + ops.xp.log(caps - weights), axis=(-2, -1))
    return (
        plan.risk_aversion * risk
        + ops.sum(probability * losses)
        + plan.ridge * ops.sum(weights * weights, axis=(-2, -1)) / 2
        + turnover * ops.sum(changes * changes, axis=(-2, -1)) / 2
        - plan.barrier * barrier
    )


def _stationarity(value, parameters, plan, ops):
    returns, probability, caps, _, previous, turnover, radius, smoothing = parameters
    h, n = returns.shape[-2:]
    weights, threshold = _positions(value, parameters, ops)
    losses = -ops.einsum("...shn,...hn->...s", returns, weights)
    sigma = _sigmoid((losses - threshold[..., None]) / smoothing[..., None], ops)
    tail = 1 - plan.confidence
    norm = ops.norm(weights, axis=(-2, -1))
    gradient = -ops.einsum("...s,...shn->...hn", probability, returns)
    gradient = gradient - plan.risk_aversion * ops.einsum(
        "...s,...shn->...hn", probability * sigma / tail, returns
    )
    gradient = (
        gradient
        + (plan.ridge + plan.risk_aversion * radius / (tail * norm))[..., None, None] * weights
    )
    gradient = gradient + turnover[..., None, None] * temporal(weights, ops)
    first = ops.cat((previous[..., None, :], ops.xp.zeros_like(weights[..., 1:, :])), axis=-2)
    gradient = gradient - turnover[..., None, None] * first
    gradient = gradient + plan.barrier * (1 / (caps - weights) - 1 / weights)
    eta = plan.risk_aversion * (1 - ops.sum(probability * sigma) / tail)
    full = ops.cat((gradient.reshape(value.shape[:-1] + (-1,)), eta[..., None]), axis=-1)
    return project(full, h, n, ops) + value - project(value, h, n, ops)


def _geometry(value, parameters, plan, ops):
    returns, probability, caps, _, _, _, radius, smoothing = parameters
    weights, threshold = _positions(value, parameters, ops)
    losses = -ops.einsum("...shn,...hn->...s", returns, weights)
    sigma = _sigmoid((losses - threshold[..., None]) / smoothing[..., None], ops)
    tail = 1 - plan.confidence
    norm = ops.norm(weights, axis=(-2, -1))
    d = plan.ridge + plan.barrier * (1 / weights**2 + 1 / (caps - weights) ** 2)
    d = d + (plan.risk_aversion * radius / (tail * norm))[..., None, None]
    diagonal = ops.cat(
        (d.reshape(value.shape[:-1] + (-1,)), ops.xp.zeros_like(threshold[..., None])), axis=-1
    )
    shape = returns.shape[:-2] + (-1,)
    factors = ops.cat((-returns.reshape(shape), -ops.xp.ones_like(losses[..., None])), axis=-1)
    radial = ops.cat(
        (weights.reshape(value.shape[:-1] + (-1,)), ops.xp.zeros_like(threshold[..., None])),
        axis=-1,
    )
    factors = ops.cat((factors, radial[..., None, :]), axis=-2)
    mass = plan.risk_aversion * probability * sigma * (1 - sigma) / (smoothing[..., None] * tail)
    mass = ops.cat((mass, (-plan.risk_aversion * radius / (tail * norm**3))[..., None]), axis=-1)
    return diagonal, factors, mass


def _newton(parameters, plan, ops):
    xp = ops.xp
    returns, probability, caps, budget, *_ = parameters
    h, n = returns.shape[-2:]
    shape = returns.shape[:-3] + (h * n + 1,)
    anchor = caps * (budget / ops.sum(caps))[..., None]
    losses = -ops.einsum("...shn,...hn->...s", returns, anchor)
    tau = parameters[-1]
    low, high = ops.min(losses) - 40 * tau, ops.max(losses) + 40 * tau
    for _ in range(48):
        eta = (low + high) / 2
        tail = ops.sum(probability * _sigmoid((losses - eta[..., None]) / tau[..., None], ops))
        low, high = (
            xp.where(tail > 1 - plan.confidence, eta, low),
            xp.where(tail > 1 - plan.confidence, high, eta),
        )
    zero = xp.zeros_like(anchor).reshape(shape[:-1] + (-1,))
    initial = ops.cat((zero, ((low + high) / 2)[..., None]), axis=-1)
    tolerance = max(plan.tolerance, 128 * machine_epsilon(initial, ops))

    def advance(value):
        gradient = _stationarity(value, parameters, plan, ops)
        active = ops.max(xp.abs(gradient)) > tolerance
        diagonal, factors, mass = _geometry(value, parameters, plan, ops)
        step = solve(diagonal, factors, mass, parameters[5], -gradient, h, n, plan, ops)
        weights, _ = _positions(value, parameters, ops)
        dw = project(step, h, n, ops)[..., :-1].reshape(weights.shape)
        safe = xp.where(dw != 0, dw, 1)
        limit = xp.where(
            dw > 0,
            0.99 * (caps - weights) / safe,
            xp.where(dw < 0, -0.99 * weights / safe, float("inf")),
        )
        alpha = xp.minimum(ops.min(limit, axis=(-2, -1)), ops.array(1))
        objective = _objective(value, parameters, plan, ops)
        slope = ops.sum(gradient * step)

        def line_step(state):
            iteration, alpha, accepted, selected = state
            candidate = value + alpha[..., None] * step
            cost = _objective(candidate, parameters, plan, ops)
            valid = xp.isfinite(cost) & (
                cost <= objective + 1e-4 * alpha * slope + 8 * machine_epsilon(cost, ops)
            )
            choose = active & ~accepted & valid
            selected = xp.where(choose[..., None], candidate, selected)
            accepted = accepted | valid
            return iteration + 1, alpha / 2, accepted, selected

        state = 0, alpha, ~active, value
        if ops.name == "jax":
            import jax

            state = jax.lax.while_loop(lambda s: (s[0] < 32) & xp.any(~s[2]), line_step, state)
        else:
            for _ in range(32):
                if bool(xp.all(state[2])):
                    break
                state = line_step(state)
        return state[-1]

    if ops.name == "jax":
        import jax

        def condition(carry):
            i, value = carry
            residual = ops.max(xp.abs(_stationarity(value, parameters, plan, ops)))
            return (i < plan.max_iterations) & xp.any(residual > tolerance)

        result = jax.lax.while_loop(condition, lambda c: (c[0] + 1, advance(c[1])), (0, initial))[1]
    else:
        result = initial
        for _ in range(plan.max_iterations):
            if bool(
                xp.all(ops.max(xp.abs(_stationarity(result, parameters, plan, ops))) <= tolerance)
            ):
                break
            result = advance(result)
    residual = ops.max(xp.abs(_stationarity(result, parameters, plan, ops)))
    return ops.check(
        result,
        (residual <= tolerance)[..., None],
        "Path-tail optimizer failed its KKT residual; increase iterations or smoothing.",
        error=ConvergenceError,
    )
