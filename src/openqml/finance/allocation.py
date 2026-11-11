"""Cost-aware portfolio layers with factor-space implicit differentiation.

The strictly convex objective is .5*risk_aversion*w'Cov*w - mean'w +
.5*ridge*||w||² + .5*turnover*||w-previous||² + fees'|w-previous|.
Feasibility is 0<=w<=caps, sum(w)=budget. FISTA identifies a face; a
matrix-free constrained solve refines it and supplies piecewise implicit AD.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..compute import _torch_compiling, compute_operation
from ..errors import ValidationError
from ..ir import integer
from ..ir.parameters import real
from ._array import Ops, admission, rows
from ._linear import linear_solve, matmat
from .projection import _prox
from .risk import FactorRisk, prepare_risk


class AllocationResult(NamedTuple):
    weights: object
    objective: object
    residual: object
    converged: object


@dataclass(frozen=True)
class AllocationPlan:
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
            or self.ridge <= 0
            or self.turnover < 0
            or self.tolerance <= 0
            or self.max_memory_mb <= 0
        ):
            raise ValidationError(
                "Allocation requires nonnegative risk/turnover and positive "
                "ridge, tolerance and budget."
            )

    def resource_profile(self, assets, rank, *, batch_size=1):
        n, r, b = (
            integer(assets, "assets", 1),
            integer(rank, "rank", 0),
            integer(batch_size, "batch_size", 1),
        )
        small = min(r, n)
        elements = b * (8 * r * n + 4 * small * small + 80 * n)
        work = (
            8
            * b
            * (
                small * r * n
                + small**3
                + (self.max_iterations + 4 * self.cg_iterations) * max(1, r) * n
            )
        )
        admission(elements, self.max_memory_mb, work=work, max_work=self.max_work)
        return dict(
            assets=n,
            rank=r,
            batch_size=b,
            primal_workspace_bytes=16 * elements,
            arithmetic_upper_estimate=work,
            solver="diagonal"
            if r == 0
            else "factor_Woodbury_refinement"
            if r <= 64 and 4 * r <= n
            else "factor_PCG; no_dense_asset_KKT",
            derivative="implicit_on_fixed_active_face",
            excludes="autodiff_tapes_allocator_and_framework_reserves",
        )

    @compute_operation
    def solve(self, mean, risk, *, previous=None, caps=1, fees=0, budget=1, warm_start=None):
        ops = Ops(
            mean, *(risk if isinstance(risk, FactorRisk) else ()), previous, caps, fees, budget
        )
        xp = ops.xp
        mean = ops.real(mean)
        if mean.ndim < 1 or mean.shape[-1] < 1:
            raise ValidationError("Expected returns require a nonempty asset axis.")
        risk = prepare_risk(risk, ops, mean.shape)
        previous = ops.real(0 if previous is None else previous)
        caps, fees, budget = (ops.real(v) for v in (caps, fees, budget))
        shape = np.broadcast_shapes(
            mean.shape, risk.mean.shape, previous.shape, caps.shape, fees.shape
        )
        prefix = np.broadcast_shapes(shape[:-1], budget.shape)
        shape = prefix + (shape[-1],)
        n, rank = shape[-1], risk.factors.shape[-2]
        self.resource_profile(n, rank, batch_size=rows(prefix))
        mean, previous, caps, fees = (ops.broadcast(v, shape) for v in (mean, previous, caps, fees))
        factors = ops.broadcast(risk.factors, prefix + (rank, n))
        mass = self.risk_aversion * ops.broadcast(risk.factor_weights, prefix + (rank,))
        diagonal = (
            self.risk_aversion * ops.broadcast(risk.diagonal, shape) + self.ridge + self.turnover
        )
        budget = ops.broadcast(budget, prefix)
        caps = ops.check(caps, caps >= 0, "Position caps must be nonnegative.")
        fees = ops.check(fees, fees >= 0, "Transaction-cost rates must be nonnegative.")
        previous = ops.check(
            previous, previous >= 0, "Long-only previous weights must be nonnegative."
        )
        eps = np.finfo(np.float64 if "64" in str(mean.dtype) else np.float32).eps
        tolerance = max(self.tolerance, 64 * eps)
        capacity = ops.sum(caps)
        budget = ops.check(
            budget,
            (budget >= 0) & (budget <= capacity + 8 * eps * max(1, n)),
            "Allocation budget exceeds available position capacity.",
        )
        budget = xp.minimum(budget, capacity)
        linear = mean + self.turnover * previous
        # No optimizer-iteration tape is retained. The returned gradient comes
        # from the refined KKT equations, including the no-trade facets.
        stopped = tuple(
            ops.stop(v) for v in (diagonal, factors, mass, linear, previous, caps, fees, budget)
        )

        def cold():
            approximate = self._identify(*stopped, ops)
            return self._face(
                approximate,
                diagonal,
                factors,
                mass,
                linear,
                previous,
                caps,
                fees,
                budget,
                ops,
                tolerance,
                eps,
            )

        compiling = ops.name == "torch" and _torch_compiling()
        if warm_start is None or compiling:
            candidate, residual, converged = cold()
        else:
            seed = ops.stop(ops.broadcast(ops.real(warm_start), shape))
            seed = _prox(
                seed,
                ops.clip(stopped[4], 0, stopped[5]),
                stopped[5],
                xp.zeros_like(stopped[6]),
                stopped[7],
                ops,
            )
            fast = self._face(
                seed,
                diagonal,
                factors,
                mass,
                linear,
                previous,
                caps,
                fees,
                budget,
                ops,
                tolerance,
                eps,
            )
            if ops.name == "jax":
                import jax

                if isinstance(fast[2], jax.core.Tracer):
                    candidate, residual, converged = jax.lax.cond(
                        xp.all(fast[2]), lambda: fast, cold
                    )
                else:
                    candidate, residual, converged = fast if bool(xp.all(fast[2])) else cold()
            else:
                candidate, residual, converged = fast if bool(xp.all(fast[2])) else cold()
        candidate = ops.check(
            candidate,
            converged[..., None],
            "Allocation failed its feasibility/KKT tolerance; "
            "increase iterations, ridge or numeric precision.",
            resource=True,
        )
        smooth = matmat(diagonal, factors, mass, candidate[..., None], ops)[..., 0]
        objective = 0.5 * ops.sum(candidate * smooth) - ops.sum(linear * candidate)
        objective += 0.5 * self.turnover * ops.sum(previous * previous)
        objective += ops.sum(fees * xp.abs(candidate - previous))
        return AllocationResult(candidate, objective, residual, converged)

    def _face(
        self,
        approximate,
        diagonal,
        factors,
        mass,
        linear,
        previous,
        caps,
        fees,
        budget,
        ops,
        tolerance,
        eps,
    ):
        xp = ops.xp
        center = ops.clip(previous, 0, caps)
        threshold = xp.minimum(ops.array(max(1e-8, 16 * eps)), caps * 0.25)
        lower = approximate <= threshold
        upper = (~lower) & ((caps - approximate) <= threshold)
        no_trade = (
            (~lower & ~upper) & (fees > 0) & (xp.abs(approximate - previous) <= max(1e-8, 16 * eps))
        )
        free = ~(lower | upper | no_trade)
        free = ops.stop(free.to(linear.dtype) if ops.name == "torch" else free.astype(linear.dtype))
        fixed = xp.where(upper, caps, xp.where(no_trade, previous, 0))
        sign = ops.stop(xp.sign(approximate - center))
        masked_factors = factors * free[..., None, :]
        masked_diagonal = diagonal * free + (1 - free)
        fixed_product = matmat(diagonal, factors, mass, fixed[..., None], ops)[..., 0]
        rhs = free * (linear - fees * sign - fixed_product)
        combined = ops.stack((rhs, free))
        solution = linear_solve(
            masked_diagonal,
            masked_factors,
            mass,
            combined,
            self.cg_iterations,
            min(1e-10, self.tolerance * 0.01),
            ops,
        )
        y, z = solution[..., 0], solution[..., 1]
        denominator = ops.sum(z)
        multiplier = (ops.sum(y) - budget + ops.sum(fixed)) / xp.where(
            denominator > 0, denominator, 1
        )
        candidate = fixed + y - multiplier[..., None] * z
        gradient = matmat(diagonal, factors, mass, candidate[..., None], ops)[..., 0] - linear
        projected = _prox(candidate - gradient, center, caps, fees, budget, ops)
        optimality = ops.max(xp.abs(candidate - projected))
        feasibility = xp.maximum(
            xp.abs(ops.sum(candidate) - budget),
            xp.maximum(ops.max(-candidate), ops.max(candidate - caps)),
        )
        residual = xp.maximum(optimality, feasibility)
        solved_residual = ops.max(
            xp.abs(matmat(masked_diagonal, masked_factors, mass, solution, ops) - combined),
            axis=(-1, -2),
        )
        reference = ops.maximum(ops.max(xp.abs(combined), axis=(-1, -2)), 1)
        converged = (
            (residual <= tolerance)
            & (solved_residual <= tolerance * reference)
            & xp.isfinite(residual)
        )
        return candidate, residual, converged

    def _identify(self, diagonal, factors, mass, linear, previous, caps, fees, budget, ops):
        xp = ops.xp
        if factors.shape[-2]:
            # This square root is confined to stopped forward data. The
            # differentiated risk and linear solve retain separate masses,
            # including finite probability derivatives at zero scenario mass.
            weighted = factors * xp.sqrt(mass)[..., :, None]
            gram = (
                ops.matmul(weighted, weighted.swapaxes(-1, -2))
                if factors.shape[-2] <= factors.shape[-1]
                else ops.matmul(weighted.swapaxes(-1, -2), weighted)
            )
            bound = xp.maximum(xp.linalg.eigvalsh(gram)[..., -1], ops.array(0))
        else:
            bound = ops.array(0)
        lipschitz = ops.max(diagonal) + bound
        step_size = 1 / lipschitz
        center = ops.clip(previous, 0, caps)
        initial = _prox(previous, center, caps, xp.zeros_like(fees), budget, ops)
        initial_state = (initial, initial, ops.array(1), xp.full_like(budget, float("inf")))
        eps = np.finfo(np.float64 if "64" in str(linear.dtype) else np.float32).eps
        threshold = max(1e-10, 8 * eps)

        def advance(state):
            old, accelerated, momentum, _ = state
            gradient = matmat(diagonal, factors, mass, accelerated[..., None], ops)[..., 0] - linear
            new = _prox(
                accelerated - step_size[..., None] * gradient,
                center,
                caps,
                step_size[..., None] * fees,
                budget,
                ops,
            )
            next_momentum = (1 + xp.sqrt(1 + 4 * momentum * momentum)) * 0.5
            extrapolated = new + ((momentum - 1) / next_momentum) * (new - old)
            difference = ops.max(xp.abs(new - old))
            return new, extrapolated, next_momentum, difference

        if ops.name == "jax":
            import jax

            def condition(c):
                return (c[0] < self.max_iterations) & xp.any(c[1][3] > threshold)

            return jax.lax.while_loop(
                condition, lambda c: (c[0] + 1, advance(c[1])), (0, initial_state)
            )[1][0]
        if ops.name == "torch" and hasattr(xp, "while_loop") and _torch_compiling():

            def condition(index, old, accelerated, momentum, difference):
                return (index < self.max_iterations) & xp.any(difference > threshold)

            def body(index, old, accelerated, momentum, difference):
                return (index + 1, *advance((old, accelerated, momentum, difference)))

            counter = xp.zeros((), dtype=xp.int64, device=linear.device)
            carried = (counter, initial_state[0], initial_state[1].clone(), *initial_state[2:])
            return xp.while_loop(condition, body, carried)[1]
        state = initial_state
        for iteration in range(self.max_iterations):
            state = advance(state)
            if iteration % 16 == 15 and not (ops.name == "torch" and _torch_compiling()):
                if bool(xp.all(state[3] <= threshold)):
                    break
        return state[0]
