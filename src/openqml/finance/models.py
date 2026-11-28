"""Causal quantum/tensor forecasts and conditional Born scenario distributions."""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from ..compute.kernel import memory_budget
from ..errors import ValidationError
from ..ir.core import integer
from ..ir.parameters import real
from ..qml.architecture import HybridArchitecture, QuantumCellSpec
from ..qml.operators import OperatorSpec
from ._array import Ops, admission, rows
from .risk import factor_risk, quantile_loss


class MarketForecast(NamedTuple):
    quantiles: object
    median: object


class ScenarioDecisionResult(NamedTuple):
    decision: object
    loss: object
    calibration: object
    realized_return: object


@dataclass(frozen=True)
class MarketForecastPlan:
    input_features: int
    num_assets: int
    quantiles: tuple = (0.05, 0.25, 0.5, 0.75, 0.95)
    return_scale: float = 0.01
    depth: int = 3
    hidden_features: int = 24
    model: str = "recurrent"
    memory_qubits: int = 1
    environment_qubits: int = 1
    quantum_depth: int = 1
    bond_dimension: int = 4
    projection_policy: str = "auto"
    max_memory_mb: float = 256

    def __post_init__(self):
        for name in (
            "input_features",
            "num_assets",
            "depth",
            "hidden_features",
            "memory_qubits",
            "environment_qubits",
            "quantum_depth",
            "bond_dimension",
        ):
            object.__setattr__(self, name, integer(getattr(self, name), name, 1))
        levels = tuple(real(q) for q in self.quantiles)
        if (
            not levels
            or any(not 0 < q < 1 for q in levels)
            or any(a >= b for a, b in zip(levels, levels[1:], strict=False))
            or 0.5 not in levels
        ):
            raise ValidationError("Quantile levels must increase in (0,1) and include 0.5.")
        if self.model not in {"recurrent", "state-attention", "tensor"}:
            raise ValidationError("Forecast model must be recurrent, state-attention or tensor.")
        object.__setattr__(self, "quantiles", levels)
        object.__setattr__(self, "return_scale", real(self.return_scale))
        object.__setattr__(self, "max_memory_mb", memory_budget(self.max_memory_mb))
        if self.return_scale <= 0 or self.max_memory_mb <= 0:
            raise ValidationError("Return scale and memory budget must be positive.")

    def architecture(self, **options):
        shared = dict(
            hidden_features=self.hidden_features,
            depth=self.depth,
            projection_policy=self.projection_policy,
            bond_dimension=self.bond_dimension,
            max_memory_mb=self.max_memory_mb,
        )
        shared.update(options)
        width = self.num_assets * len(self.quantiles)
        if self.model == "recurrent":
            return HybridArchitecture.recurrent(
                self.input_features,
                width,
                memory_qubits=self.memory_qubits,
                environment_qubits=self.environment_qubits,
                quantum_depth=self.quantum_depth,
                **shared,
            )
        if self.model == "state-attention":
            return HybridArchitecture.transformer(
                self.input_features,
                width,
                attention="state",
                causal=True,
                quantum=QuantumCellSpec(kind="qcnn", num_qubits=2, depth=self.quantum_depth),
                **shared,
            )
        shared.setdefault("native_execution", "tensor-network")
        return HybridArchitecture(
            self.input_features,
            width,
            quantum=QuantumCellSpec(num_qubits=4, depth=self.quantum_depth),
            **shared,
        )

    def build(self, framework="torch", *, architecture_options=None, **options):
        return self.architecture(**(architecture_options or {})).build(framework, **options)

    def decode(self, outputs):
        ops = Ops(outputs)
        x = ops.real(outputs)
        count = len(self.quantiles)
        if x.shape[-1:] != (self.num_assets * count,):
            raise ValidationError("Forecast output width differs from the plan.")
        x = x.reshape(x.shape[:-1] + (self.num_assets, count))
        middle = self.quantiles.index(0.5)
        if count == 1:
            return MarketForecast(x * self.return_scale, x[..., 0] * self.return_scale)
        gaps = ops.xp.maximum(x[..., 1:], ops.array(0)) + ops.xp.log1p(
            ops.xp.exp(-ops.xp.abs(x[..., 1:]))
        )
        gaps = gaps * ops.array(np.diff(self.quantiles))
        offsets = ops.cat((ops.xp.zeros_like(x[..., :1]), ops.cumsum(gaps)), axis=-1)
        median = x[..., 0] * self.return_scale
        values = median[..., None] + self.return_scale * (
            offsets - offsets[..., middle : middle + 1]
        )
        return MarketForecast(values, median)

    def loss(self, outputs, targets, *, mask=None):
        values = quantile_loss(self.decode(outputs).quantiles, targets, self.quantiles, mask=mask)
        if values.ndim < 2:
            return values
        ops = Ops(values)
        return ops.sum(values, axis=tuple(range(1, values.ndim))) / rows(values.shape[1:])


@dataclass(frozen=True)
class ScenarioPlan:
    """Finite user-supplied joint return/path atoms with trainable Born weights.

    Atoms have shape [S,N] or [S,H,N], S a power of two. Fit/select them on
    training data only. A finite support is not an unrestricted market process.
    """

    scenarios: object
    max_memory_mb: float = 256
    max_work: int = 1_000_000_000
    _flat: object = field(init=False, repr=False)
    _distances: object = field(init=False, repr=False)
    _order: object = field(init=False, repr=False)

    def __post_init__(self):
        x = np.asarray(self.scenarios, dtype=float)
        if (
            x.ndim not in (2, 3)
            or min(x.shape) < 1
            or x.shape[0] < 2
            or x.shape[0] & (x.shape[0] - 1)
            or not np.all(np.isfinite(x))
            or np.any(x < -1)
        ):
            raise ValidationError(
                "Provide finite arithmetic-return atoms [2**k,N] or [2**k,H,N], "
                "bounded below by -1."
            )
        object.__setattr__(self, "max_memory_mb", memory_budget(self.max_memory_mb))
        object.__setattr__(self, "max_work", integer(self.max_work, "max_work", 1))
        s, dimension = x.shape[0], rows(x.shape[1:])
        elements = 6 * s if dimension == 1 else 4 * (s * dimension + s * s)
        work = s * s.bit_length() if dimension == 1 else 4 * s * s * dimension
        admission(elements, self.max_memory_mb, work=work, max_work=self.max_work)
        x = x.copy()
        flat = x.reshape((s, dimension))
        order = distance = None
        if dimension == 1:
            order = np.argsort(flat[:, 0], kind="stable")
            flat = flat[order]
            order.setflags(write=False)
        else:
            # Keep the small separation of nearly coincident atoms. Each
            # triangular row avoids an [S,S,D] subtraction workspace.
            distance = np.zeros((s, s))
            for i in range(s - 1):
                row = np.hypot.reduce(flat[i + 1 :] - flat[i], axis=-1)
                distance[i + 1 :, i] = distance[i, i + 1 :] = row
            distance.setflags(write=False)
        x.setflags(write=False)
        flat.setflags(write=False)
        object.__setattr__(self, "scenarios", x)
        object.__setattr__(self, "_flat", flat)
        object.__setattr__(self, "_distances", distance)
        object.__setattr__(self, "_order", order)

    @property
    def num_assets(self):
        return self.scenarios.shape[-1]

    @property
    def num_scenarios(self):
        return self.scenarios.shape[0]

    def architecture(
        self,
        input_features,
        *,
        depth=2,
        memory_qubits=1,
        environment_qubits=1,
        quantum_depth=1,
        readout_depth=2,
        **options,
    ):
        memory = QuantumCellSpec(
            kind="memory",
            num_qubits=memory_qubits + environment_qubits,
            memory_qubits=memory_qubits,
            depth=quantum_depth,
        )
        options.setdefault("max_memory_mb", self.max_memory_mb)
        return HybridArchitecture.generative(
            input_features,
            self.num_scenarios.bit_length() - 1,
            depth=depth,
            quantum=memory,
            operators=OperatorSpec.memory(),
            readout_depth=readout_depth,
            **options,
        )

    def probabilities(self, probabilities):
        ops = Ops(probabilities)
        return ops.probabilities(probabilities, self.num_scenarios)

    def energy_score(self, probabilities, observations):
        ops = Ops(probabilities, observations)
        p = ops.probabilities(probabilities, self.num_scenarios)
        target = ops.real(observations)
        if target.shape[-(self.scenarios.ndim - 1) :] != self.scenarios.shape[1:]:
            raise ValidationError("Observed return/path shape differs from scenario atoms.")
        target = target.reshape(
            target.shape[: -(self.scenarios.ndim - 1)] + (self._flat.shape[-1],)
        )
        prefix = ops.broadcast_shape(p.shape[:-1], target.shape[:-1])
        if self._order is not None:
            # Integrate the empirical CDF against the observation step.
            # Accumulate each tail directly to preserve rare scenario mass.
            elements = rows(prefix) * self.num_scenarios
            admission(14 * elements, self.max_memory_mb, work=16 * elements, max_work=self.max_work)
            order = ops.kernel.asarray(self._order, xp=ops.xp, backend=ops.name, like=ops.like)
            mass = p[..., order]
            cdf = ops.cumsum(mass)[..., :-1]
            survival = ops.xp.flip(ops.cumsum(ops.xp.flip(mass, (-1,))), (-1,))[..., 1:]
            atoms = ops.array(self._flat[:, 0])
            left, right = atoms[:-1], atoms[1:]
            split = ops.xp.minimum(ops.xp.maximum(target, left), right)
            intervals = (split - left) * cdf * cdf + (right - split) * survival * survival
            outside = ops.maximum(atoms[0] - target[..., 0], 0) + ops.maximum(
                target[..., 0] - atoms[-1], 0
            )
            return outside + ops.sum(ops.xp.where(left < right, intervals, 0))
        admission(6 * rows(prefix) * self._flat.size + 2 * self._distances.size, self.max_memory_mb)
        distances = ops.norm(ops.array(self._flat) - target[..., None, :])
        first = ops.sum(p * distances)
        second = ops.sum(p * (ops.matmul(p, ops.array(self._distances))))
        return first - 0.5 * second

    def returns(self, *, step=None):
        if self.scenarios.ndim == 2:
            if step not in (None, 0):
                raise ValidationError("Single-step scenarios have only step zero.")
            return self.scenarios
        if step is not None:
            index = integer(step, "step", 0)
            if index >= self.scenarios.shape[1]:
                raise ValidationError("Scenario step exceeds the path horizon.")
            return self.scenarios[:, index]
        total_loss = self.scenarios == -1
        log_growth = np.log1p(np.where(total_loss, 0, self.scenarios)).sum(axis=1)
        return np.expm1(np.where(total_loss.any(axis=1), -np.inf, log_growth))

    def risk(self, probabilities, *, step=None, shrinkage=0.1, variance_floor=1e-8):
        return factor_risk(
            self.returns(step=step),
            probabilities,
            shrinkage=shrinkage,
            variance_floor=variance_floor,
            max_memory_mb=self.max_memory_mb,
        )

    def entropic_attribution(self, probabilities, weights, *, step=None, confidence=0.95):
        """Euler EVaR contributions and excess over nominal expected-loss contributions."""
        from .entropic import EntropicAttribution, entropic_value_at_risk

        ops = Ops(weights, probabilities)
        returns, w = ops.real(self.returns(step=step)), ops.real(weights)
        if w.shape[-1:] != returns.shape[-1:]:
            raise ValidationError("Portfolio weights must match the scenario asset axis.")
        p = ops.probabilities(probabilities, self.num_scenarios)
        prefix = ops.broadcast_shape(w.shape[:-1], p.shape[:-1])
        s, n = returns.shape
        admission(
            s * n + rows(prefix) * (8 * s + 8 * n),
            self.max_memory_mb,
            work=rows(prefix) * s * (64 + 3 * n),
            max_work=self.max_work,
        )
        losses = -ops.einsum("sn,...n->...s", returns, w)
        risk = entropic_value_at_risk(
            losses, probabilities=p, confidence=confidence, max_memory_mb=self.max_memory_mb
        )
        marginal = -ops.matmul(risk.worst_probabilities, returns)
        nominal = -ops.matmul(p, returns)
        return EntropicAttribution(risk, marginal, w * marginal, w * (marginal - nominal))

    def tree(self, node_ids, **options):
        """Build a causal tree over this plan's multi-period return atoms."""
        from .scenario_tree import ScenarioTreePlan

        options.setdefault("max_memory_mb", self.max_memory_mb)
        options.setdefault("max_work", self.max_work)
        return ScenarioTreePlan(self.scenarios, node_ids, **options)

    def horizon_risk(self, probabilities, *, shrinkage=0.1, variance_floor=1e-8):
        """Per-period factor risk from one predicted distribution over whole paths."""
        if self.scenarios.ndim != 3:
            raise ValidationError("Horizon risk requires multi-period scenario paths.")
        ops = Ops(probabilities)
        p = ops.probabilities(probabilities, self.num_scenarios)
        return factor_risk(
            ops.array(self.scenarios).swapaxes(-3, -2),
            p[..., None, :],
            shrinkage=shrinkage,
            variance_floor=variance_floor,
            max_memory_mb=self.max_memory_mb,
        )

    def horizon_decision(
        self, probabilities, *, allocation=None, shrinkage=0.1, variance_floor=1e-8, **options
    ):
        """Differentiate a coupled open-loop horizon decision through Born path weights."""
        from ..errors import _require_instance
        from .multiperiod import MultiPeriodPlan

        if allocation is None:
            allocation = MultiPeriodPlan(max_memory_mb=self.max_memory_mb, max_work=self.max_work)
        _require_instance(allocation, MultiPeriodPlan, "allocation")
        risk = self.horizon_risk(probabilities, shrinkage=shrinkage, variance_floor=variance_floor)
        return allocation.solve(risk.mean, risk, **options)

    def tail_decision(self, probabilities, *, allocation=None, **options):
        """Optimize joint-path tail losses through a predictive Born distribution."""
        from ..errors import _require_instance
        from .tail import TailAllocationPlan

        if self.scenarios.ndim != 3:
            raise ValidationError("Path-tail decisions require multi-period scenario paths.")
        if allocation is None:
            allocation = TailAllocationPlan(
                max_memory_mb=self.max_memory_mb, max_work=self.max_work
            )
        _require_instance(allocation, TailAllocationPlan, "allocation")
        ops = Ops(probabilities)
        return allocation.solve(ops.array(self.scenarios), probabilities=probabilities, **options)

    def execution_decision(self, probabilities, proposal, *, step=None, **options):
        """Project a proposal using this Born distribution and after-cost risk.

        Path atoms without step mean a single trade held for the whole horizon.
        Supply the cash return for the same horizon. This submits no orders.
        """
        from .execution_risk import execution_risk_projection

        options.setdefault("max_memory_mb", self.max_memory_mb)
        return execution_risk_projection(
            proposal, self.returns(step=step), probabilities=probabilities, **options
        )

    def execution_loss(
        self,
        probabilities,
        proposal,
        observations,
        *,
        calibration_weight=1,
        decision_weight=0.1,
        **options,
    ):
        """Native proper-score plus observed after-cost loss for end-to-end training.

        Returns per-example losses for HybridTrainer/NetworkTrainer. Observed
        labels affect training loss only. Scenario risk uses the predictive
        distribution and supplied proposal. Path labels describe a buy-and-hold
        horizon, not a sequence of optimized future trades.
        """
        if options.get("step") is not None:
            raise ValidationError("execution_loss uses the full predictive and observed horizon.")
        calibration_weight, decision_weight = real(calibration_weight), real(decision_weight)
        if (
            min(calibration_weight, decision_weight) < 0
            or calibration_weight + decision_weight <= 0
        ):
            raise ValidationError("Use nonnegative decision-loss weights with positive total.")
        decision = self.execution_decision(probabilities, proposal, **options)
        ops = Ops(probabilities, observations, decision.positions, decision.cash, decision.costs)
        target = ops.real(observations)
        target = ops.check(target, target >= -1, "Observed arithmetic returns must be at least -1.")
        calibration = self.energy_score(ops.real(probabilities), target)
        if self.scenarios.ndim == 3:
            target = (
                (1 + target).prod(dim=-2) - 1
                if ops.name == "torch"
                else ops.xp.prod(1 + target, axis=-2) - 1
            )
        cash_return = ops.real(options.get("cash_return", 0))
        realized = (
            ops.sum(decision.positions * target) + cash_return * decision.cash - decision.costs
        )
        loss = calibration_weight * calibration - decision_weight * realized
        return ScenarioDecisionResult(decision, loss, calibration, realized)

    def expected_payoff(self, probabilities, payoffs, *, discount=1):
        ops = Ops(probabilities, payoffs)
        p = ops.probabilities(probabilities, self.num_scenarios)
        payoff = ops.real(payoffs)
        discount = real(discount)
        if payoff.shape[-1:] != (self.num_scenarios,) or discount <= 0:
            raise ValidationError(
                "Payoffs must have one value per scenario and discount be positive."
            )
        return discount * ops.sum(p * payoff)
