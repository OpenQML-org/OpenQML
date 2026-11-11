"""Causal quantum/tensor forecasts and conditional Born scenario distributions."""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from ..compute import get_compute_kernel, memory_budget
from ..errors import ValidationError
from ..ir import integer
from ..ir.parameters import real
from ..qml.architecture import HybridArchitecture, QuantumCellSpec
from ..qml.operators import OperatorSpec
from ._array import Ops, admission, rows
from .risk import factor_risk, quantile_loss


class MarketForecast(NamedTuple):
    quantiles: object
    median: object


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
        admission(
            4 * (s * dimension + s * s),
            self.max_memory_mb,
            work=4 * s * s * dimension,
            max_work=self.max_work,
        )
        x = x.copy()
        flat = x.reshape((s, dimension))
        square = np.sum(flat * flat, axis=-1)
        distance = np.sqrt(
            np.maximum(
                square[:, None] + square[None, :] - 2 * get_compute_kernel().matmul(flat, flat.T), 0
            )
        )
        np.fill_diagonal(distance, 0)
        for value in (x, flat, distance):
            value.setflags(write=False)
        object.__setattr__(self, "scenarios", x)
        object.__setattr__(self, "_flat", flat)
        object.__setattr__(self, "_distances", distance)

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
        prefix = np.broadcast_shapes(p.shape[:-1], target.shape[:-1])
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
        return np.prod(1 + self.scenarios, axis=1) - 1

    def risk(self, probabilities, *, step=None, shrinkage=0.1, variance_floor=1e-8):
        return factor_risk(
            self.returns(step=step),
            probabilities,
            shrinkage=shrinkage,
            variance_floor=variance_floor,
            max_memory_mb=self.max_memory_mb,
        )

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
