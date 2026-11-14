"""Functional distance-aware Quantum Natural Muon with portable native state."""

import hashlib
import json
from dataclasses import dataclass, field
from typing import NamedTuple

from ..compute._norms import stop
from ..compute.kernel import _array_backend, compute_operation, get_compute_kernel
from ..errors import ResourceError, ValidationError
from ..ir.core import integer
from ..ir.parameters import real
from .geometry import QuantumMetricPlan
from .native_geometry import _inputs, _natural_solve, _polar_mix, _spectral_shapes


class DistanceAwareState(NamedTuple):
    initial_parameters: object
    momentum: object
    max_distance: object
    steps: object
    signature: object


class DistanceAwareResult(NamedTuple):
    step: object
    state: DistanceAwareState
    natural: object
    radius: object
    alignment: object
    restarted: object


@dataclass(frozen=True, eq=False)
class DistanceAwareQNM:
    """QFIM solve -> natural momentum -> active polar mixing -> distance radius.

    Uses F=4*Re(QGT). Selected coordinates are shared scalar bindings; auxiliary
    circuit inputs may carry the existing weighted sample axis. State is an
    explicit Torch/JAX tree, detached between steps and checked against this
    circuit, parameter order and optimizer configuration. No parameters mutate.
    """

    plan: QuantumMetricPlan
    block_shapes: tuple
    damping: float = 0.01
    momentum: float = 0.9
    mixing: float = 0.1
    learning_rate: float = 0.01
    initial_radius: float = 0.01
    distance_growth: float = 0.1
    max_step_norm: float = 0.1
    normalization_epsilon: float = 1e-8
    tolerance: float = 1e-6
    max_iterations: int = 32
    alignment_guard: bool = True
    sample_chunk_size: int | None = None
    _signature: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if self.sample_chunk_size is not None:
            object.__setattr__(
                self, "sample_chunk_size", integer(self.sample_chunk_size, "sample_chunk_size", 1)
            )
        if not isinstance(self.plan, QuantumMetricPlan) or not self.plan.parameters:
            raise ValidationError("Distance-aware QNM needs a nonempty QuantumMetricPlan.")
        shapes = _spectral_shapes(self.block_shapes, len(self.plan.parameters))
        object.__setattr__(self, "block_shapes", shapes)
        settings = {}
        for name in (
            "damping",
            "momentum",
            "mixing",
            "learning_rate",
            "initial_radius",
            "distance_growth",
            "max_step_norm",
            "normalization_epsilon",
            "tolerance",
        ):
            value = real(getattr(self, name))
            settings[name] = value
            object.__setattr__(self, name, value)
        if not 0 <= self.momentum < 1 or not 0 <= self.mixing <= 1 or self.distance_growth < 0:
            raise ValidationError(
                "Use momentum in [0,1), mixing in [0,1] and nonnegative distance growth."
            )
        if any(
            settings[n] <= 0
            for n in (
                "damping",
                "learning_rate",
                "initial_radius",
                "max_step_norm",
                "normalization_epsilon",
                "tolerance",
            )
        ):
            raise ValidationError(
                "QNM damping, rates, radii, epsilon and tolerance must be positive."
            )
        object.__setattr__(
            self, "max_iterations", integer(self.max_iterations, "max_iterations", 1)
        )
        if type(self.alignment_guard) is not bool:
            raise ValidationError("alignment_guard must be a boolean.")
        payload = json.dumps(
            [
                "openqml.distance-aware-qnm.v1",
                self.plan.parameters,
                shapes,
                settings,
                int(self.max_iterations),
                self.alignment_guard,
            ],
            sort_keys=True,
        ).encode()
        digest = hashlib.sha256(payload)
        digest.update(
            self.plan.circuit.to_json(
                indent=None, max_bytes=int(self.plan.max_memory_mb * 2**20)
            ).encode()
        )
        value = digest.digest()
        object.__setattr__(
            self,
            "_signature",
            tuple(int.from_bytes(value[i : i + 4], "little", signed=True) for i in range(0, 32, 4)),
        )

    @compute_operation
    def step(
        self,
        bindings,
        gradient,
        *,
        state=None,
        backend="torch",
        sample_weights=None,
        learning_rate=None,
        metric_reducer=None,
    ):
        """Return a displacement to subtract and the next immutable native state.

        The nested residual certifies (F+damping I)u=g, before momentum and polar
        mixing. The optional safeguard restarts momentum from u if the proposed
        direction lacks positive observed gradient alignment. It is not a line
        search or a loss-decrease certificate.
        """
        ops, local, gradient, mass = _inputs(
            self.plan, bindings, gradient, backend, sample_weights, self.sample_chunk_size
        )
        if any(local[name].ndim for name in self.plan.parameters):
            raise ValidationError(
                "Stateful QNM selected coordinates must be shared scalar bindings."
            )
        xp = ops.xp
        rate = ops.array(
            self.learning_rate if learning_rate is None else learning_rate, real_value=True
        )
        if rate.ndim:
            raise ValidationError("The learning-rate override must be a scalar.")
        rate = ops.check(
            rate,
            xp.isfinite(rate) & (rate >= 0),
            "Learning-rate overrides must be finite and nonnegative.",
        )
        if metric_reducer is not None and not callable(metric_reducer):
            raise ValidationError("metric_reducer must be callable or None.")
        coordinates = xp.stack(tuple(local[name] for name in self.plan.parameters))
        get_compute_kernel().admit(
            64 * len(self.plan.parameters) + 8 * sum(min(s) ** 2 for s in self.block_shapes),
            self.plan.max_memory_mb,
        )
        if backend == "torch":
            signature = xp.tensor(self._signature, dtype=xp.int32, device=gradient.device)
            counter = xp.zeros((), dtype=xp.int32, device=gradient.device)
        else:
            signature = xp.asarray(self._signature, dtype=xp.int32)
            counter = xp.zeros((), dtype=xp.int32)
        if state is None:
            state = DistanceAwareState(
                stop(coordinates, backend),
                gradient * 0,
                ops.array(0, real_value=True),
                counter,
                signature,
            )
        state = self._check_state(state, gradient, signature, ops, backend)
        natural = _natural_solve(
            self.plan,
            local,
            gradient,
            mass,
            ops,
            backend,
            metric_scale=4,
            metric_reducer=metric_reducer,
            sample_chunk_size=self.sample_chunk_size,
            damping=self.damping,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
        )
        momentum = self.momentum * state.momentum + (1 - self.momentum) * natural.direction
        direction = _polar_mix(momentum, self.block_shapes, self.mixing, ops, backend)
        alignment = (gradient * direction).sum()
        restart = (~(xp.isfinite(alignment) & (alignment > 0))) & self.alignment_guard
        momentum = xp.where(restart, natural.direction, momentum)
        direction = xp.where(restart, natural.direction, direction)
        distance = xp.maximum(state.max_distance, ops.norm(coordinates - state.initial_parameters))
        radius = xp.minimum(
            rate * ops.norm(momentum),
            xp.minimum(
                self.initial_radius + self.distance_growth * distance,
                ops.array(self.max_step_norm, real_value=True),
            ),
        )
        displacement = radius * (direction / (ops.norm(direction) + self.normalization_epsilon))
        displacement = ops.check(
            displacement, xp.isfinite(displacement), "QNM displacement exceeds finite precision."
        )
        momentum = ops.check(momentum, xp.isfinite(momentum), "QNM momentum must remain finite.")
        distance = ops.check(
            distance, xp.isfinite(distance), "QNM trajectory distance must remain finite."
        )
        next_state = DistanceAwareState(
            stop(state.initial_parameters, backend),
            stop(momentum, backend),
            stop(distance, backend),
            state.steps + 1,
            signature,
        )
        return DistanceAwareResult(
            displacement, next_state, natural, radius, (gradient * displacement).sum(), restart
        )

    @staticmethod
    def _check_state(state, gradient, signature, ops, backend):
        if not isinstance(state, DistanceAwareState):
            raise ValidationError("Provide DistanceAwareState from the same optimizer.")
        shapes = (gradient.shape, gradient.shape, (), (), signature.shape)
        valid = signature[0] == signature[0]
        for i, (value, shape) in enumerate(zip(state, shapes, strict=True)):
            dtype = gradient.dtype if i < 3 else ops.xp.int32
            if _array_backend(value) != backend or value.shape != shape or value.dtype != dtype:
                raise ValidationError(
                    "QNM state shapes, framework and dtypes must match the current optimizer."
                )
            if backend == "torch" and value.device != gradient.device:
                raise ValidationError(
                    "Move all QNM state tensors explicitly to the selected device."
                )
            valid = valid & ops.xp.all(ops.xp.isfinite(value))
        valid = (
            valid
            & ops.xp.all(state.signature == signature)
            & (state.max_distance >= 0)
            & (state.steps >= 0)
        )
        initial = ops.check(
            state.initial_parameters,
            valid,
            "QNM state must be finite, have valid counters and match the circuit, "
            "parameter order and optimizer configuration.",
        )
        initial = ops.check(
            initial,
            state.steps < 2_147_483_647,
            "QNM step counter is exhausted.",
            error=ResourceError,
        )
        # Construct directly: older Dynamo treats NamedTuple._replace as a
        # skipped stdlib function even though all of its fields are native.
        return DistanceAwareState(
            initial, state.momentum, state.max_distance, state.steps, state.signature
        )
