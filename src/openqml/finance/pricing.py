"""Exact bounded-payoff encoding for coherent pricing and risk expectations."""

from dataclasses import dataclass, field

import numpy as np

from ..compute import memory_budget
from ..errors import ResourceError, ValidationError
from ..ir import Circuit, CircuitBuilder, integer
from ..ir.parameters import real
from ..sim.amplitude import AmplitudeEstimationPlan, AmplitudeOracle
from ._array import Ops, admission
from ._results import PayoffEstimate
from .quantum import _fixed


@dataclass(frozen=True)
class PayoffOraclePlan:
    """Encode a fixed signed payoff table under a coherent scenario preparation.

    A known-mean control variate can reduce the residual encoding range. The
    affine arcsin encoding is exact; table compilation still costs O(S log S).
    A learned physical measure is not automatically a risk-neutral measure.
    """

    preparation: Circuit
    payoffs: object
    bounds: tuple | None = None
    discount: float = 1
    control_values: object = None
    control_expectation: float | None = None
    control_coefficient: float = 1
    max_operations: int = 100_000
    max_memory_mb: float = 256
    _oracle: object = field(default=None, init=False, repr=False, compare=False)
    offset: float = field(init=False)
    scale: float = field(init=False)

    def __post_init__(self):
        if (
            not isinstance(self.preparation, Circuit)
            or self.preparation.has_effects
            or self.preparation.parameters
            or self.preparation.num_clbits
        ):
            raise ValidationError("Payoff preparation must be an explicitly bound unitary Circuit.")
        values = _fixed(self.payoffs)
        if (
            values.ndim != 1
            or not len(values)
            or len(values) & (len(values) - 1)
            or len(values).bit_length() - 1 != self.preparation.num_qubits
        ):
            raise ValidationError("Payoff entries must match every q0-MSB scenario basis state.")
        memory, discount = memory_budget(self.max_memory_mb), real(self.discount)
        limit = integer(self.max_operations, "max_operations", 1)
        admission(12 * len(values), memory, work=8 * len(values) * len(values).bit_length())
        if discount <= 0:
            raise ValidationError("Discount must be positive.")
        coefficient, known, residual = real(self.control_coefficient), 0.0, values
        if (self.control_values is None) != (self.control_expectation is None):
            raise ValidationError("Supply both control values and their known expectation.")
        if self.control_values is not None:
            control = _fixed(self.control_values, values.shape)
            expectation = real(self.control_expectation)
            residual = values - coefficient * control
            known = coefficient * expectation
            object.__setattr__(self, "control_values", control)
            object.__setattr__(self, "control_expectation", expectation)
        if not np.all(np.isfinite(residual)) or not np.isfinite(known):
            raise ValidationError("Control-adjusted payoffs must remain finite.")
        bounds = (
            (float(residual.min()), float(residual.max()))
            if self.bounds is None
            else tuple(real(v) for v in self.bounds)
        )
        if len(bounds) != 2 or bounds[0] > residual.min() or bounds[1] < residual.max():
            raise ValidationError("Payoff bounds must enclose every control-adjusted payoff.")
        lower, upper = bounds
        scale = upper - lower
        if not np.isfinite(scale) or not np.isfinite(lower + known):
            raise ValidationError("Payoff encoding range must remain finite.")
        if np.all(residual == residual[0]):
            lower, scale = float(residual[0]), 0.0
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "payoffs", values)
        object.__setattr__(self, "discount", discount)
        object.__setattr__(self, "control_coefficient", coefficient)
        object.__setattr__(self, "max_operations", limit)
        object.__setattr__(self, "max_memory_mb", memory)
        object.__setattr__(self, "offset", lower + known)
        object.__setattr__(self, "scale", scale)

    @property
    def oracle(self):
        if self.scale == 0:
            return None
        if self._oracle is not None:
            return self._oracle
        # FWHT coefficients followed in cyclic Gray-code order synthesize a
        # multiplexed RY using one extra payoff qubit, without dense matrices.
        source, good = self.preparation, self.preparation.num_qubits
        values = self.payoffs
        residual = (
            values
            if self.control_values is None
            else values - self.control_coefficient * self.control_values
        )
        admission(
            32 * len(values), self.max_memory_mb, work=8 * len(values) * len(values).bit_length()
        )
        if 2 * len(values) + len(source.operations) > self.max_operations:
            raise ResourceError("Payoff table exceeds the circuit operation budget.")
        angles = 2 * np.arcsin(np.sqrt((residual - self.bounds[0]) / self.scale))
        coefficients = angles.copy()
        stride = 1
        while stride < len(values):
            blocks = coefficients.reshape((-1, 2, stride))
            a, b = blocks[:, 0].copy(), blocks[:, 1].copy()
            blocks[:, 0], blocks[:, 1] = a + b, a - b
            stride *= 2
        coefficients /= len(values)
        builder = CircuitBuilder(good + 1, global_phase=source.global_phase).extend(
            source.operations
        )
        for i in range(len(values)):
            gray, next_index = i ^ (i >> 1), (i + 1) % len(values)
            next_gray = next_index ^ (next_index >> 1)
            builder.ry(float(coefficients[gray]), good)
            builder.cx(good - (gray ^ next_gray).bit_length(), good)
        result = AmplitudeOracle(builder.build(), good, self.max_operations, self.max_memory_mb)
        object.__setattr__(self, "_oracle", result)
        return result

    def classical_value(self, probabilities):
        """Use exact finite-scenario summation when probabilities are available."""
        ops = Ops(probabilities)
        p = ops.probabilities(probabilities, len(self.payoffs))
        return self.discount * ops.sum(p * ops.array(self.payoffs))

    def estimate(self, plan=None, *, backend="auto", **options):
        from ..sim.adaptive_amplitude import AdaptiveAmplitudeEstimationPlan

        if plan is None:
            plan = AmplitudeEstimationPlan.geometric()
        if not isinstance(plan, (AmplitudeEstimationPlan, AdaptiveAmplitudeEstimationPlan)):
            raise ValidationError("Supply a fixed or adaptive amplitude estimation plan.")
        if self.oracle is None:
            value = self.discount * self.offset
            return PayoffEstimate(value, ((value, value),), plan.confidence, None)
        result = plan.run(self.oracle, backend=backend, **options)
        intervals = tuple(
            (
                self.discount * (self.offset + self.scale * a),
                self.discount * (self.offset + self.scale * b),
            )
            for a, b in result.intervals
        )
        return PayoffEstimate(
            self.discount * (self.offset + self.scale * result.probability),
            intervals,
            result.confidence,
            result,
        )
