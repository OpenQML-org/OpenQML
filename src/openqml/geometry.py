"""Matrix-free pure-state quantum geometry using tangent and adjoint sweeps.

The Fubini--Study product is Re[J* (I-|psi><psi|) J v]. A directional
forward sweep and reverse contraction avoid storing all parameter tangents or
the quadratic metric. Natural gradients solve the damped system by linear CG.
This is classical state-vector computation, with exponential qubit cost.
Research background: https://arxiv.org/abs/1909.02108 and
https://arxiv.org/abs/2011.02991. This is not the nonlinear CQNG algorithm.
"""

from dataclasses import dataclass, field

import numpy as np

from ._gradient_common import _real_array
from .adjoint import AdjointPlan, _reverse, _tangent_forward
from .errors import ResourceError, ValidationError
from .ir import Circuit, integer
from .observables import PauliString
from .parameters import real


@dataclass(frozen=True)
class NaturalGradientResult:
    direction: np.ndarray
    parameters: tuple[str, ...]
    converged: bool
    iterations: int
    metric_products: int
    residual_norm: float
    damping: float
    reason: str


@dataclass(frozen=True, eq=False)
class QuantumMetricPlan:
    """Exact numeric pure-state metric products at one binding, on CPU.

    The metric is Re(QGT), not four-times QGT (quantum Fisher information).
    Symbolic angle chain rules include parameter reuse and nonlinear maps.
    Global phase directions cancel. Effects and hardware sampling are excluded.
    """

    circuit: object
    parameters: tuple | None = None
    max_qubits: int | None = None
    max_memory_mb: float = 256
    _plan: object = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.circuit, Circuit):
            raise ValidationError("Quantum geometry requires an OpenQML Circuit.")
        # Reuse the existing validation, derivative compilation and conservative
        # twelve-state working-budget admission, without retaining state values.
        plan = AdjointPlan(
            self.circuit,
            (PauliString("I" * self.circuit.num_qubits),),
            self.parameters,
            self.max_qubits,
            self.max_memory_mb,
        )
        object.__setattr__(self, "_plan", plan)
        object.__setattr__(self, "parameters", plan.parameters)

    def _bindings(self, bindings):
        columns, count = self._plan._bindings(bindings, None)
        if count != 1:
            raise ValidationError("Quantum metric products require one binding at a time.")
        return columns

    def _product(self, bindings, vector):
        if not self.parameters:
            return np.empty(0)
        state, tangent = _tangent_forward(
            self.circuit, bindings, self._plan._derivatives, vector[None, :]
        )
        overlap = np.sum(state.conj() * tangent, axis=-1)
        projected = tangent - state * overlap[:, None]
        return (
            0.5
            * _reverse(
                self.circuit,
                bindings,
                state,
                projected,
                self._plan._derivatives,
                len(self.parameters),
            )[0]
        )

    def matvec(self, bindings, vector):
        """Apply the real Fubini--Study metric without constructing it."""
        value = _real_array(vector, (len(self.parameters),), "Metric vector")
        return self._product(self._bindings(bindings), value)

    def matrix(self, bindings):
        """Explicit small-metric diagnostic; the natural-gradient solver avoids it."""
        columns = self._bindings(bindings)
        count = len(self.parameters)
        if count * count * 8 + self._plan._row_mib * 2**20 > self.max_memory_mb * 2**20:
            raise ResourceError("Explicit metric plus working state exceeds max_memory_mb.")
        result = np.empty((count, count))
        for j in range(count):
            direction = np.zeros(count)
            direction[j] = 1
            result[:, j] = self._product(columns, direction)
        return (result + result.T) * 0.5

    def natural_gradient(
        self, bindings, gradient, *, damping=1e-3, tolerance=1e-10, max_iterations=64
    ):
        """Solve (metric + damping*I) direction = gradient by linear CG.

        Convergence and a freshly recomputed residual are reported explicitly.
        This is not a promise of convergence to the global optimum of a loss.
        """
        damping, tolerance = real(damping), real(tolerance)
        if damping <= 0 or tolerance <= 0:
            raise ValidationError("Metric damping and tolerance must be positive.")
        integer(max_iterations, "max_iterations", 1)
        columns = self._bindings(bindings)
        gradient = _real_array(gradient, (len(self.parameters),), "Gradient")
        direction, residual = np.zeros_like(gradient), gradient.copy()
        search = residual.copy()
        squared = float(residual @ residual)
        if not np.isfinite(squared):
            raise ValidationError("Gradient norm exceeds finite precision.")
        threshold = tolerance * max(1.0, np.sqrt(squared))
        iterations = products = 0
        reason = "converged" if np.sqrt(squared) <= threshold else "iteration_limit"

        def apply(value):
            nonlocal products
            products += 1
            return self._product(columns, value) + damping * value

        if reason != "converged":
            for iteration in range(max_iterations):
                iterations = iteration + 1
                product = apply(search)
                curvature = float(search @ product)
                if not np.isfinite(curvature) or curvature <= 0:
                    reason = "nonpositive_numeric_curvature"
                    break
                step = squared / curvature
                direction += step * search
                if not np.isfinite(direction).all():
                    raise ValidationError("Natural-gradient direction exceeds finite precision.")
                residual -= step * product
                updated = float(residual @ residual)
                if np.sqrt(updated) <= threshold:
                    break
                search = residual + (updated / squared) * search
                squared = updated
        true_residual = float(np.linalg.norm(gradient - apply(direction)))
        converged = true_residual <= threshold
        if converged:
            reason = "converged"
        return NaturalGradientResult(
            direction,
            self.parameters,
            converged,
            iterations,
            products,
            true_residual,
            damping,
            reason,
        )


def quantum_metric(circuit, *, parameters=None, max_qubits=None, max_memory_mb=256):
    """Prepare matrix-free metric products and damped natural-gradient solves."""
    return QuantumMetricPlan(circuit, parameters, max_qubits, max_memory_mb)
