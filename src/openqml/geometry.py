"""Matrix-free pure-state quantum geometry using tangent and adjoint sweeps.

The Fubini--Study product is Re[J* (I-|psi><psi|) J v]. A directional
forward sweep and reverse contraction avoid storing all parameter tangents or
the quadratic metric. Natural gradients solve the damped system by linear CG.
Independent state factors contribute additive metrics, including shared
parameters. State-vector cost is exponential in the largest connected factor.
Research background: https://arxiv.org/abs/1909.02108 and
https://arxiv.org/abs/2011.02991. This is not the nonlinear CQNG algorithm.
"""

from dataclasses import dataclass, field

import numpy as np

from ._gradient_common import _binding_columns, _names, _real_array
from ._structure import split_circuit
from .adjoint import AdjointPlan, _check_adjoint, _reverse, _tangent_forward
from .compiler import _check_circuit_limits
from .compute import compute_operation, get_compute_kernel, memory_budget
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
    max_work: int = 1_000_000
    _plans: tuple = field(init=False, repr=False)
    _row_mib: float = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.circuit, Circuit):
            raise ValidationError("Quantum geometry requires an OpenQML Circuit.")
        _check_circuit_limits(self.circuit)
        if self.circuit.has_effects:
            raise ValidationError("Quantum geometry requires a unitary pure-state circuit.")
        if self.max_qubits is not None:
            integer(self.max_qubits, "max_qubits", 1)
            if self.circuit.num_qubits > self.max_qubits:
                raise ResourceError("Quantum geometry exceeds the explicit max_qubits limit.")
        budget = memory_budget(self.max_memory_mb)
        if budget <= 0:
            raise ValidationError("max_memory_mb must be positive.")
        integer(self.max_work, "max_work", 1)
        names = _names(self.parameters, self.circuit.parameters)
        work = 0

        def charge(amount):
            nonlocal work
            work += amount
            if work > self.max_work:
                raise ResourceError("Quantum geometry dependency analysis exceeds max_work.")

        components = split_circuit(self.circuit, charge)
        metadata = (256 * work + 128 * len(names)) / 2**20
        if metadata >= budget:
            raise ResourceError("Quantum geometry descriptors exceed max_memory_mb.")
        plans, peak = [], 0
        positions = {name: i for i, name in enumerate(names)}
        for _, _, child in components:
            indices = tuple(sorted(positions[p] for p in child.parameters if p in positions))
            if not indices:
                continue
            _check_adjoint(child, None, budget - metadata)
            plan = AdjointPlan(
                child,
                (PauliString("I" * child.num_qubits),),
                tuple(names[i] for i in indices),
                None,
                budget - metadata,
            )
            plans.append((plan, indices))
            peak = max(peak, plan._row_mib)
        object.__setattr__(self, "_plans", tuple(plans))
        object.__setattr__(self, "_row_mib", peak + metadata)
        object.__setattr__(self, "parameters", names)
        object.__setattr__(self, "max_memory_mb", budget)

    def to_dict(self):
        return {
            "scope": "real_Fubini_Study_of_zero_initialized_pure_product_factors",
            "num_qubits": self.circuit.num_qubits,
            "parameters": list(self.parameters),
            "differentiated_components": len(self._plans),
            "max_component_qubits": max((p.circuit.num_qubits for p, _ in self._plans), default=0),
            "working_mb": self._row_mib,
            "parameter_reuse": "additive_factor_metrics_in_shared_global_coordinates",
            "roundoff_excluded": True,
        }

    def _bindings(self, bindings):
        columns, count = _binding_columns(
            {} if bindings is None else bindings, self.circuit.parameters, None, label="Metric"
        )
        if count != 1:
            raise ValidationError("Quantum metric products require one binding at a time.")
        return columns

    def _product(self, bindings, vector):
        result = np.zeros(len(self.parameters))
        for plan, indices in self._plans:
            local = {p: bindings[p] for p in plan.circuit.parameters}
            state, tangent = _tangent_forward(
                plan.circuit, local, plan._derivatives, vector[np.asarray(indices)][None, :]
            )
            overlap = np.sum(state.conj() * tangent, axis=-1)
            projected = tangent - state * overlap[:, None]
            result[np.asarray(indices)] += (
                0.5
                * _reverse(plan.circuit, local, state, projected, plan._derivatives, len(indices))[
                    0
                ]
            )
        return result

    def matvec(self, bindings, vector):
        """Apply the real Fubini--Study metric without constructing it."""
        value = _real_array(vector, (len(self.parameters),), "Metric vector")
        return self._product(self._bindings(bindings), value)

    def matrix(self, bindings):
        """Explicit small-metric diagnostic; the natural-gradient solver avoids it."""
        columns = self._bindings(bindings)
        count = len(self.parameters)
        if count * count * 8 + self._row_mib * 2**20 > memory_budget(self.max_memory_mb) * 2**20:
            raise ResourceError("Explicit metric plus working state exceeds max_memory_mb.")
        result = np.empty((count, count))
        for j in range(count):
            direction = np.zeros(count)
            direction[j] = 1
            result[:, j] = self._product(columns, direction)
        return (result + result.T) * 0.5

    @compute_operation
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
        squared = float(get_compute_kernel().matmul(residual, residual))
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
                curvature = float(get_compute_kernel().matmul(search, product))
                if not np.isfinite(curvature) or curvature <= 0:
                    reason = "nonpositive_numeric_curvature"
                    break
                step = squared / curvature
                direction += step * search
                if not np.isfinite(direction).all():
                    raise ValidationError("Natural-gradient direction exceeds finite precision.")
                residual -= step * product
                updated = float(get_compute_kernel().matmul(residual, residual))
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


def quantum_metric(
    circuit, *, parameters=None, max_qubits=None, max_memory_mb=256, max_work=1_000_000
):
    """Prepare matrix-free metric products and damped natural-gradient solves."""
    return QuantumMetricPlan(circuit, parameters, max_qubits, max_memory_mb, max_work)
