"""Budgeted, opt-in MPS approximation with routing and certified rank refinement."""

from dataclasses import dataclass, replace

import numpy as np

from .backends import BackendCapabilities, ExecutionOptions
from .compiler import _check_circuit_limits, decompose
from .errors import ResourceError, ValidationError
from .ir import UNITARY_GATES, Circuit, Operation, integer
from .observables import Hamiltonian
from .parameters import real
from .truncated_mps import MPSApproximation, TruncatedMPSBackend


def _route(circuit, max_operations):
    """Temporary nearest-neighbor swaps restore the original wire convention."""
    circuit = decompose(
        circuit, basis=UNITARY_GATES - {"ccx", "cswap"}, max_operations=max_operations
    )
    count = 0
    for op in circuit.operations:
        if op.name == "barrier":
            count += 1
            continue
        if len(op.qubits) > 2:
            raise ValidationError("Synthesize multi-wire matrices before adaptive MPS routing.")
        count += 1 + (
            2 * max(0, abs(op.qubits[0] - op.qubits[1]) - 1) if len(op.qubits) == 2 else 0
        )
        if count > max_operations:
            raise ResourceError("MPS nearest-neighbor routing exceeds max_operations.")
    result = []
    for op in circuit.operations:
        if len(op.qubits) < 2 or op.name == "barrier":
            result.append(op)
            continue
        a, b = sorted(op.qubits)
        swaps = tuple(Operation("swap", (i, i + 1)) for i in range(b - 1, a, -1))
        result.extend(swaps)
        mapped = (a, a + 1) if op.qubits[0] == a else (a + 1, a)
        result.append(replace(op, qubits=mapped))
        result.extend(reversed(swaps))
    return replace(circuit, operations=tuple(result))


@dataclass(frozen=True)
class AdaptiveMPSResult:
    approximation: MPSApproximation
    attempted_bonds: tuple
    routed_operations: int
    total_work_bound: int

    @property
    def values(self):
        return self.approximation.values

    @property
    def observable_error_bounds(self):
        return self.approximation.observable_error_bounds

    @property
    def truncation_free(self):
        return not any(self.approximation.discarded_weights)


@dataclass(frozen=True)
class AdaptiveMPSBackend:
    """Refine bond rank until every declared observable meets its error budget.

    Every retry starts from the original circuit; a low-rank state is never
    mistaken for an exact initial state. Bounds exclude floating-point error.
    This backend is explicit and does not masquerade as an exact gradient path.
    """

    initial_bond_dimension: int = 2
    max_bond_dimension: int = 64
    max_observable_error: float = 1e-3
    max_operations: int = 100_000
    max_work: int = 100_000_000
    max_total_work: int = 500_000_000
    name: str = "adaptive-mps"
    capabilities: BackendCapabilities = BackendCapabilities(
        native_expectations=True, deterministic_expectations=True
    )

    def __post_init__(self):
        for name in (
            "initial_bond_dimension",
            "max_bond_dimension",
            "max_operations",
            "max_work",
            "max_total_work",
        ):
            integer(getattr(self, name), name, 1)
        if self.initial_bond_dimension > self.max_bond_dimension:
            raise ValidationError("Initial bond must not exceed the maximum bond.")
        if real(self.max_observable_error) < 0:
            raise ValidationError("max_observable_error must be nonnegative.")

    def evaluate(self, circuit, observables, bindings=None, *, options=None):
        from .batch import _observables

        if not isinstance(circuit, Circuit) or circuit.has_effects:
            raise ValidationError("Adaptive MPS requires a unitary Circuit.")
        _check_circuit_limits(circuit, max_operations=self.max_operations)
        options = ExecutionOptions() if options is None else options
        if not isinstance(options, ExecutionOptions):
            raise ValidationError("options must be ExecutionOptions.")
        circuit = circuit.bind(bindings) if bindings is not None else circuit
        if circuit.parameters:
            raise ValidationError("Bind all parameters before adaptive MPS execution.")
        obs = _observables(observables, [circuit])
        routed = _route(circuit, self.max_operations)
        n = routed.num_qubits
        terms = sum(len(o.terms) if isinstance(o, Hamiltonian) else 1 for o in obs)
        two = sum(len(op.qubits) == 2 and op.name != "barrier" for op in routed.operations)
        maximum = min(self.max_bond_dimension, 1 << min(n // 2, 30))
        cap, work, attempts = min(self.initial_bond_dimension, maximum), 0, []
        while True:
            estimate = (two + terms + 1) * n * cap**3 + len(routed.operations) * cap**2
            if work + estimate > self.max_total_work:
                raise ResourceError("MPS rank refinement exceeds max_total_work.")
            work += estimate
            attempts.append(cap)
            backend = TruncatedMPSBackend(
                max_bond_dimension=cap,
                discarded_weight_tolerance=0,
                max_state_error=2,
                max_operations=self.max_operations,
                max_work=self.max_work,
            )
            report = backend.evaluate(routed, obs, options=options)
            if np.all(report.observable_error_bounds <= self.max_observable_error):
                return AdaptiveMPSResult(report, tuple(attempts), len(routed.operations), work)
            if cap == maximum:
                raise ResourceError(
                    "MPS rank cap cannot certify max_observable_error; increase rank/resources "
                    "or choose exact contraction/QPU execution."
                )
            cap = min(maximum, cap * 2)

    def expectations_batch(self, circuits, observables, *, options):
        obs = tuple(observables)
        return np.asarray([self.evaluate(c, obs, options=options).values for c in circuits])

    def prepare_gradient_circuit(self, circuit):
        raise ValidationError(
            "Adaptive rank/truncation is nonlinear. Use an exact native/adjoint gradient path; "
            "ordinary parameter shifts do not differentiate this approximation."
        )
