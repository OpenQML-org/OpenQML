"""Explicit approximate MPS execution with canonical Schmidt truncation.

Canonicalization makes each discarded local Schmidt weight a global-state
quantity. The triangle inequality accumulates normalized state-distance bounds,
which bound observable errors. Bounds exclude floating-point roundoff.
Exact MPS execution and automatic execution planning remain separate.
"""

from dataclasses import dataclass

import numpy as np

from .backends import BackendCapabilities, ExecutionOptions
from .compiler import _check_circuit_limits
from .errors import ResourceError, ValidationError
from .ir import Circuit, integer
from .mps import _read_word
from .observables import Hamiltonian
from .parameters import real


def _canonicalize(sites, left, right):
    for q in range(left):
        site = sites[q]
        orthogonal, remainder = np.linalg.qr(site.reshape(-1, site.shape[2]), mode="reduced")
        sites[q] = orthogonal.reshape(site.shape[0], 2, orthogonal.shape[1])
        sites[q + 1] = np.tensordot(remainder, sites[q + 1], axes=(1, 0))
    for q in range(len(sites) - 1, right, -1):
        site = sites[q]
        orthogonal, remainder = np.linalg.qr(site.reshape(site.shape[0], -1).T, mode="reduced")
        sites[q] = orthogonal.T.reshape(orthogonal.shape[1], 2, site.shape[2])
        sites[q - 1] = np.tensordot(sites[q - 1], remainder.T, axes=(2, 0))


def _truncate(sites, qubits, matrix, cap, tolerance):
    a, b = sorted(qubits)
    _canonicalize(sites, a, b)
    if qubits[0] != a:
        matrix = matrix.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
    left, right = sites[a], sites[b]
    block = np.einsum("asb,btc->astc", left, right)
    block = np.einsum("uvst,astc->auvc", matrix.reshape(2, 2, 2, 2), block)
    u, singular, vh = np.linalg.svd(
        block.reshape(2 * left.shape[0], 2 * right.shape[2]), full_matrices=False
    )
    weights = np.square(singular)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValidationError("MPS Schmidt normalization is nonfinite or zero.")
    desired = len(singular)
    if tolerance:
        desired = min(
            desired, int(np.searchsorted(np.cumsum(weights), (1 - tolerance) * total)) + 1
        )
    keep = max(1, min(cap, desired))
    lost = max(0.0, min(1.0, float(weights[keep:].sum()) / total))
    norm = float(np.sqrt(weights[:keep].sum()))
    sites[a] = u[:, :keep].reshape(left.shape[0], 2, keep)
    sites[b] = ((singular[:keep] / norm)[:, None] * vh[:keep]).reshape(keep, 2, right.shape[2])
    distance = float(np.sqrt(2 * lost / (1 + np.sqrt(1 - lost))))
    return lost, distance


@dataclass(frozen=True)
class MPSApproximation:
    values: np.ndarray
    observable_error_bounds: np.ndarray
    state_distance_bound: float
    discarded_weights: tuple[float, ...]
    max_bond_dimension: int
    workspace_bytes: int
    floating_point_error_included: bool = False


@dataclass(frozen=True)
class TruncatedMPSBackend:
    """Opt-in unitary nearest-neighbor approximation, with bounded tensor ranks.

    Use evaluate() to obtain the truncation report. A hard rank cap takes
    precedence over the per-step discarded-weight target; max_state_error is
    checked after every truncation. This backend is never selected implicitly.
    """

    max_bond_dimension: int = 64
    discarded_weight_tolerance: float = 1e-12
    max_state_error: float = 0.1
    max_operations: int = 100_000
    max_work: int = 100_000_000
    name: str = "truncated-mps"
    capabilities: BackendCapabilities = BackendCapabilities(
        native_expectations=True, deterministic_expectations=True
    )

    def __post_init__(self):
        for name in ("max_bond_dimension", "max_operations", "max_work"):
            integer(getattr(self, name), name, 1)
        tolerance, error = real(self.discarded_weight_tolerance), real(self.max_state_error)
        if not 0 <= tolerance < 1 or not 0 <= error <= 2:
            raise ValidationError(
                "Use a discarded weight in [0, 1) and state-error limit in [0, 2]."
            )

    def evaluate(self, circuit, observables, bindings=None, *, options=None):
        from .batch import _observables
        from .simulator import _gate_matrix, _TensorOps

        if not isinstance(circuit, Circuit):
            raise ValidationError("Truncated MPS execution requires a Circuit.")
        options = ExecutionOptions() if options is None else options
        if not isinstance(options, ExecutionOptions):
            raise ValidationError("options must be ExecutionOptions.")
        _check_circuit_limits(circuit, max_operations=self.max_operations)
        if circuit.has_effects:
            raise ValidationError("Truncated MPS currently requires a pure unitary circuit.")
        circuit = circuit.bind(bindings) if bindings is not None else circuit
        if circuit.parameters:
            raise ValidationError("Bind all parameters before truncated MPS execution.")
        observables = _observables(observables, [circuit])
        width = circuit.num_qubits
        if options.max_qubits is not None and width > options.max_qubits:
            raise ResourceError("Truncated MPS exceeds max_qubits.")
        two_site = 0
        for op in circuit.operations:
            if op.name == "barrier":
                continue
            if len(op.qubits) > 2 or (
                len(op.qubits) == 2 and abs(op.qubits[0] - op.qubits[1]) != 1
            ):
                raise ValidationError("Truncated MPS requires nearest-neighbor one/two-wire gates.")
            two_site += len(op.qubits) == 2
        cap = min(self.max_bond_dimension, 1 << min(width // 2, 30))
        terms = sum(len(o.terms) if isinstance(o, Hamiltonian) else 1 for o in observables)
        # Conservative state, QR/SVD and contraction buffers. This admission
        # model is not a process RSS guarantee, nor a vendor-LAPACK proof.
        workspace = (64 * width + 8192) * cap * cap + 4096
        if workspace > options.max_memory_mb * 2**20:
            raise ResourceError("Truncated MPS tensor/workspace bound exceeds max_memory_mb.")
        if (two_site + terms + 1) * width * cap**3 + len(
            circuit.operations
        ) * cap**2 > self.max_work:
            raise ResourceError("Truncated MPS canonicalization/readout bound exceeds max_work.")
        sites = [np.array([[[1], [0]]], complex) for _ in range(width)]
        discarded, distance, maximum = [], 0.0, 1
        ops = _TensorOps()
        for op in circuit.operations:
            if op.name == "barrier":
                continue
            matrix = _gate_matrix(op.name, [p.evaluate({}) for p in op.params], ops, op.matrix)
            if len(op.qubits) == 1:
                q = op.qubits[0]
                sites[q] = np.einsum("st,atb->asb", matrix, sites[q])
            else:
                lost, local_error = _truncate(
                    sites, op.qubits, matrix, cap, self.discarded_weight_tolerance
                )
                discarded.append(lost)
                distance = min(2.0, distance + local_error)
                maximum = max(maximum, *(s.shape[2] for s in sites))
                if distance > self.max_state_error:
                    raise ResourceError("MPS truncation exceeds max_state_error.")
        values, bounds, cache = [], [], {}
        for observable in observables:
            words = observable.terms if isinstance(observable, Hamiltonian) else (observable,)
            value, norm_bound = 0.0, 0.0
            for term in words:
                if term.paulis not in cache:
                    cache[term.paulis] = _read_word(sites, term.paulis)
                value += term.coefficient * cache[term.paulis]
                norm_bound += abs(term.coefficient)
            values.append(value)
            bounds.append(min(2.0, 2 * distance) * norm_bound)
        if not np.isfinite(values).all() or not np.isfinite(bounds).all():
            raise ValidationError("MPS readout/error bounds exceed finite precision.")
        return MPSApproximation(
            np.asarray(values), np.asarray(bounds), distance, tuple(discarded), maximum, workspace
        )

    def expectations_batch(self, circuits, observables, *, options):
        observables = tuple(observables)
        return np.asarray([self.evaluate(c, observables, options=options).values for c in circuits])

    def prepare_gradient_circuit(self, circuit):
        raise ValidationError(
            "Ordinary parameter-shift rules do not differentiate nonlinear MPS truncation. "
            "Use exact MPS/adjoint gradients or a separately validated differentiation method."
        )
