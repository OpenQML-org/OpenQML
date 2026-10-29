"""Full-SVD nearest-neighbor MPS readouts with planned tensor dimensions.

No singular value is intentionally discarded, including numerical zeros. Bond
dimensions follow matrix shapes rather than a numerical rank threshold. The
admission model reserves explicit state, contraction and LAPACK work space;
it does not impose an operating-system RSS limit or count SDK imports.
"""

from dataclasses import dataclass

import numpy as np

from .backends import BackendCapabilities
from .compiler import _check_circuit_limits
from .errors import ResourceError, ValidationError
from .ir import integer
from .observables import Hamiltonian


def _layout(circuit, max_bond_dimension, max_work):
    width = circuit.num_qubits
    if width > max_work:
        return None, "MPS site bookkeeping exceeds max_work."
    if circuit.has_effects:
        return None, "MPS currently requires a unitary circuit."
    bonds = [1] * (width + 1)
    payload, peak, maximum = width * 32, width * 32 + 128, 1
    for index, op in enumerate(circuit.operations):
        if index >= max_work:
            return None, "MPS layout analysis exceeds max_work."
        if op.name == "barrier":
            continue
        if len(op.qubits) > 2 or (len(op.qubits) == 2 and abs(op.qubits[0] - op.qubits[1]) != 1):
            return None, "Full-SVD MPS requires nearest-neighbor one/two-wire gates."
        if len(op.qubits) == 1:
            q = op.qubits[0]
            peak = max(peak, payload + 128 * bonds[q] * bonds[q + 1])
            continue
        cut = min(op.qubits) + 1
        left, old, right = bonds[cut - 1], bonds[cut], bonds[cut + 1]
        # A full SVD of (2*left, 2*right) retains exactly this many columns.
        size = 2 * min(left, right)
        maximum = max(maximum, size)
        if size > max_bond_dimension:
            return {
                "max_bond": size,
                "max_log2_bond": size.bit_length() - 1,
                "workspace_bytes": None,
                "analysis_complete": False,
            }, "Full-SVD bond dimension exceeds max_bond_dimension."
        # Includes both two-site contractions, SVD copies/output and a large
        # reserve for vendor LAPACK workspace. Small fixed matrices are extra.
        peak = max(peak, payload + 2048 * left * right + 1024)
        payload += 32 * (left + right) * (size - old)
        bonds[cut] = size
        peak = max(peak, payload)
    # Pauli transfer contractions retain at most a few bond-square buffers.
    peak = max(peak, payload + 256 * maximum * maximum + 1024)
    return {
        "max_bond": maximum,
        "max_log2_bond": maximum.bit_length() - 1,
        "workspace_bytes": peak,
        "analysis_complete": True,
        "truncation": "none_full_SVD_columns_retained",
    }, "Full-SVD tensor-shape and workspace admission completed."


def _two_site(sites, qubits, matrix):
    a, b = sorted(qubits)
    if qubits[0] != a:
        matrix = matrix.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
    left, right = sites[a], sites[b]
    block = np.einsum("asb,btc->astc", left, right)
    transformed = np.einsum("uvst,astc->auvc", matrix.reshape(2, 2, 2, 2), block)
    rows, columns = 2 * left.shape[0], 2 * right.shape[2]
    u, singular, vh = np.linalg.svd(transformed.reshape(rows, columns), full_matrices=False)
    size = len(singular)
    sites[a] = u.reshape(left.shape[0], 2, size)
    sites[b] = (singular[:, None] * vh).reshape(size, 2, right.shape[2])


def _read_word(sites, word):
    from .simulator import _I, _X, _Y, _Z

    matrices = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
    environment = np.ones((1, 1), dtype=complex)
    for site, pauli in zip(sites, word, strict=True):
        matrix = matrices[pauli]
        result = np.zeros((site.shape[2], site.shape[2]), dtype=complex)
        for s in range(2):
            for t in range(2):
                if matrix[s, t]:
                    product = environment @ site[:, t, :]
                    result += matrix[s, t] * (site[:, s, :].conj().T @ product)
        environment = result
    return float(environment[0, 0].real)


@dataclass(frozen=True)
class MPSBackend:
    """Numeric exact-representation MPS; native autodiff uses gradient bridges.

    Input is the all-zero product state. Dense state output, effects, nonlocal
    routing and approximation are deliberately outside this engine's contract.
    """

    name: str = "mps"
    max_bond_dimension: int = 256
    max_operations: int = 100_000
    max_work: int = 1_000_000
    capabilities: BackendCapabilities = BackendCapabilities(
        native_expectations=True, deterministic_expectations=True
    )

    def __post_init__(self):
        for name in ("max_bond_dimension", "max_operations", "max_work"):
            integer(getattr(self, name), name, 1)

    def expectations_batch(self, circuits, observables, *, options):
        from .batch import _observables
        from .simulator import _gate_matrix, _TensorOps

        _observables(observables, circuits)
        rows, ops = [], _TensorOps()
        numerator, denominator = float(options.max_memory_mb).as_integer_ratio()
        budget = (numerator << 20) // denominator
        for circuit in circuits:
            _check_circuit_limits(circuit, max_operations=self.max_operations)
            if circuit.parameters:
                raise ValidationError("Bind parameters before numeric MPS execution.")
            if options.max_qubits is not None and circuit.num_qubits > options.max_qubits:
                raise ResourceError("MPS source exceeds the explicit max_qubits limit.")
            profile, reason = _layout(circuit, self.max_bond_dimension, self.max_work)
            if (
                profile is None
                or not profile["analysis_complete"]
                or profile["workspace_bytes"] > budget
            ):
                raise ResourceError("MPS admission rejected the circuit: " + reason)
            terms = sum(len(o.terms) if isinstance(o, Hamiltonian) else 1 for o in observables)
            if terms * circuit.num_qubits > self.max_work:
                raise ResourceError("MPS readout descriptions exceed max_work.")
            sites = [np.array([[[1], [0]]], dtype=complex) for _ in range(circuit.num_qubits)]
            for operation in circuit.operations:
                if operation.name == "barrier":
                    continue
                matrix = _gate_matrix(
                    operation.name,
                    tuple(value.evaluate({}) for value in operation.params),
                    ops,
                    operation.matrix,
                )
                if len(operation.qubits) == 1:
                    q = operation.qubits[0]
                    sites[q] = np.einsum("st,atb->asb", matrix, sites[q])
                else:
                    _two_site(sites, operation.qubits, matrix)
            values, cache = [], {}
            for observable in observables:
                terms = observable.terms if isinstance(observable, Hamiltonian) else (observable,)
                value = 0.0
                for term in terms:
                    if term.paulis not in cache:
                        cache[term.paulis] = _read_word(sites, term.paulis)
                    value += term.coefficient * cache[term.paulis]
                values.append(value)
            rows.append(values)
        return np.asarray(rows)
