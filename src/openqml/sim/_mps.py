"""Routed matrix-product simulation with exact or canonical Schmidt readouts."""

import math
from dataclasses import dataclass, replace

import numpy as np

from ..compute.kernel import get_compute_kernel
from ..errors import ResourceError, ValidationError
from ..ir.compiler import _check_circuit_limits
from ..ir.core import _require_circuit, integer
from ..ir.observables import Hamiltonian
from ..ir.parameters import real
from ..runtime.backends import BackendCapabilities, ExecutionOptions


def _routed(circuit, order):
    """Maintain a logical-to-site permutation, including across remote gates."""
    position = list(order)

    def exchange(a, b):
        order[a], order[b] = order[b], order[a]
        position[order[a]], position[order[b]] = a, b

    for gate in circuit.operations:
        if gate.name == "barrier":
            continue
        sites = tuple(position[q] for q in gate.qubits)
        if len(sites) == 1:
            yield gate, sites
            continue
        if len(sites) != 2:
            raise ResourceError("MPS requires gates on at most two wires.")
        a, b = sites
        if gate.name == "swap":
            exchange(a, b)
            continue
        step = 1 if a < b else -1
        while abs(a - b) > 1:
            yield None, (a, a + step)
            exchange(a, a + step)
            a += step
        yield gate, (a, b)


def _layout(circuit, max_bond_dimension, max_work, *, truncate=False):
    width = circuit.num_qubits
    if width > max_work or len(circuit.operations) > max_work:
        return None, "MPS site or operation bookkeeping exceeds max_work."
    if circuit.has_effects:
        return None, "MPS requires a unitary circuit."
    if any(len(g.qubits) > 2 and g.name != "barrier" for g in circuit.operations):
        return None, "MPS requires one/two-wire gates."
    order = list(range(width))
    work, swaps, center = len(circuit.operations), 0, 0
    bonds = [1] * (width + 1)
    payload, peak, maximum = width * 288, width * 288 + 128, 1
    for op, wires in _routed(circuit, order):
        if op is None:
            swaps += 1
            work += 1
        if len(wires) == 2 and truncate:
            a = min(wires)
            work += abs(a - center)
            center = a + 1
        if work > max_work:
            return None, "MPS routing and canonical movement exceed max_work."
        if len(wires) == 1:
            q = wires[0]
            peak = max(peak, payload + 128 * bonds[q] * bonds[q + 1])
            continue
        cut = min(wires) + 1
        left, old, right = bonds[cut - 1], bonds[cut], bonds[cut + 1]
        # A full SVD of (2*left, 2*right) retains exactly this many columns.
        size = 2 * min(left, right)
        if truncate:
            size = min(size, max_bond_dimension)
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
    peak = max(peak, payload + (512 if truncate else 256) * maximum * maximum + 1024)
    return {
        "max_bond": maximum,
        "max_log2_bond": maximum.bit_length() - 1,
        "workspace_bytes": peak,
        "analysis_complete": True,
        "routing_swaps": swaps,
        "truncation": "canonical_Schmidt" if truncate else "none_full_SVD_columns_retained",
    }, "Routed MPS tensor shapes and workspace admitted."


def _two_site(sites, qubits, matrix, max_bond, tolerance):
    a, b = sorted(qubits)
    if qubits[0] != a:
        matrix = matrix.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
    left, right = sites[a], sites[b]
    block = get_compute_kernel().einsum("asb,btc->astc", left, right)
    transformed = get_compute_kernel().einsum("uvst,astc->auvc", matrix.reshape(2, 2, 2, 2), block)
    rows, columns = 2 * left.shape[0], 2 * right.shape[2]
    u, singular, vh = get_compute_kernel().svd(
        transformed.reshape(rows, columns), full_matrices=False
    )
    size, angle = len(singular), 0.0
    if tolerance is not None:
        # Both environments are isometric here. Squared singular values are
        # physical Schmidt probabilities, including for the inserted SWAPs.
        tail = np.cumsum(np.square(singular[::-1]))
        size = min(
            max_bond, max(1, size - int(np.searchsorted(tail, tolerance * tail[-1], side="right")))
        )
        if size < len(singular):
            discarded = tail[len(singular) - size - 1] / tail[-1]
            angle = math.asin(math.sqrt(min(1.0, discarded)))
        singular = singular[:size] / np.linalg.norm(singular[:size])
        u, vh = u[:, :size], vh[:size]
    sites[a] = u.reshape(left.shape[0], 2, size)
    sites[b] = (singular[:, None] * vh).reshape(size, 2, right.shape[2])
    return angle, size


def _move_center(sites, center, target):
    # Thin QR moves the orthogonality center without changing the state.
    while center < target:
        left, _, right = sites[center].shape
        q, r = np.linalg.qr(sites[center].reshape(2 * left, right))
        sites[center] = q.reshape(left, 2, -1)
        sites[center + 1] = np.tensordot(r, sites[center + 1], axes=1)
        center += 1
    while center > target:
        left, _, right = sites[center].shape
        q, r = np.linalg.qr(sites[center].reshape(left, 2 * right).T)
        sites[center] = q.T.reshape(-1, 2, right)
        sites[center - 1] = np.tensordot(sites[center - 1], r.T, axes=1)
        center -= 1


def _read_word(sites, word):
    from .dense import _I, _X, _Y, _Z

    matrices = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
    environment = np.ones((1, 1), dtype=complex)
    for site, pauli in zip(sites, word, strict=True):
        matrix = matrices[pauli]
        result = np.zeros((site.shape[2], site.shape[2]), dtype=complex)
        for s in range(2):
            for t in range(2):
                if matrix[s, t]:
                    product = get_compute_kernel().matmul(environment, site[:, t, :])
                    result += matrix[s, t] * (
                        get_compute_kernel().matmul(site[:, s, :].conj().T, product)
                    )
        environment = result
    return float(environment[0, 0].real)


@dataclass(frozen=True)
class MPSReadout:
    """Observable values and truncation-only error bounds in input order."""

    values: tuple[float, ...]
    absolute_error_bounds: tuple[float, ...]
    trace_distance_bound: float
    max_bond_dimension: int
    routing_swaps: int


@dataclass(frozen=True)
class MPSBackend:
    """Route unitary one/two-wire gates on an MPS starting in the zero product state."""

    name: str = "mps"
    max_bond_dimension: int = 256
    max_operations: int = 100_000
    max_work: int = 1_000_000
    capabilities: BackendCapabilities = BackendCapabilities(
        native_expectations=True, deterministic_expectations=True, exact_expectations=True
    )
    max_discarded_weight: float | None = None

    def __post_init__(self):
        for name in ("max_bond_dimension", "max_operations", "max_work"):
            integer(getattr(self, name), name, 1)
        if self.max_discarded_weight is not None:
            tolerance = real(self.max_discarded_weight)
            if not 0 <= tolerance < 1:
                raise ValidationError("max_discarded_weight must be in [0, 1).")
            object.__setattr__(self, "max_discarded_weight", tolerance)
            object.__setattr__(
                self, "capabilities", replace(self.capabilities, exact_expectations=False)
            )

    def readout(self, circuit, observables, *, options=None):
        from .batch import _observables
        from .dense import _FIXED, _gate_matrix, _TensorOps

        _require_circuit(circuit)
        options = ExecutionOptions() if options is None else options
        if not isinstance(options, ExecutionOptions):
            raise ValidationError("options must be ExecutionOptions.")
        _check_circuit_limits(circuit, max_operations=self.max_operations)
        if circuit.parameters:
            raise ValidationError("Bind parameters before numeric MPS execution.")
        if options.max_qubits is not None and circuit.num_qubits > options.max_qubits:
            raise ResourceError("MPS source exceeds the explicit max_qubits limit.")
        observables = _observables(observables, (circuit,))
        terms = sum(len(o.terms) if isinstance(o, Hamiltonian) else 1 for o in observables)
        if terms * circuit.num_qubits > self.max_work:
            raise ResourceError("MPS readout descriptions exceed max_work.")
        # Allow for order/inverse maps, bond counts and site references before planning.
        budget = get_compute_kernel()._prepare_budget(options.max_memory_mb).max_bytes
        if 256 * circuit.num_qubits > budget:
            raise ResourceError("MPS site bookkeeping exceeds max_memory_mb.")
        tolerance = self.max_discarded_weight
        profile, reason = _layout(
            circuit, self.max_bond_dimension, self.max_work, truncate=tolerance is not None
        )
        if profile is None or not profile["analysis_complete"]:
            raise ResourceError("MPS admission rejected the circuit: " + reason)
        if profile["workspace_bytes"] > budget:
            raise ResourceError("Routed MPS working tensors exceed max_memory_mb.")
        order = list(range(circuit.num_qubits))
        sites = [np.array([[[1], [0]]], dtype=complex) for _ in order]
        ops, center, angle, peak = _TensorOps(), 0, 0.0, 1
        for gate, wires in _routed(circuit, order):
            matrix = (
                _FIXED["swap"]
                if gate is None
                else _gate_matrix(
                    gate.name,
                    tuple(v.evaluate({}) for v in gate.params),
                    ops,
                    gate.matrix,
                )
            )
            if len(wires) == 1:
                q = wires[0]
                sites[q] = get_compute_kernel().einsum("st,atb->asb", matrix, sites[q])
            else:
                if tolerance is not None:
                    _move_center(sites, center, min(wires))
                    center = min(wires) + 1
                delta, size = _two_site(sites, wires, matrix, self.max_bond_dimension, tolerance)
                angle += delta
                peak = max(peak, size)
        distance = math.sin(min(angle, math.pi / 2))
        values, bounds, cache = [], [], {}
        for observable in observables:
            terms = observable.terms if isinstance(observable, Hamiltonian) else (observable,)
            value, norm = 0.0, 0.0
            for term in terms:
                if term.paulis not in cache:
                    word = "".join(term.paulis[q] for q in order)
                    cache[term.paulis] = _read_word(sites, word)
                value += term.coefficient * cache[term.paulis]
                if term.paulis.count("I") != len(order):
                    norm += abs(term.coefficient)
            values.append(value)
            bounds.append(2 * distance * norm)
        return MPSReadout(tuple(values), tuple(bounds), distance, peak, profile["routing_swaps"])

    def expectations_batch(self, circuits, observables, *, options):
        from .batch import _observables

        observables = _observables(observables, circuits)
        return np.asarray([self.readout(c, observables, options=options).values for c in circuits])
