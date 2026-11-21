"""Routed matrix-product simulation with exact or canonical Schmidt readouts."""

import math
from dataclasses import dataclass, replace

import numpy as np

from ..compute.kernel import get_compute_kernel
from ..errors import NumericalError, ResourceError, ValidationError
from ..ir.compiler import _check_circuit_limits
from ..ir.core import _require_circuit, integer
from ..ir.observables import Hamiltonian
from ..ir.parameters import real
from ..runtime.backends import BackendCapabilities, ExecutionOptions


def _routed(circuit, order):
    """Maintain a logical-to-site permutation, including across remote gates."""
    position = [0] * len(order)
    for site, wire in enumerate(order):
        position[wire] = site

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


def _interaction_order(circuit):
    """Pack frequent two-site interactions into disjoint open chains."""
    width = circuit.num_qubits
    counts = {}
    for gate in circuit.operations:
        if len(gate.qubits) == 2 and gate.name not in {"swap", "barrier"}:
            edge = tuple(sorted(gate.qubits))
            counts[edge] = counts.get(edge, 0) + 1
    neighbors = [[] for _ in range(width)]
    roots = list(range(width))

    def root(q):
        while roots[q] != q:
            roots[q] = roots[roots[q]]
            q = roots[q]
        return q

    for a, b in sorted(counts, key=lambda edge: (-counts[edge], edge)):
        if len(neighbors[a]) == 2 or len(neighbors[b]) == 2:
            continue
        x, y = root(a), root(b)
        if x != y:
            neighbors[a].append(b)
            neighbors[b].append(a)
            roots[y] = x
    ordered, seen = [], set()
    for start, linked in enumerate(neighbors):
        if len(linked) != 1 or start in seen:
            continue
        current, prev = start, None
        while current is not None:
            ordered.append(current)
            seen.add(current)
            other = next((q for q in neighbors[current] if q != prev), None)
            prev, current = current, other
    ordered.extend(q for q in range(width) if q not in seen)
    return tuple(ordered)


def _layout(circuit, max_bond_dimension, max_work, *, truncate=False, initial_order=None):
    width = circuit.num_qubits
    if width > max_work or len(circuit.operations) > max_work:
        return None, "MPS site or operation bookkeeping exceeds max_work."
    if circuit.has_effects:
        return None, "MPS requires a unitary circuit."
    if any(len(g.qubits) > 2 and g.name != "barrier" for g in circuit.operations):
        return None, "MPS requires one/two-wire gates."
    order = list(range(width) if initial_order is None else initial_order)
    work, swaps, center, splits = len(circuit.operations), 0, 0, 0
    bonds = [1] * (width + 1)
    payload, peak, maximum = width * 288, width * 288 + 128, 1
    split_work, svd_work = 0, 0
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
        splits += 1
        cut = min(wires) + 1
        left, old, right = bonds[cut - 1], bonds[cut], bonds[cut + 1]
        # Exact splitting retains the full smaller dimension in either gauge.
        size = 2 * min(left, right)
        factor_work = 4 * left * right * size
        split_work += 4 * left * right * (old + 4)
        if truncate or left > right:
            split_work += factor_work
        if truncate:
            svd_work += factor_work
            size = min(size, max_bond_dimension)
        maximum = max(maximum, size)
        if size > max_bond_dimension:
            return {
                "max_bond": size,
                "max_log2_bond": size.bit_length() - 1,
                "workspace_bytes": None,
                "analysis_complete": False,
            }, "Exact bond dimension exceeds max_bond_dimension."
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
        "svd_work": svd_work,
        "split_work": split_work,
        "splits": splits,
        "truncation": "canonical_Schmidt" if truncate else "none_full_bond_retained",
    }, "Routed MPS tensor shapes and workspace admitted."


def _two_site(sites, qubits, matrix, max_bond, tolerance):
    a, b = sorted(qubits)
    if qubits[0] != a:
        matrix = matrix.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
    left, right = sites[a], sites[b]
    block = get_compute_kernel().einsum("asb,btc->astc", left, right)
    transformed = get_compute_kernel().einsum("uvst,astc->auvc", matrix.reshape(2, 2, 2, 2), block)
    rows, columns = 2 * left.shape[0], 2 * right.shape[2]
    block = transformed.reshape(rows, columns)
    kernel = get_compute_kernel()
    if tolerance is None:
        size = min(rows, columns)
        kernel._workspace((block,), ((rows + columns) * size,))
        block = kernel.finite(block)
        if rows <= columns:
            # The left space fits on the bond: I @ block is already left-isometric.
            u, remainder = np.eye(rows, dtype=block.dtype), block
        else:
            try:
                u, remainder = np.linalg.qr(block, mode="reduced")
            except np.linalg.LinAlgError as exc:
                raise NumericalError("MPS QR decomposition did not converge.") from exc
        discarded = 0.0
    else:
        u, singular, vh = kernel.svd(block, full_matrices=False)
        tail = np.cumsum(np.square(singular[::-1]))
        drop = int(np.searchsorted(tail, tolerance * tail[-1], side="right"))
        size = min(max_bond, max(1, len(singular) - drop))
        discarded = (
            min(1.0, tail[len(singular) - size - 1] / tail[-1]) if size < len(singular) else 0.0
        )
        u = u[:, :size]
        kept = singular[:size]
        remainder = (kept / np.linalg.norm(kept))[:, None] * vh[:size]
    sites[a] = u.reshape(left.shape[0], 2, size)
    sites[b] = remainder.reshape(size, 2, right.shape[2])
    return discarded, size


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


@dataclass(frozen=True)
class MPSReadout:
    """Observable values and truncation-only error bounds in input order."""

    values: tuple[float, ...]
    absolute_error_bounds: tuple[float, ...]
    trace_distance_bound: float
    max_bond_dimension: int
    routing_swaps: int
    simulated_wires: tuple[int, ...] | None = None
    readout_transfers: int = 0


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
    layout: str = "source"

    def __post_init__(self):
        if self.layout not in ("source", "auto"):
            raise ValidationError("MPS layout must be source or auto.")
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

    def readout(self, circuit, observables, *, options=None, absolute_error=None, strict=True):
        """Use per-observable Schmidt budgets when absolute_error is supplied."""
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
        budget = get_compute_kernel()._prepare_budget(options.max_memory_mb).max_bytes
        observables = _observables(
            observables, (circuit,), max_items=min(self.max_work, budget // 8) // circuit.num_qubits
        )
        terms = sum(len(o.terms) if isinstance(o, Hamiltonian) else 1 for o in observables)
        if terms * circuit.num_qubits > self.max_work:
            raise ResourceError("MPS readout descriptions exceed max_work.")
        # Allow for order/inverse maps, bond counts and site references before planning.
        if 256 * circuit.num_qubits > budget:
            raise ResourceError("MPS site bookkeeping exceeds max_memory_mb.")
        tolerance = self.max_discarded_weight
        if type(strict) is not bool:
            raise ValidationError("strict must be boolean.")
        if absolute_error is not None and tolerance is None:
            tolerance = 1.0
        profile, reason = _layout(
            circuit, self.max_bond_dimension, self.max_work, truncate=tolerance is not None
        )
        initial_order = range(circuit.num_qubits)
        inputs = len(circuit.operations)
        if (
            self.layout == "auto"
            and 384 * (circuit.num_qubits + inputs) <= budget
            and 4 * circuit.num_qubits + inputs * (4 + inputs.bit_length()) <= self.max_work
        ):
            candidate = _interaction_order(circuit)
            if any(q != i for i, q in enumerate(candidate)):
                trial, trial_reason = _layout(
                    circuit,
                    self.max_bond_dimension,
                    self.max_work,
                    truncate=tolerance is not None,
                    initial_order=candidate,
                )

                def score(p):
                    if p is None or not p["analysis_complete"]:
                        return (True, True)
                    return (
                        False,
                        p["workspace_bytes"] > budget,
                        p["split_work"],
                        p["routing_swaps"],
                        p["workspace_bytes"],
                    )

                if score(trial) < score(profile):
                    initial_order, profile, reason = candidate, trial, trial_reason
        if profile is None or not profile["analysis_complete"]:
            raise ResourceError("MPS admission rejected the circuit: " + reason)
        if profile["workspace_bytes"] > budget:
            raise ResourceError("Routed MPS working tensors exceed max_memory_mb.")
        controller = None
        if absolute_error is not None:
            from ._readout_error import ReadoutErrorBudget

            controller = ReadoutErrorBudget(
                circuit,
                observables,
                absolute_error,
                profile["splits"],
                self.max_work,
                budget - profile["workspace_bytes"],
                initial_order,
            )
        order = list(initial_order)
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
                cutoff = (
                    min(tolerance, controller.cutoff()) if controller is not None else tolerance
                )
                weight, size = _two_site(sites, wires, matrix, self.max_bond_dimension, cutoff)
                if controller is not None:
                    controller.record(weight)
                angle += math.asin(math.sqrt(weight))
                peak = max(peak, size)
        distance = math.sin(min(angle, math.pi / 2))
        from ._mps_readout import read_words

        targets = achieved = None
        if controller is not None:
            targets, achieved = controller.targets, controller.used
            del controller
        terms_by_output = tuple(
            o.terms if isinstance(o, Hamiltonian) else (o,) for o in observables
        )
        words = dict.fromkeys(
            t.paulis
            for row in terms_by_output
            for t in row
            if t.coefficient and t.paulis.count("I") < len(order)
        )
        cache, transfers = read_words(sites, order, words, budget - profile["workspace_bytes"])
        values, bounds = [], []
        for terms in terms_by_output:
            value, norm = 0.0, 0.0
            for term in terms:
                if not term.coefficient:
                    continue
                if term.paulis.count("I") == len(order):
                    value += term.coefficient
                    continue
                value += term.coefficient * cache[term.paulis]
                norm += abs(term.coefficient)
            if not math.isfinite(value) or not math.isfinite(norm):
                raise ValidationError("MPS observable exceeds finite precision.")
            values.append(value)
            bounds.append(2 * distance * norm)
        if achieved is not None:
            bounds = np.minimum(bounds, achieved)
            if strict and np.any(bounds > targets):
                raise ResourceError(
                    "MPS bond or workspace did not meet the requested readout tolerance."
                )
        return MPSReadout(
            tuple(values),
            tuple(bounds),
            distance,
            peak,
            profile["routing_swaps"],
            readout_transfers=transfers,
        )

    def expectations_batch(self, circuits, observables, *, options):
        from .batch import _observables

        observables = _observables(observables, circuits)
        return np.asarray([self.readout(c, observables, options=options).values for c in circuits])
