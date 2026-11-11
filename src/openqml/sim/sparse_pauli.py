"""Noise-aware sparse Heisenberg propagation with per-output truncation bounds.

Rudolph et al., arXiv:2505.21606, Theory Box 3 (2026 revision): sum discarded
Pauli coefficient L1 norms. CPTP adjoints contract Hermitian operator norm, so
the bound holds for any input state; evaluation here uses the all-zero state.
No random-circuit/average-case assumptions establish these reported bounds.
Roundoff and approximate validation of supplied matrix payloads are excluded.
"""

import math
from collections.abc import Mapping, Set
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from ..autodiff._common import _binding_columns
from ..compute import get_compute_kernel, memory_budget
from ..errors import ResourceError, ValidationError
from ..ir import CHANNEL_GATES, integer
from ..ir._structure import compact_pair
from ..ir.compiler import _check_circuit_limits
from ..ir.observables import Hamiltonian, PauliString, SparseHamiltonian, SparsePauliString
from ..ir.parameters import real
from .dense import _I, _X, _Y, _Z, _gate_matrix, _TensorOps
from .pauli import _CLIFFORD, _ROTATIONS, _clifford, _clifford_table, _masks, _rotation_sign


def _terms(observable, width):
    terms = (
        observable.terms
        if isinstance(observable, (Hamiltonian, SparseHamiltonian))
        else (observable,)
    )
    result = []
    for term in terms:
        if isinstance(term, PauliString) and len(term.paulis) == width:
            sites = tuple((q, p) for q, p in enumerate(term.paulis) if p != "I")
            result.append((sites, term.coefficient, len(term.paulis)))
        elif isinstance(term, SparsePauliString) and term.num_qubits == width:
            result.append((term.sites, term.coefficient, len(term.sites)))
        else:
            raise ValidationError("Sparse propagation observable width/type is invalid.")
    return tuple(result)


@dataclass(frozen=True)
class SparsePauliResult:
    values: np.ndarray
    error_bounds: np.ndarray
    peak_terms: tuple[int, ...]
    discarded_terms: tuple[int, ...]
    work: int
    active_qubits: int
    num_qubits: int

    def to_dict(self):
        return {
            "scope": "zero_state_expectations_of_unconditional_CPTP_circuit",
            "values": self.values.tolist(),
            "error_bounds": self.error_bounds.tolist(),
            "bound": "sum_discarded_coefficient_L1; "
            "excludes_floating_point_and_payload_validation_error",
            "peak_terms": list(self.peak_terms),
            "discarded_terms": list(self.discarded_terms),
            "work": self.work,
            "active_qubits": self.active_qubits,
            "num_qubits": self.num_qubits,
            "intentional_approximation": bool(np.any(self.error_bounds)),
            "hardware_execution_checked": False,
        }


@dataclass(frozen=True, eq=False)
class SparsePauliPlan:
    circuit: object
    observables: tuple
    max_error: float = 0.0
    max_terms: int = 4096
    coefficient_cutoff: float = 0.0
    max_memory_mb: float = 256
    max_work: int = 2_000_000
    max_operations: int = 100_000
    max_qubits: int | None = None
    _circuit: object = field(init=False, repr=False)
    _initial: tuple = field(init=False, repr=False)
    _capacity: int = field(init=False, repr=False)
    _metadata: int = field(init=False, repr=False)

    def __post_init__(self):
        _check_circuit_limits(self.circuit, max_operations=self.max_operations)
        integer(self.max_terms, "max_terms", 1)
        integer(self.max_work, "max_work", 1)
        if self.max_qubits is not None:
            integer(self.max_qubits, "max_qubits", 1)
            if self.circuit.num_qubits > self.max_qubits:
                raise ResourceError("Sparse propagation exceeds the explicit max_qubits limit.")
        error, cutoff, budget = (
            real(self.max_error),
            real(self.coefficient_cutoff),
            memory_budget(self.max_memory_mb),
        )
        if error < 0 or cutoff < 0 or budget <= 0 or (cutoff and not error):
            raise ValidationError("Use nonnegative error/cutoff and a positive memory budget.")
        if any(op.condition is not None for _, op in self.circuit.walk_operations()):
            raise ValidationError(
                "Sparse propagation excludes classical feedback; use ComponentPlan."
            )
        if isinstance(self.observables, (str, bytes, Mapping, Set)):
            raise ValidationError("Sparse propagation requires ordered observables.")
        try:
            observables = tuple(self.observables)
        except TypeError as exc:
            raise ValidationError("Sparse propagation requires ordered observables.") from exc
        if not observables:
            raise ValidationError("Sparse propagation requires at least one observable.")
        work = 0

        def charge(amount):
            nonlocal work
            work += amount
            if work > self.max_work:
                raise ResourceError("Sparse propagation preparation exceeds max_work.")

        active = sorted(
            {q for op in self.circuit.operations if op.name != "barrier" for q in op.qubits}
        )
        quantum = {q: i for i, q in enumerate(active)}
        compact, _ = compact_pair(self.circuit, self.circuit, charge)
        initial = []
        for observable in observables:
            frontier = {}
            for sites, coefficient, cost in _terms(observable, self.circuit.num_qubits):
                charge(1 + cost)
                if not coefficient or any(q not in quantum and p in "XY" for q, p in sites):
                    continue
                x = z = 0
                for q, p in sites:
                    if q in quantum:
                        bit = 1 << quantum[q]
                        x |= bit if p in "XY" else 0
                        z |= bit if p in "YZ" else 0
                frontier[x, z] = frontier.get((x, z), 0) + coefficient
            initial.append(tuple(sorted((key, c) for key, c in frontier.items() if c)))
        metadata = 512 * (work + len(active) + sum(len(f) for f in initial))
        capacity_bytes = int(budget * 2**20)
        entry_bytes = 512 + 4 * ((len(active) + 7) // 8)
        capacity = min(self.max_terms, (capacity_bytes - metadata) // (4 * entry_bytes))
        if capacity < 1:
            raise ResourceError("Sparse propagation descriptors/frontier exceed max_memory_mb.")
        for name, value in (
            ("observables", observables),
            ("max_error", error),
            ("coefficient_cutoff", cutoff),
            ("max_memory_mb", budget),
            ("_circuit", compact),
            ("_initial", tuple(initial)),
            ("_capacity", capacity),
            ("_metadata", metadata),
        ):
            object.__setattr__(self, name, value)

    def to_dict(self):
        return {
            "num_qubits": self.circuit.num_qubits,
            "active_qubits": len({q for op in self._circuit.operations for q in op.qubits}),
            "max_error_per_output": self.max_error,
            "effective_max_terms": self._capacity,
            "max_work": self.max_work,
            "max_memory_mb": self.max_memory_mb,
            "approximation": "explicit_only; no_polynomial_runtime_or_average_case_guarantee",
            "autodiff": "numeric_report_only; truncation_does_not_obey_parameter_shift",
        }

    def evaluate(self, bindings=None):
        columns, count = _binding_columns(
            {} if bindings is None else bindings,
            self.circuit.parameters,
            None,
            label="Sparse Pauli",
        )
        if count != 1:
            raise ValidationError("Sparse propagation requires one scalar binding at a time.")
        bindings = {name: float(value[0]) for name, value in columns.items()}
        work = 0

        def charge(amount):
            nonlocal work
            work += amount
            if work > self.max_work:
                raise ResourceError("Sparse propagation exceeds max_work.")

        real(self.circuit.global_phase.evaluate(bindings))
        for op in self._circuit.operations:
            charge(1 + len(op.params))
            parameters = tuple(real(p.evaluate(bindings)) for p in op.params)
            if op.name in CHANNEL_GATES - {"kraus"} and not 0 <= parameters[0] <= 1:
                raise ValidationError("Channel probability must lie in [0, 1].")

        values, errors, peaks, discarded, memo = [], [], [], [], {}
        for initial in self._initial:
            if initial not in memo:
                frontier, error, removed = dict(initial), 0.0, 0
                peak = len(frontier)

                def trim(frontier):
                    nonlocal error, removed
                    frontier = {key: c for key, c in frontier.items() if c}
                    if any(not math.isfinite(c) for c in frontier.values()):
                        raise ResourceError("Sparse coefficients exceed finite precision.")
                    candidates = sorted(frontier, key=lambda k: (abs(frontier[k]), k))
                    required = max(0, len(frontier) - self._capacity)
                    chosen = candidates[:required]
                    cost = math.fsum(abs(frontier[k]) for k in chosen)
                    bound = math.nextafter(error + cost, math.inf) if cost else error
                    if bound > self.max_error:
                        raise ResourceError("Sparse term capacity requires more than max_error.")
                    for key in candidates[required:]:
                        if (
                            not self.coefficient_cutoff
                            or abs(frontier[key]) > self.coefficient_cutoff
                        ):
                            break
                        trial = math.fsum((cost, abs(frontier[key])))
                        next_bound = math.nextafter(error + trial, math.inf)
                        if next_bound > self.max_error:
                            break
                        chosen.append(key)
                        cost, bound = trial, next_bound
                    for key in chosen:
                        del frontier[key]
                    removed += len(chosen)
                    error = bound
                    return frontier

                frontier = trim(frontier)
                for op in reversed(self._circuit.operations):
                    charge(1 + len(frontier) + len(op.qubits))
                    if not frontier or op.name == "barrier":
                        continue
                    support = sum(1 << q for q in op.qubits)
                    if all(not ((x | z) & support) for x, z in frontier):
                        continue
                    transitions = _transitions(
                        op, bindings, charge, self.max_work, self.max_memory_mb
                    )
                    output = {}
                    for (x, z), coefficient in frontier.items():
                        for key, factor in transitions(x, z):
                            charge(1)
                            output[key] = output.get(key, 0.0) + coefficient * factor
                        if len(output) > 2 * self._capacity:
                            output = trim(output)
                    peak = max(peak, len(output))
                    frontier = trim(output)
                value = math.fsum(c for (x, _), c in frontier.items() if not x)
                memo[initial] = value, error, peak, removed
            value, error, peak, removed = memo[initial]
            values.append(value)
            errors.append(error)
            peaks.append(peak)
            discarded.append(removed)
        arrays = np.asarray(values), np.asarray(errors)
        for array in arrays:
            array.setflags(write=False)
        return SparsePauliResult(
            *arrays,
            tuple(peaks),
            tuple(discarded),
            work,
            len({q for op in self._circuit.operations for q in op.qubits}),
            self.circuit.num_qubits,
        )


def _transitions(op, bindings, charge, max_work, max_memory_mb):
    name, wires = op.name, op.qubits
    if name in _CLIFFORD:
        table = _clifford_table(name)

        def clifford(x, z):
            key, sign = _clifford(x, z, wires, table)
            return ((key, sign),)

        return clifford
    if name in _ROTATIONS:
        angle = real(op.params[0].evaluate(bindings))
        gx, gz = _masks("".join(_ROTATIONS[name] for _ in wires))
        gx = sum(((gx >> i) & 1) << q for i, q in enumerate(wires))
        gz = sum(((gz >> i) & 1) << q for i, q in enumerate(wires))
        cosine, sine = math.cos(angle), math.sin(angle)

        def rotation(x, z):
            if ((gx & z).bit_count() + (gz & x).bit_count()) % 2 == 0:
                return (((x, z), 1.0),)
            return (
                ((x, z), cosine),
                ((x ^ gx, z ^ gz), sine * _rotation_sign(gx, gz, x, z)),
            )

        return rotation
    if name in CHANNEL_GATES - {"kraus"} or name in {"measure", "reset"}:
        p = real(op.params[0].evaluate(bindings)) if op.params else 0.0
        if not 0 <= p <= 1:
            raise ValidationError("Channel probability must lie in [0, 1].")
        bit = 1 << wires[0]

        def noise(x, z):
            digit = bool(x & bit) + 2 * bool(z & bit)
            key = (x, z)
            if not digit:
                return ((key, 1.0),)
            if name == "measure":
                return () if x & bit else ((key, 1.0),)
            if name == "reset":
                return () if x & bit else (((x, z ^ bit), 1.0),)
            if name == "amplitude_damping" and digit == 2:
                return ((key, 1 - p), ((x, z ^ bit), p))
            if name in {"amplitude_damping", "phase_damping"}:
                return ((key, math.sqrt(1 - p) if x & bit else 1.0),)
            if name == "depolarizing":
                return ((key, 1 - 4 * p / 3),)
            commutes = digit == (1 if name == "bit_flip" else 2)
            return ((key, 1.0 if commutes else 1 - 2 * p),)

        return noise
    # Arbitrary local unitaries/channels use an independently constructed Pauli
    # transfer map. The exponential LOCAL workspace is admitted before powers.
    arity = len(wires)
    exponent = 4 * arity
    capacity = min(max_work, int(memory_budget(max_memory_mb) * 2**20) // 64)
    if exponent >= capacity.bit_length():
        raise ResourceError("Local Pauli transfer expansion exceeds work/memory budget.")
    charge(1 << exponent)
    if name == "kraus":
        matrices = tuple(np.asarray(k, complex) for k in op.kraus)
    else:
        matrices = (
            _gate_matrix(name, [p.evaluate(bindings) for p in op.params], _TensorOps(), op.matrix),
        )
    basis = []
    for digits in product(range(4), repeat=arity):
        matrix = np.ones((1, 1), complex)
        for digit in digits:
            matrix = np.kron(matrix, (_I, _X, _Z, _Y)[digit])
        basis.append(matrix)
    basis = np.asarray(basis)
    cache = {}
    clear = sum(1 << q for q in wires)

    def transfer(x, z):
        local = 0
        for q in wires:
            local = (local << 2) | ((x >> q) & 1) | (((z >> q) & 1) << 1)
        if local not in cache:
            evolved = sum(
                get_compute_kernel().matmul(
                    get_compute_kernel().matmul(k.conj().T, basis[local]), k
                )
                for k in matrices
            )
            coefficients = get_compute_kernel().einsum("aij,ji->a", basis, evolved).real / len(
                evolved
            )
            cache[local] = tuple((i, float(c)) for i, c in enumerate(coefficients) if c)
            charge(len(coefficients) * len(evolved) ** 2 * max(1, len(matrices)))
        result = []
        for index, coefficient in cache[local]:
            xx, zz = x & ~clear, z & ~clear
            for q in reversed(wires):
                digit, index = index & 3, index >> 2
                xx |= (digit & 1) << q
                zz |= ((digit >> 1) & 1) << q
            result.append(((xx, zz), coefficient))
        return tuple(result)

    return transfer
