"""Correlated one/two-site locally optimal dual-frame classical shadows.

Independent pilot tomography selects disjoint correlated blocks. Positive,
regularized block states determine weighted dual frames; production samples
alone determine readouts. Every dual is checked against its POVM frame.
Research basis: https://arxiv.org/abs/2511.02555 (v2, June 2026).
This implementation uses bounded blocks and greedy pair matching, not the
paper's full graph-partitioning pipeline or its reported molecular benchmarks.
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from .backends import ExecutionOptions
from .errors import BackendError, ResourceError, ValidationError
from .ir import Circuit, integer
from .measurement import _rotate, measurement_plan
from .observables import Hamiltonian
from .parameters import real
from .simulator import _I, _X, _Y, _Z

_PAULIS = (_I, _X, _Y, _Z)


def _frame(width):
    local = np.zeros((6, 4))
    for axis in range(3):
        for bit in range(2):
            local[2 * axis + bit, 0] = 1 / 6
            local[2 * axis + bit, axis + 1] = (1 - 2 * bit) / 6
    result = np.ones((1, 1))
    for _ in range(width):
        result = np.kron(result, local)
    return result


def _pauli_basis(width):
    result = []
    for axes in product(range(4), repeat=width):
        matrix = np.ones((1, 1))
        for axis in axes:
            matrix = np.kron(matrix, _PAULIS[axis])
        result.append(matrix)
    return np.asarray(result)


def _density(data, qubits, regularization):
    basis = _pauli_basis(len(qubits))
    coefficients = []
    for axes in product(range(4), repeat=len(qubits)):
        scores = np.ones(len(data[0]))
        for q, axis in zip(qubits, axes, strict=True):
            if axis:
                scores *= (data[0][:, q] == axis - 1) * 3 * (1 - 2 * data[1][:, q].astype(float))
        coefficients.append(float(scores.mean()))
    rho = np.einsum("p,pij->ij", coefficients, basis) / (1 << len(qubits))
    rho = (rho + rho.conj().T) / 2
    eigenvalues, vectors = np.linalg.eigh(rho)
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues /= eigenvalues.sum()
    rho = (vectors * eigenvalues) @ vectors.conj().T
    return (1 - regularization) * rho + regularization * np.eye(len(rho)) / len(rho)


def _entropy(rho):
    values = np.linalg.eigvalsh(rho)
    values = values[values > 0]
    return float(-np.sum(values * np.log(values)))


def _dual(rho):
    width = len(rho).bit_length() - 1
    basis, frame = _pauli_basis(width), _frame(width)
    coordinates = np.einsum("pij,ji->p", basis, rho).real
    probability = frame @ coordinates
    if np.min(probability) <= 0:
        raise ValidationError("Dual-frame reference state must have positive POVM probabilities.")
    weighted = frame / probability[:, None]
    gram = frame.T @ weighted
    dual = np.linalg.solve(gram, weighted.T).T
    # Identity has an exact constant estimator, and retains this simpler column.
    dual[:, 0] = 1
    residual = float(np.max(np.abs(frame.T @ dual - np.eye(4**width))))
    if not np.isfinite(dual).all() or residual > 1e-9:
        raise ValidationError("Local dual frame failed its reconstruction check.")
    return dual, residual


def _collect(circuit, shots, stream, backend, options, max_settings, max_work):
    from .batch import execute_batch

    width = circuit.num_qubits
    if shots * width > max_work:
        raise ResourceError("Shadow basis/outcome storage exceeds max_work.")
    basis_stream, sample_stream = stream.spawn(2)
    rng = np.random.default_rng(basis_stream)
    bases = rng.integers(0, 3, size=(shots, width), dtype=np.int8)
    settings, labels = np.unique(bases, axis=0, return_inverse=True)
    if len(settings) > max_settings:
        raise ResourceError("Shadow acquisition exceeds max_settings.")
    outcomes = np.zeros_like(bases)
    streams = sample_stream.spawn(len(settings))
    offset = circuit.num_clbits
    for index, (setting, local_stream) in enumerate(zip(settings, streams, strict=True)):
        rows = np.flatnonzero(labels == index)
        rotated = _rotate(circuit, "".join("XYZ"[axis] for axis in setting))
        result = execute_batch(
            [rotated],
            backend=backend,
            shots=len(rows),
            seed=int(local_stream.generate_state(1)[0]),
            max_qubits=options.max_qubits,
            max_memory_mb=options.max_memory_mb,
            max_branches=options.max_branches,
            classical_bits=tuple(range(offset, offset + width)),
        )[0]
        if sum(result.counts.values()) != len(rows):
            raise BackendError("Shadow counts do not match the requested allocation.")
        cursor = 0
        for bits, count in result.counts.items():
            if len(bits) != width or set(bits) - {"0", "1"} or type(count) is not int or count < 0:
                raise BackendError("Invalid canonical shadow counts.")
            outcomes[rows[cursor : cursor + count]] = np.asarray(tuple(bits), dtype=np.int8)
            cursor += count
    return bases, outcomes


@dataclass(frozen=True)
class ShadowEstimate:
    values: np.ndarray
    covariance: np.ndarray
    blocks: tuple[tuple[int, ...], ...]
    shots: int
    pilot_shots: int
    frame_residual: float
    basis_settings: int
    _production: tuple = field(repr=False)
    _duals: tuple = field(repr=False)
    _max_work: int = field(repr=False)
    _max_memory_mb: float = field(repr=False)

    @property
    def standard_errors(self):
        return np.sqrt(np.maximum(0, np.diag(self.covariance)))

    def reestimate(self, observables):
        """Estimate additional Pauli observables from the retained production data."""
        return _readouts(
            self._production,
            observables,
            self.blocks,
            self._duals,
            self.shots,
            self.pilot_shots,
            self.frame_residual,
            self._max_work,
            self._max_memory_mb,
        )


def _readouts(
    production, observables, blocks, duals, shots, pilot_shots, residual, max_work, max_memory_mb
):
    observables = tuple(observables)
    plan = measurement_plan(observables)
    width, count = production[0].shape[1], len(production[0])
    if plan.num_qubits != width:
        raise ValidationError("Shadow observable width must match its samples.")
    terms = sum(len(o.terms) if isinstance(o, Hamiltonian) else 1 for o in observables)
    if terms * count * width > max_work:
        raise ResourceError("Shadow postprocessing exceeds max_work.")
    if (3 * count * len(observables) + 4 * len(observables) ** 2) * 8 > max_memory_mb * 2**20:
        raise ResourceError("Shadow postprocessing exceeds max_memory_mb.")
    indices = []
    for block in blocks:
        outcome = np.zeros(count, dtype=int)
        for q in block:
            outcome = outcome * 6 + 2 * production[0][:, q] + production[1][:, q]
        indices.append(outcome)
    scores = np.zeros((count, len(observables)))
    for column, observable in enumerate(observables):
        for term in observable.terms if isinstance(observable, Hamiltonian) else (observable,):
            value = np.full(count, term.coefficient)
            for block, dual, outcome in zip(blocks, duals, indices, strict=True):
                target = 0
                for q in block:
                    target = target * 4 + "IXYZ".index(term.paulis[q])
                value *= dual[outcome, target]
            scores[:, column] += value
    mean = scores.mean(axis=0)
    centered = scores - mean
    covariance = centered.T @ centered / (count * (count - 1))
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValidationError("Shadow readouts exceed finite precision.")
    settings = len(np.unique(production[0], axis=0))
    return ShadowEstimate(
        mean,
        covariance,
        blocks,
        shots,
        pilot_shots,
        residual,
        settings,
        production,
        tuple(duals),
        max_work,
        max_memory_mb,
    )


def shadow_estimate(
    circuit,
    observables,
    bindings=None,
    *,
    shots=1024,
    pilot_shots=128,
    block_size=2,
    regularization=0.05,
    backend="numpy",
    seed=None,
    max_settings=512,
    max_work=5_000_000,
    max_qubits=None,
    max_memory_mb=256,
    max_branches=256,
):
    """Use independent pilot-fitted block duals for unbiased Pauli readouts.

    shots includes pilot and production. block_size is one or two. With zero
    pilot shots, use canonical one-site shadows. The reported covariance is
    sample-based, conditional on the pilot, and excludes hardware bias.
    Different observables share the SAME production measurement records.
    """
    if not isinstance(circuit, Circuit):
        raise ValidationError("Shadow acquisition requires a Circuit.")
    shots, pilot_shots = integer(shots, "shots", 2), integer(pilot_shots, "pilot_shots")
    block_size = integer(block_size, "block_size", 1)
    integer(max_settings, "max_settings", 1)
    integer(max_work, "max_work", 1)
    regularization = real(regularization)
    if block_size > 2 or not 0 < regularization <= 1:
        raise ValidationError("Use block_size 1 or 2 and regularization in (0, 1].")
    if shots - pilot_shots < 2 or pilot_shots == 1:
        raise ValidationError("Require zero or >=2 pilot shots and >=2 production shots.")
    observables = tuple(observables)
    plan = measurement_plan(observables)
    if plan.num_qubits != circuit.num_qubits:
        raise ValidationError("Shadow observable width must match the circuit.")
    circuit = circuit.bind(bindings) if bindings is not None else circuit
    if circuit.parameters:
        raise ValidationError("Bind all parameters before shadow acquisition.")
    width = circuit.num_qubits
    terms = sum(len(o.terms) if isinstance(o, Hamiltonian) else 1 for o in observables)
    if terms * (shots - pilot_shots) * width > max_work:
        raise ResourceError("Shadow postprocessing exceeds max_work.")
    if shots * (width + len(observables)) + pilot_shots * width**2 * 16 > max_work:
        raise ResourceError("Shadow tomography/readout descriptions exceed max_work.")
    workspace = (
        16 * shots * width
        + 24 * (shots - pilot_shots) * len(observables)
        + 32 * len(observables) ** 2
        + 512 * width**2
        + 8192
    )
    if workspace > max_memory_mb * 2**20:
        raise ResourceError("Shadow records/readouts exceed max_memory_mb.")
    options = ExecutionOptions(max_qubits, max_memory_mb, seed, max_branches)
    if max_qubits is not None and width > max_qubits:
        raise ResourceError("Shadow acquisition exceeds max_qubits.")
    streams = np.random.SeedSequence(seed).spawn(2)
    singles, pairs, blocks = {}, {}, []
    if pilot_shots:
        pilot = _collect(circuit, pilot_shots, streams[0], backend, options, max_settings, max_work)
        singles = {q: _density(pilot, (q,), regularization) for q in range(width)}
        if block_size == 2:
            candidates = []
            for a in range(width):
                for b in range(a + 1, width):
                    rho = _density(pilot, (a, b), regularization)
                    tensor = rho.reshape(2, 2, 2, 2)
                    mutual = (
                        _entropy(np.trace(tensor, axis1=1, axis2=3))
                        + _entropy(np.trace(tensor, axis1=0, axis2=2))
                        - _entropy(rho)
                    )
                    pairs[a, b] = rho
                    candidates.append((-mutual, a, b))
            used = set()
            for _, a, b in sorted(candidates):
                if a not in used and b not in used:
                    blocks.append((a, b))
                    used.update((a, b))
    used = {q for block in blocks for q in block}
    blocks.extend((q,) for q in range(width) if q not in used)
    blocks = tuple(sorted(blocks))
    duals, residual = [], 0.0
    for block in blocks:
        rho = pairs[block] if len(block) == 2 else singles.get(block[0], np.eye(2) / 2)
        dual, error = _dual(rho)
        duals.append(dual)
        residual = max(residual, error)
    production = _collect(
        circuit, shots - pilot_shots, streams[1], backend, options, max_settings, max_work
    )
    for array in production:
        array.flags.writeable = False
    return _readouts(
        production,
        observables,
        blocks,
        duals,
        shots,
        pilot_shots,
        residual,
        max_work,
        max_memory_mb,
    )
