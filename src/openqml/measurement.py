"""Shared Pauli measurement plans, total shot budgets and covariance-aware estimates."""

from dataclasses import dataclass

import numpy as np

from .backends import BackendCapabilities, ExecutionOptions, get_backend
from .errors import BackendError, ValidationError
from .ir import integer
from .observables import Hamiltonian, PauliString


@dataclass(frozen=True)
class MeasurementGroup:
    basis: str
    paulis: tuple[str, ...]
    coefficients: tuple[tuple[float, ...], ...]  # observables by terms


@dataclass(frozen=True)
class MeasurementPlan:
    num_qubits: int
    num_observables: int
    constants: tuple[float, ...]
    groups: tuple[MeasurementGroup, ...]

    def allocate(self, shots):
        """Distribute a total per-input-circuit budget, with >=2 shots per group."""
        integer(shots, "shots", 1)
        if shots > np.iinfo(np.int64).max:
            raise ValidationError("shots must fit in a signed 64-bit integer.")
        count = len(self.groups)
        if not count:
            return ()
        if shots < 2 * count:
            raise ValidationError(f"At least {2 * count} shots are needed for {count} groups.")
        importance = np.asarray([np.abs(g.coefficients).sum() for g in self.groups])
        quota = (shots - 2 * count) * importance / importance.sum()
        allocation = np.floor(quota).astype(np.int64) + 2
        remainder = shots - int(allocation.sum())
        order = np.argsort(-(quota - np.floor(quota)), kind="stable")
        allocation[order[:remainder]] += 1
        return tuple(int(n) for n in allocation)


@dataclass(frozen=True)
class EstimateResult:
    values: np.ndarray  # batch by observables
    covariance: np.ndarray  # covariance of the estimated means, batch by observables^2
    shots_per_group: tuple[int, ...]
    plan: MeasurementPlan

    @property
    def standard_errors(self):
        return np.sqrt(np.maximum(0, np.diagonal(self.covariance, axis1=-2, axis2=-1)))

    @property
    def shots_per_circuit(self):
        return sum(self.shots_per_group)

    @property
    def execution_count(self):
        return self.values.shape[0] * len(self.plan.groups)


def measurement_plan(observables):
    """Greedy qubit-wise commuting groups shared across every output observable.

    Equal Pauli words are measured once even when reused in many Hamiltonians.
    Identity contributions are analytic and consume no shots. Grouping preserves
    covariance between outputs reconstructed from the same measurement record.
    """
    observables = tuple(observables)
    if not observables:
        raise ValidationError("A measurement plan needs at least one observable.")
    words, width = {}, None
    for index, observable in enumerate(observables):
        terms = observable.terms if isinstance(observable, Hamiltonian) else (observable,)
        for term in terms:
            if not isinstance(term, PauliString):
                raise ValidationError("Measurement plans require Pauli observables.")
            if width is not None and len(term.paulis) != width:
                raise ValidationError("All measured observables must share a width.")
            width = len(term.paulis)
            words.setdefault(term.paulis, np.zeros(len(observables)))[index] += term.coefficient
    constants = words.pop("I" * width, np.zeros(len(observables)))
    words = {word: coefficients for word, coefficients in words.items() if np.any(coefficients)}
    groups = []
    for word in sorted(words, key=lambda w: (-np.abs(words[w]).sum(), w)):
        for group in groups:
            if all(a == "I" or b == "I" or a == b for a, b in zip(group[0], word, strict=True)):
                group[0] = "".join(
                    b if a == "I" else a for a, b in zip(group[0], word, strict=True)
                )
                group[1].append(word)
                break
        else:
            groups.append([word, [word]])
    return MeasurementPlan(
        width,
        len(observables),
        tuple(constants),
        tuple(
            MeasurementGroup(
                basis,
                tuple(paulis),
                tuple(tuple(row) for row in np.stack([words[p] for p in paulis], axis=1)),
            )
            for basis, paulis in groups
        ),
    )


def _rotate(circuit, basis):
    offset = circuit.num_clbits
    for wire, pauli in enumerate(basis):
        if pauli == "Y":
            circuit = circuit.sdg(wire).h(wire)
        elif pauli == "X":
            circuit = circuit.h(wire)
    for wire in range(circuit.num_qubits):
        circuit = circuit.measure(wire, offset + wire)
    return circuit


def _statistics(counts, group, width, shots):
    coefficients = np.asarray(group.coefficients)
    first = np.zeros(coefficients.shape[0])
    second = np.zeros((coefficients.shape[0], coefficients.shape[0]))
    if sum(counts.values()) != shots:
        raise BackendError("Measurement counts do not match the requested shot allocation.")
    for bits, count in counts.items():
        if (
            not isinstance(bits, str)
            or len(bits) != width
            or set(bits) - {"0", "1"}
            or isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or count < 0
        ):
            raise BackendError("A sampler returned invalid canonical measurement counts.")
        signs = np.asarray(
            [
                (-1) ** sum(int(b) for b, p in zip(bits, word, strict=True) if p != "I")
                for word in group.paulis
            ]
        )
        scores = coefficients @ signs
        first += count * scores
        second += count * np.outer(scores, scores)
    mean = first / shots
    covariance = (second - shots * np.outer(mean, mean)) / (shots * (shots - 1))
    return mean, covariance


def estimate(
    circuits,
    observables,
    bindings=None,
    *,
    shots=1024,
    backend="numpy",
    seed=None,
    max_qubits=None,
    max_memory_mb=256,
    max_branches=256,
):
    """Sample grouped readouts with a total shot budget per original input circuit.

    The returned covariance includes shared-term correlations within each group.
    Independent random streams are used between groups. Standard errors are
    sample-based uncertainty estimates, not guaranteed confidence bounds.
    Only final readout-bit marginals are requested; old program records remain
    available until feedback has consumed them.
    """
    from .batch import _bound, execute_batch

    ExecutionOptions(max_qubits, max_memory_mb, seed, max_branches)
    get_backend(backend)
    plan = measurement_plan(observables)
    allocation = plan.allocate(shots)
    circuits = _bound(circuits, bindings)
    if any(c.num_qubits != plan.num_qubits for c in circuits):
        raise ValidationError("Estimate requires circuits of the observable width.")
    values = np.broadcast_to(plan.constants, (len(circuits), plan.num_observables)).copy()
    covariance = np.zeros((len(circuits), plan.num_observables, plan.num_observables))
    streams = np.random.SeedSequence(seed).spawn(len(plan.groups))
    for group, count, stream in zip(plan.groups, allocation, streams, strict=True):
        offsets = {}
        for row, c in enumerate(circuits):
            offsets.setdefault(c.num_clbits, []).append((row, _rotate(c, group.basis)))
        # Readout marginals do not need old program records once feedback has
        # consumed them. Group differing classical widths to preserve batching.
        substreams = stream.spawn(len(offsets)) if len(offsets) > 1 else (stream,)
        for (offset, items), substream in zip(offsets.items(), substreams, strict=True):
            results = execute_batch(
                [c for _, c in items],
                backend=backend,
                shots=count,
                seed=int(substream.generate_state(1)[0]),
                max_qubits=max_qubits,
                max_memory_mb=max_memory_mb,
                max_branches=max_branches,
                classical_bits=tuple(range(offset, offset + plan.num_qubits)),
            )
            for (row, _), result in zip(items, results, strict=True):
                mean, cov = _statistics(result.counts, group, plan.num_qubits, count)
                values[row] += mean
                covariance[row] += cov
    return EstimateResult(values, covariance, allocation, plan)


@dataclass(frozen=True)
class ShotBackend:
    """Use any registered sampler for quantum-layer forwards and shift-rule backwards."""

    backend: object = "numpy"
    shots: int = 1024
    seed: int | None = None
    name: str = "shot-estimator"

    def __post_init__(self):
        integer(self.shots, "shots", 1)
        ExecutionOptions(seed=self.seed)
        get_backend(self.backend)

    @property
    def capabilities(self):
        native = get_backend(self.backend).capabilities
        return BackendCapabilities(
            gates=native.gates,
            max_qubits=native.max_qubits,
            native_expectations=True,
            channels=native.channels,
            reset=native.reset,
            classical_control=native.classical_control,
            measurement_expectations=True,
        )

    def expectations_batch(self, circuits, observables, *, options: ExecutionOptions):
        return estimate(
            circuits,
            observables,
            backend=self.backend,
            shots=self.shots,
            seed=self.seed if options.seed is None else options.seed,
            max_qubits=options.max_qubits,
            max_memory_mb=options.max_memory_mb,
            max_branches=options.max_branches,
        ).values

    def prepare_gradient_circuit(self, circuit):
        native = get_backend(self.backend)
        prepare = getattr(native, "prepare_gradient_circuit", None)
        return circuit if prepare is None else prepare(circuit)
