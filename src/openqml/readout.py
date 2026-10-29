"""Exact, opt-in causal reduction for declared unitary expectation readouts."""

import json
from collections.abc import Mapping, Set
from dataclasses import asdict, dataclass, replace

from .compiler import CircuitStats, _check_circuit_limits, circuit_stats
from .errors import Diagnostic, ValidationError, conversion_error
from .ir import Circuit
from .observables import Hamiltonian, PauliString


@dataclass(frozen=True, eq=False)
class ReadoutPlan:
    """A reduced program and its readouts, with source wire/operation provenance.

    Equivalence covers only these real expectation values from the all-zero
    initial state. The reduced circuit is not a full-program replacement.
    qubits maps reduced wire indices to source indices. Identity-only outputs
    retain source wire zero because the IR requires at least one quantum wire.
    No bindings, quantum states, SDK objects or approximate thresholds are used.
    """

    source_circuit: Circuit
    source_observables: tuple
    circuit: Circuit
    observables: tuple
    qubits: tuple[int, ...]
    operation_indices: tuple[int, ...]
    removed_parameters: tuple[str, ...]
    before: CircuitStats
    after: CircuitStats

    @property
    def diagnostics(self):
        return (
            Diagnostic(
                "DECLARED_EXPECTATIONS_ONLY",
                "Equivalence covers the declared readouts from the all-zero initial state; "
                "state vectors, samples and other readouts require the source circuit.",
                severity="info",
            ),
            Diagnostic(
                "READOUT_CAUSAL_REDUCTION",
                f"Retained {self.after.num_qubits} of {self.before.num_qubits} wires and "
                f"{len(self.operation_indices)} of {len(self.source_circuit.operations)} "
                "operations. Global phase and unused classical storage are omitted.",
                severity="info",
            ),
            *(
                Diagnostic(
                    "READOUT_PARAMETER_ELIMINATED",
                    f"{name!r} does not affect the declared expectation values.",
                    "parameters",
                    severity="info",
                )
                for name in self.removed_parameters
            ),
        )

    def to_dict(self):
        def term(value):
            return {"paulis": value.paulis, "coefficient": value.coefficient}

        return {
            "semantics": "declared_expectations",
            "initial_state": "all_zero",
            "qubits": list(self.qubits),
            "operation_indices": list(self.operation_indices),
            "removed_parameters": list(self.removed_parameters),
            "before": asdict(self.before),
            "after": asdict(self.after),
            "observables": [
                {"type": "hamiltonian", "terms": [term(p) for p in ob.terms]}
                if isinstance(ob, Hamiltonian)
                else {"type": "pauli", **term(ob)}
                for ob in self.observables
            ],
            "diagnostics": [asdict(d) for d in self.diagnostics],
        }

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


def plan_readout(circuit, observables, *, max_operations=100_000):
    """Trace a conservative backward cone, then compact its wires and readouts.

    A retained gate makes every operand live before that gate. Disjoint gates
    occurring later can still be removed, including gates on an earlier live
    wire. Multiple Pauli/Hamiltonian outputs share one union cone. Zero-weight
    terms do not seed the cone; coefficients, term order and output order stay
    intact. Channels, measurements and classical feedback fail explicitly.
    Full-program conversion never invokes this pass implicitly.
    """
    _check_circuit_limits(circuit, max_operations=max_operations)
    if circuit.has_effects:
        raise conversion_error(
            "READOUT_EFFECTS_UNSUPPORTED",
            "Readout reduction requires a unitary circuit without channels, "
            "measurements or feedback.",
        )
    if isinstance(observables, (str, bytes, Mapping, Set)):
        raise ValidationError("Readout observables require an ordered iterable.")
    try:
        obs = tuple(observables)
    except TypeError as exc:
        raise ValidationError("Readout observables require an ordered iterable.") from exc
    if not obs:
        raise ValidationError("Readout reduction requires at least one observable.")
    live = set()
    for ob in obs:
        for term in ob.terms if isinstance(ob, Hamiltonian) else (ob,):
            if not isinstance(term, PauliString) or len(term.paulis) != circuit.num_qubits:
                raise ValidationError("Readout observable width must match the circuit.")
            if term.coefficient:
                live.update(q for q, pauli in enumerate(term.paulis) if pauli != "I")

    indices = []
    for index in range(len(circuit.operations) - 1, -1, -1):
        op = circuit.operations[index]
        if op.name != "barrier" and live.intersection(op.qubits):
            indices.append(index)
            live.update(op.qubits)
    qubits = tuple(sorted(live)) or (0,)
    wire_map = {q: i for i, q in enumerate(qubits)}
    indices = tuple(reversed(indices))
    operations = []
    for index in indices:
        op = circuit.operations[index]
        mapped = tuple(wire_map[q] for q in op.qubits)
        operations.append(op if mapped == op.qubits else replace(op, qubits=mapped))

    def compact(term):
        # Zero terms may name wires outside the cone; keep their zero value.
        paulis = "".join(term.paulis[q] for q in qubits) if term.coefficient else "I" * len(qubits)
        return PauliString(paulis, term.coefficient)

    reduced = Circuit(len(qubits), tuple(operations))
    return ReadoutPlan(
        circuit,
        obs,
        reduced,
        tuple(
            Hamiltonian(tuple(compact(t) for t in ob.terms))
            if isinstance(ob, Hamiltonian)
            else compact(ob)
            for ob in obs
        ),
        qubits,
        indices,
        tuple(sorted(set(circuit.parameters) - set(reduced.parameters))),
        circuit_stats(circuit),
        circuit_stats(reduced),
    )
