"""Exact, opt-in causal reduction for declared unitary expectation readouts."""

import json
from collections.abc import Mapping, Set
from dataclasses import asdict, dataclass, replace

from ..errors import Diagnostic, ResourceError, ValidationError, _as_tuple, conversion_error
from ..ir.compiler import CircuitStats, _check_circuit_limits, circuit_stats
from ..ir.core import Circuit
from ..ir.observables import Hamiltonian, PauliString


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

    def evaluate(self, bindings=None, *, absolute_error, backend=None, options=None, strict=True):
        """Evaluate this prepared causal cone under per-readout truncation tolerances."""
        from ..runtime.backends import ExecutionOptions
        from ._mps import MPSBackend

        backend = MPSBackend(layout="auto") if backend is None else backend
        options = ExecutionOptions() if options is None else options
        if not isinstance(backend, MPSBackend) or not isinstance(options, ExecutionOptions):
            raise ValidationError("Specify an MPSBackend and ExecutionOptions.")
        if options.max_qubits is not None and self.source_circuit.num_qubits > options.max_qubits:
            raise ResourceError("Certified source exceeds max_qubits.")
        _check_circuit_limits(self.source_circuit, max_operations=backend.max_operations)
        local = self._binding_values(bindings)
        circuit = self.circuit.bind(local) if local else self.circuit
        result = backend.readout(
            circuit, self.observables, options=options, absolute_error=absolute_error, strict=strict
        )
        return replace(result, simulated_wires=self.qubits)

    def _binding_values(self, bindings):
        if bindings is None:
            bindings = {}
        if not isinstance(bindings, Mapping):
            raise ValidationError("Supply a mapping of circuit parameters.")
        if self.source_circuit.parameters or bindings:
            self.source_circuit.bind(bindings)
        return {name: bindings[name] for name in self.circuit.parameters}

    def certified_vjp(self, bindings, cotangents, **options):
        """Return a parameter-shift vector–Jacobian product and truncation intervals."""
        from ..autodiff._certified import certified_vjp

        return certified_vjp(self, bindings, cotangents, **options)

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


def _causal_cone(circuit, live):
    """Return source indices and wires without copying immutable matrix payloads."""
    live, indices = set(live), []
    for index in range(len(circuit.operations) - 1, -1, -1):
        op = circuit.operations[index]
        if op.name != "barrier" and live.intersection(op.qubits):
            indices.append(index)
            live.update(op.qubits)
    return tuple(reversed(indices)), tuple(sorted(live))


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

    indices, qubits = _causal_cone(circuit, live)
    qubits = qubits or (0,)
    wire_map = {q: i for i, q in enumerate(qubits)}
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


def _readout_source(program, use_tape_outputs, source_options, backend, options, limit):
    from ..interop.program import QuantumProgram
    from ..ir.core import integer

    if source_options is not None and not isinstance(source_options, Mapping):
        raise ValidationError("source_options must be an importer option mapping.")
    p = program if isinstance(program, QuantumProgram) else QuantumProgram.capture(program)
    settings = dict(p.import_options) | dict(source_options or {})
    if p.framework == "ir" and settings:
        raise ValidationError("Native importer options require an SDK source.")
    if p.framework in {"qiskit", "cirq", "pennylane"}:
        requested = integer(
            settings.get("max_operations", backend.max_operations), "max_operations"
        )
        settings["max_operations"] = min(requested, backend.max_operations)
    if p.framework == "pennylane" and use_tape_outputs and getattr(p.native, "measurements", ()):
        from ..interop.adapters.pennylane import _expectation_readout

        return _expectation_readout(
            p.native,
            description_limit=limit,
            wire_limit=min(backend.max_work, options.max_qubits or backend.max_work),
            **settings,
        )
    return p.to_ir(**settings), None


def certified_readout(
    program,
    observables=None,
    *,
    classical_bits=None,
    bindings=None,
    source_options=None,
    absolute_error,
    backend=None,
    options=None,
    strict=True,
):
    """Bound declared expectations or P(classical bit=1) from imported or IR circuits.

    Terminal measurements dephase explicit quantum observables. Classical bits use
    their final measurement or constant assignment and zero initialization. PennyLane
    tapes can supply analytic Pauli expectations in native return order. The reported
    state distance applies to the reduced register identified by simulated_wires.
    """
    from ..compute.kernel import get_compute_kernel
    from ..ir.core import integer
    from ..runtime.backends import ExecutionOptions
    from ._mps import MPSBackend
    from .batch import _observables

    backend = MPSBackend(layout="auto") if backend is None else backend
    options = ExecutionOptions() if options is None else options
    if not isinstance(backend, MPSBackend) or not isinstance(options, ExecutionOptions):
        raise ValidationError("Certified readouts require an MPSBackend and ExecutionOptions.")
    budget = get_compute_kernel()._prepare_budget(options.max_memory_mb).max_bytes
    limit = min(backend.max_work, budget // 8)
    if isinstance(program, Circuit) and source_options is None:
        circuit = program
    else:
        circuit, native_outputs = _readout_source(
            program,
            observables is None and classical_bits is None,
            source_options,
            backend,
            options,
            limit,
        )
        if native_outputs is not None:
            observables = native_outputs
    _check_circuit_limits(circuit, max_operations=backend.max_operations)
    width = circuit.num_qubits
    if (options.max_qubits is not None and width > options.max_qubits) or width > backend.max_work:
        raise ResourceError("Certified source exceeds the configured wire or work budget.")
    if bindings is not None:
        circuit = circuit.bind(bindings)
    if circuit.parameters:
        raise ValidationError("Provide finite bindings for all circuit parameters.")
    if observables is not None and classical_bits is not None:
        raise ValidationError("Select quantum observables or classical output bits.")
    writes = {
        op.clbits[0]: op for op in circuit.operations if op.name in {"measure", "classical_set"}
    }
    unitary_input = circuit
    if any(op.name == "classical_set" for op in circuit.operations):
        unitary_input = replace(
            circuit, operations=tuple(op for op in circuit.operations if op.name != "classical_set")
        )
    pure = unitary_input.unitary_part() if unitary_input.has_effects else unitary_input
    if observables is None:
        if classical_bits is None and not writes:
            raise ValidationError("Specify observables or classical_bits.")
        bits = range(circuit.num_clbits) if classical_bits is None else classical_bits
        if isinstance(bits, (str, bytes, Set, Mapping)):
            raise ValidationError("classical_bits must be an ordered collection.")
        bits = _as_tuple(bits, "classical_bits", max_items=limit // width)
        identity = "I" * width
        obs = []
        for bit in bits:
            bit = integer(bit, "classical bit")
            if bit >= circuit.num_clbits:
                raise ValidationError("Classical bit exceeds the source register.")
            write = writes.get(bit)
            if write is None or write.name == "classical_set":
                obs.append(
                    PauliString(identity, 0.0 if write is None else write.params[0].constant)
                )
            else:
                q = write.qubits[0]
                z = identity[:q] + "Z" + identity[q + 1 :]
                obs.append(Hamiltonian((PauliString(identity, 0.5), PauliString(z, -0.5))))
    else:
        if isinstance(observables, Set):
            raise ValidationError("Readout observables require an ordered iterable.")
        obs = _observables(observables, (circuit,), max_items=limit // width)
    term_count = sum(len(o.terms) if isinstance(o, Hamiltonian) else 1 for o in obs)
    if width * term_count + len(circuit.operations) > limit:
        raise ResourceError("Certified readout descriptions exceed the work or memory budget.")
    if observables is not None and writes:
        measured = {op.qubits[0] for op in circuit.operations if op.name == "measure"}

        def dephase(term):
            return (
                PauliString("I" * width, 0)
                if any(term.paulis[q] in "XY" for q in measured)
                else term
            )

        obs = tuple(
            Hamiltonian(tuple(dephase(t) for t in o.terms))
            if isinstance(o, Hamiltonian)
            else dephase(o)
            for o in obs
        )
    plan = plan_readout(pure, obs, max_operations=backend.max_operations)
    return plan.evaluate(
        absolute_error=absolute_error, backend=backend, options=options, strict=strict
    )
