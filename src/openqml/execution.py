"""Bounded exact expectation planning over causal cones and product components.

Components are separated by circuit connectivity, never by numeric thresholds
or an assumed absence of entanglement. Each local query uses either symbolic
Pauli arithmetic or dense simulation. Products use multiplication, not division,
so zero local expectations retain their correct derivatives.
"""

import json
import math
from collections.abc import Mapping, Set
from dataclasses import asdict, dataclass, field, replace

from ._scheduling import _evaluate_dense, _schedule_dense
from .compiler import _check_circuit_limits, circuit_stats
from .errors import ConversionError, Diagnostic, ResourceError, ValidationError, conversion_error
from .ir import Circuit, integer
from .observables import Hamiltonian, PauliString
from .parameters import Parameter, real
from .pauli import _UNARY, PauliPlan, _Arena, _evaluate_program, _prepare_bindings
from .readout import _causal_cone


def _failure(code, message, location="circuit"):
    return conversion_error(
        code, message, location, "Inspect the plan limits or select another exact execution method."
    )


def _program_work(nodes):
    return sum(8 if n[0] in _UNARY | {"pow", "div"} else 1 for n in nodes)


def _reference_work(circuit, outputs, limit):
    """Bound the unfactored reference, including expanded expression-tree work."""
    if circuit.num_qubits >= limit.bit_length():
        raise ResourceError("Source verification exceeds max_reference_work.")
    work = (1 << circuit.num_qubits) * (
        outputs + sum(1 << len(op.qubits) for op in circuit.operations if op.name != "barrier")
    )
    memo = {}
    expressions = [circuit.global_phase, *(p for op in circuit.operations for p in op.params)]
    for root in expressions:
        pending = [(root, False)]
        while pending:
            current, ready = pending.pop()
            if id(current) in memo:
                continue
            if len(memo) + len(pending) > limit:
                raise ResourceError("Source verification exceeds max_reference_work.")
            if current.op is None:
                value, depth = 1 + 2 * len(current.terms), 0
            elif not ready:
                pending.append((current, True))
                pending.extend((arg, False) for arg in reversed(current.args))
                continue
            else:
                value = min(limit + 1, 1 + sum(memo[id(arg)][0] for arg in current.args))
                depth = 1 + max(memo[id(arg)][1] for arg in current.args)
            if depth > 128:
                raise ResourceError("Source verification expressions exceed the nesting ceiling.")
            memo[id(current)] = value, depth
        work += memo[id(root)][0]
        if work > limit:
            raise ResourceError("Source verification exceeds max_reference_work.")


def _import_program(arena, nodes, roots):
    mapping = []
    for node in nodes:
        if node[0] in {"const", "param"}:
            index = arena.make(node)
        elif node[0] in _UNARY:
            index = arena.trig(node[0], mapping[node[1]])
        else:
            index = arena.binary(node[0], mapping[node[1]], mapping[node[2]])
        mapping.append(index)
    return tuple(mapping[i] for i in roots)


def _reuse_factors(nodes, roots, groups, batches, factor_count, max_nodes):
    """Intern output arithmetic again after exact quantum functions are shared."""
    aliases, coefficients = list(range(factor_count)), {}
    for group in groups:
        if group.pauli_plan is not None:
            for slot, factor in zip(group.program_indices, group.factor_indices, strict=True):
                aliases[factor] = coefficients.setdefault(slot, factor)
    for batch in batches:
        for destinations in batch.factor_maps:
            readouts = {}
            for factor, readout in destinations:
                aliases[factor] = readouts.setdefault(readout, factor)
    arena, mapping = _Arena(max_nodes), []
    arena.location = "shared_outputs"
    for node in nodes:
        if node[0] == "factor":
            index = arena.make(("factor", aliases[node[1]]))
        elif node[0] == "const":
            index = arena.make(node)
        else:
            index = arena.binary(node[0], mapping[node[1]], mapping[node[2]])
        mapping.append(index)
    nodes, roots = arena.compact(tuple(mapping[i] for i in roots))
    return nodes, roots, tuple(aliases)


def _components(circuit, indices):
    parent = {}

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    for index in indices:
        qubits = circuit.operations[index].qubits
        for q in qubits:
            parent.setdefault(q, q)
        root = find(qubits[0])
        for q in qubits[1:]:
            parent[find(q)] = root
    wires, operations = {}, {}
    for q in sorted(parent):
        wires.setdefault(find(q), []).append(q)
    for index in indices:
        root = find(circuit.operations[index].qubits[0])
        operations.setdefault(root, []).append(index)
    return parent, tuple((tuple(wires[root]), tuple(operations[root])) for root in wires)


def _materialize(source, qubits, indices):
    mapping, operations = {q: i for i, q in enumerate(qubits)}, []
    for index in indices:
        op = source.operations[index]
        target = tuple(mapping[q] for q in op.qubits)
        operations.append(op if target == op.qubits else replace(op, qubits=target))
    return Circuit(len(qubits), tuple(operations))


@dataclass(frozen=True, eq=False)
class _DenseProgram:
    circuit: Circuit
    names: tuple
    nodes: tuple
    roots: tuple
    active_parameters: tuple
    construction_nodes: int


def _compile_dense(circuit, max_nodes):
    arena, names, operations = _Arena(max_nodes), {}, []
    arena.location = "parameters"
    for op in circuit.operations:
        params = []
        for angle in op.params:
            root = arena.angle(angle)
            name = names.setdefault(root, f"_planned_angle_{len(names)}")
            params.append(Parameter(name))
        operations.append(replace(op, params=tuple(params)) if params else op)
    nodes, roots = arena.compact(tuple(names))
    active = tuple(sorted({n[1] for n in nodes if n[0] == "param"}))
    return _DenseProgram(
        Circuit(circuit.num_qubits, tuple(operations)),
        tuple(names.values()),
        nodes,
        roots,
        active,
        len(arena.nodes),
    )


@dataclass(frozen=True, eq=False)
class ExecutionGroup:
    """One shared local circuit with source provenance and an exact chosen engine."""

    circuit: Circuit
    observables: tuple
    source_qubits: tuple[int, ...]
    source_operations: tuple[int, ...]
    factor_indices: tuple[int, ...]
    method: str
    estimated_work: int
    working_bytes_per_row: int
    pauli_work: int | None
    dense_work: int | None
    diagnostics: tuple
    pauli_plan: PauliPlan | None = field(repr=False)
    dense_program: _DenseProgram | None = field(repr=False)
    program_indices: tuple = ()

    @property
    def active_parameters(self):
        return (
            self.dense_program.active_parameters
            if self.pauli_plan is None
            else self.pauli_plan.active_parameters
        )

    def to_dict(self):
        return {
            "method": self.method,
            "source_qubits": list(self.source_qubits),
            "source_operations": list(self.source_operations),
            "factor_indices": list(self.factor_indices),
            "readouts": [p.paulis for p in self.observables],
            "resources": asdict(circuit_stats(self.circuit)),
            "estimated_work": self.estimated_work,
            "working_bytes_per_row": self.working_bytes_per_row,
            "candidate_work": {"pauli": self.pauli_work, "statevector": self.dense_work},
            "coefficient_slots": list(self.program_indices),
            "diagnostics": [asdict(d) for d in self.diagnostics],
        }


@dataclass(frozen=True)
class ExpectationCheck:
    """Numerical equivalence result for one concrete binding, without model values."""

    matched: bool
    reference_backend: str
    max_absolute_error: float
    mismatched_outputs: tuple[int, ...]
    atol: float
    rtol: float

    def to_dict(self):
        return {"scope": "declared_expectations_at_bindings", **asdict(self)}

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


@dataclass
class _Draft:
    qubits: tuple
    indices: tuple
    circuit: Circuit
    readouts: dict = field(default_factory=dict)


class _Builder:
    def __init__(self, plan):
        self.plan, self.work = plan, 0
        self.drafts, self.factors, self.queries = {}, 0, {}
        self.arena = _Arena(plan.max_nodes)
        self.arena.location = "outputs"

    def charge(self, amount):
        if amount > self.plan.max_planning_work - self.work:
            raise _failure("EXPECTATION_PLANNING_LIMIT", "Planning exceeds max_planning_work.")
        self.work += amount

    def query(self, label):
        if label in self.queries:
            return self.queries[label]
        source = self.plan.circuit
        self.charge(len(label) + len(source.operations))
        indices, _ = _causal_cone(source, (q for q, p in enumerate(label) if p != "I"))
        self.charge(sum(len(source.operations[i].qubits) for i in indices))
        parent, components = _components(source, indices)
        if any(p in "XY" and q not in parent for q, p in enumerate(label)):
            self.queries[label] = self.arena.zero
            return self.arena.zero
        values = []
        for qubits, operations in components:
            local = "".join(label[q] for q in qubits)
            if set(local) == {"I"}:
                continue
            key = qubits, operations
            if key not in self.drafts:
                if len(self.drafts) >= self.plan.max_groups:
                    raise _failure("EXPECTATION_GROUP_LIMIT", "Planning exceeds max_groups.")
                self.charge(len(qubits) + len(operations))
                self.drafts[key] = _Draft(
                    qubits, operations, _materialize(source, qubits, operations)
                )
            draft = self.drafts[key]
            if local not in draft.readouts:
                draft.readouts[local] = self.factors
                self.factors += 1
            values.append(self.arena.make(("factor", draft.readouts[local])))
        while len(values) > 1:
            values = [
                self.arena.binary("mul", values[i], values[i + 1])
                for i in range(0, len(values) - 1, 2)
            ] + (values[-1:] if len(values) % 2 else [])
        result = values[0] if values else self.arena.one
        self.queries[label] = result
        return result

    def outputs(self, values):
        if isinstance(values, (str, bytes, Mapping, Set)):
            raise ValidationError("Planned observables require an ordered iterable.")
        try:
            iterator = iter(values)
        except TypeError as exc:
            raise ValidationError("Planned observables require an ordered iterable.") from exc
        observables, roots, count = [], [], 0
        for ob in iterator:
            if len(observables) >= self.plan.max_nodes:
                raise _failure("EXPECTATION_NODE_LIMIT", "Output count exceeds max_nodes.")
            observables.append(ob)
            terms = []
            for term in ob.terms if isinstance(ob, Hamiltonian) else (ob,):
                count += 1
                if count > self.plan.max_nodes:
                    raise _failure("EXPECTATION_NODE_LIMIT", "Source term count exceeds max_nodes.")
                if (
                    not isinstance(term, PauliString)
                    or len(term.paulis) != self.plan.circuit.num_qubits
                ):
                    raise ValidationError("Planned observable width must match the circuit.")
                if term.coefficient:
                    value = self.arena.binary(
                        "mul", self.arena.const(term.coefficient), self.query(term.paulis)
                    )
                    terms.append(value)
            roots.append(self.arena.total(terms))
        if not observables:
            raise ValidationError("Expectation planning requires observables.")
        nodes, roots = self.arena.compact(roots)
        return tuple(observables), nodes, roots

    def select(self, draft, reserve):
        plan, circuit = self.plan, draft.circuit
        readouts = tuple(PauliString(p) for p in draft.readouts)
        notes, dense_work, dense_bytes, dense_program = [], None, None, None
        available = plan._memory_bytes - reserve
        width = circuit.num_qubits
        if width <= plan.max_dense_qubits and width < 63:
            local = max((len(op.qubits) for op in circuit.operations), default=1)
            dense_bytes = 64 * (1 << width) + 32 * (1 << (2 * local))
            if dense_bytes <= available:
                remaining = plan.max_planning_work - self.work
                allowance = min(plan.max_nodes, max(0, remaining - 2 * len(circuit.operations)))
                if allowance >= 2:
                    try:
                        dense_program = _compile_dense(circuit, allowance)
                    except ConversionError as exc:
                        self.charge(2 * len(circuit.operations) + allowance)
                        notes.extend(
                            replace(d, code="DENSE_PARAMETER_LIMIT", severity="info")
                            for d in exc.diagnostics
                        )
                    else:
                        self.charge(2 * len(circuit.operations) + dense_program.construction_nodes)
                        dense_bytes += 8 * (len(dense_program.nodes) + len(dense_program.roots) + 8)
                        if dense_bytes <= available:
                            dense_work = (
                                2
                                * (1 << width)
                                * (
                                    len(readouts)
                                    + sum(1 << len(op.qubits) for op in circuit.operations)
                                )
                            )
                            dense_work += sum(
                                8 if n[0] in _UNARY | {"pow", "div"} else 1
                                for n in dense_program.nodes
                            )
                        else:
                            notes.append(
                                Diagnostic(
                                    "DENSE_WORKING_LIMIT",
                                    "Prepared angle arrays exceed the memory budget.",
                                    severity="info",
                                )
                            )
                else:
                    notes.append(
                        Diagnostic(
                            "DENSE_SEARCH_LIMIT",
                            "Remaining planning work cannot fund dense parameter preparation.",
                            severity="info",
                        )
                    )
            else:
                notes.append(
                    Diagnostic(
                        "DENSE_WORKING_LIMIT",
                        "Local dense working arrays exceed the memory budget.",
                        severity="info",
                    )
                )
        else:
            notes.append(
                Diagnostic(
                    "DENSE_WIDTH_LIMIT",
                    "Local circuit exceeds max_dense_qubits or the array index ceiling.",
                    severity="info",
                )
            )
        candidate, pauli_work = None, None
        remaining = plan.max_planning_work - self.work
        fixed = 3 * len(circuit.operations) + len(readouts)
        allowance = max(0, remaining - fixed)
        work_limit = min(plan.max_work, allowance // 2)
        node_limit = min(plan.max_nodes, allowance - work_limit)
        opaque = next(
            (
                i
                for i, op in enumerate(circuit.operations)
                if op.name == "unitary" and len(op.qubits) > 1
            ),
            None,
        )
        if opaque is not None:
            self.charge(len(circuit.operations))
            notes.append(
                Diagnostic(
                    "DECOMPOSITION_UNAVAILABLE",
                    "Pauli execution requires unavailable multi-qubit matrix lowering.",
                    f"operations[{opaque}]",
                    severity="info",
                )
            )
        elif work_limit >= 1 and node_limit >= 2:
            try:
                candidate = PauliPlan(
                    circuit,
                    readouts,
                    max_terms=plan.max_terms,
                    max_nodes=node_limit,
                    max_work=work_limit,
                    max_operations=plan.max_operations,
                    max_memory_mb=plan.max_memory_mb,
                )
            except ConversionError as exc:
                self.charge(fixed + work_limit + node_limit)
                notes.extend(replace(d, severity="info") for d in exc.diagnostics)
            else:
                self.charge(fixed + candidate.propagation_work + candidate.construction_nodes)
                if (len(candidate._nodes) + 8) * 8 <= available:
                    pauli_work = sum(
                        8 if n[0] in _UNARY | {"pow", "div"} else 1 for n in candidate._nodes
                    )
                else:
                    notes.append(
                        Diagnostic(
                            "PAULI_WORKING_LIMIT",
                            "Local Pauli arithmetic exceeds the memory budget.",
                            severity="info",
                        )
                    )
        else:
            notes.append(
                Diagnostic(
                    "PAULI_SEARCH_LIMIT",
                    "Remaining planning work cannot fund another Pauli candidate.",
                    severity="info",
                )
            )
        if pauli_work is None and dense_work is None:
            raise ConversionError(
                (
                    Diagnostic(
                        "NO_EXACT_EXECUTION",
                        "No exact local execution method fits the configured limits.",
                        hint="Inspect candidate diagnostics and adjust explicit limits.",
                    ),
                    *notes,
                )
            )
        use_pauli = pauli_work is not None and (dense_work is None or pauli_work <= dense_work)
        return ExecutionGroup(
            circuit,
            readouts,
            draft.qubits,
            draft.indices,
            tuple(draft.readouts.values()),
            "pauli" if use_pauli else "statevector",
            pauli_work if use_pauli else dense_work,
            (len(candidate._nodes) + 8) * 8 if use_pauli else dense_bytes,
            pauli_work,
            dense_work,
            tuple(notes),
            candidate if use_pauli else None,
            None if use_pauli else dense_program,
        )


@dataclass(frozen=True, eq=False)
class ExpectationPlan:
    """Factor declared expectations, share local queries, select bounded exact engines.

    Equivalence covers the declared readouts from the all-zero initial state.
    Selection uses a deterministic static scalar-work heuristic, not a runtime
    speed guarantee. No bindings, states or framework tensors are retained.
    Pauli limits apply per candidate; max_planning_work bounds aggregate search,
    charging failed candidates their reserved allowance. Source exports remain
    full programs. Channels, measurements and feedback are rejected explicitly.
    """

    circuit: Circuit
    observables: tuple
    max_groups: int = 256
    max_dense_qubits: int = 12
    max_planning_work: int = 10_000_000
    max_terms: int = 4096
    max_nodes: int = 10_000
    max_work: int = 100_000
    max_operations: int = 100_000
    max_memory_mb: float = 256
    max_dense_batch: int = 32
    groups: tuple = field(init=False)
    dense_batches: tuple = field(init=False)
    active_parameters: tuple = field(init=False)
    planning_work: int = field(init=False)
    factor_count: int = field(init=False)
    factor_aliases: tuple = field(init=False)
    working_bytes_per_row: int = field(init=False)
    _nodes: tuple = field(init=False, repr=False)
    _roots: tuple = field(init=False, repr=False)
    _coefficient_nodes: tuple = field(init=False, repr=False)
    _coefficient_roots: tuple = field(init=False, repr=False)
    _source_parameters: frozenset = field(init=False, repr=False)
    _retained_bytes_per_row: int = field(init=False, repr=False)
    _memory_bytes: int = field(init=False, repr=False)

    def __post_init__(self):
        for name in (
            "max_groups",
            "max_dense_qubits",
            "max_planning_work",
            "max_terms",
            "max_nodes",
            "max_work",
            "max_operations",
            "max_dense_batch",
        ):
            minimum = 0 if name == "max_dense_qubits" else 2 if name == "max_nodes" else 1
            object.__setattr__(self, name, integer(getattr(self, name), name, minimum))
        budget = real(self.max_memory_mb)
        if budget <= 0:
            raise ValidationError("max_memory_mb must be positive.")
        object.__setattr__(self, "max_memory_mb", budget)
        numerator, denominator = budget.as_integer_ratio()
        object.__setattr__(self, "_memory_bytes", numerator * 2**20 // denominator)
        _check_circuit_limits(self.circuit, max_operations=self.max_operations)
        if self.circuit.has_effects:
            raise _failure(
                "EXPECTATION_EFFECTS_UNSUPPORTED",
                "Planned execution requires a unitary circuit without effects.",
            )
        builder = _Builder(self)
        try:
            obs, nodes, roots = builder.outputs(self.observables)
        except ConversionError as exc:
            if any(d.code == "PAULI_NODE_LIMIT" for d in exc.diagnostics):
                raise ConversionError(
                    tuple(replace(d, code="EXPECTATION_NODE_LIMIT") for d in exc.diagnostics)
                ) from exc
            raise
        reserve = 8 * (len(nodes) + builder.factors + 8)
        if reserve > self._memory_bytes:
            raise _failure(
                "EXPECTATION_WORKING_LIMIT", "Readout contraction arrays exceed the memory budget."
            )
        selected = tuple(builder.select(draft, reserve) for draft in builder.drafts.values())
        shared, slots, groups = _Arena(self.max_nodes), {}, []
        shared.location = "shared_coefficients"
        for group in selected:
            program = group.pauli_plan if group.pauli_plan is not None else group.dense_program
            source_nodes = program._nodes if group.pauli_plan is not None else program.nodes
            source_roots = program._roots if group.pauli_plan is not None else program.roots
            builder.charge(len(source_nodes))
            try:
                imported = _import_program(shared, source_nodes, source_roots)
            except ConversionError as exc:
                raise ConversionError(
                    tuple(replace(d, code="EXPECTATION_NODE_LIMIT") for d in exc.diagnostics)
                ) from exc
            indices = tuple(slots.setdefault(root, len(slots)) for root in imported)
            groups.append(replace(group, program_indices=indices))
        coefficient_nodes, coefficient_roots = shared.compact(tuple(slots))
        groups = tuple(groups)
        dense_batches = _schedule_dense(groups, builder.charge)
        builder.charge(len(nodes) + builder.factors)
        nodes, roots, factor_aliases = _reuse_factors(
            nodes, roots, groups, dense_batches, builder.factors, self.max_nodes
        )
        reserve = 8 * (len(nodes) + builder.factors + 8)
        retained = reserve + 8 * (len(coefficient_nodes) + len(coefficient_roots))
        required = retained + max(
            (batch.working_bytes_per_lane for batch in dense_batches), default=0
        )
        if required > self._memory_bytes:
            raise _failure(
                "EXPECTATION_WORKING_LIMIT",
                "Shared coefficient and local arrays exceed the memory budget.",
            )
        active = tuple(sorted(set().union(*(g.active_parameters for g in groups))))
        for name, value in (
            ("observables", obs),
            ("groups", groups),
            ("dense_batches", dense_batches),
            ("active_parameters", active),
            ("planning_work", builder.work),
            ("factor_count", builder.factors),
            ("factor_aliases", factor_aliases),
            ("working_bytes_per_row", required),
            ("_nodes", nodes),
            ("_roots", roots),
            ("_coefficient_nodes", coefficient_nodes),
            ("_coefficient_roots", coefficient_roots),
            ("_source_parameters", frozenset(self.circuit.parameters)),
            ("_retained_bytes_per_row", retained),
        ):
            object.__setattr__(self, name, value)

    @property
    def diagnostics(self):
        return (
            Diagnostic(
                "DECLARED_EXPECTATIONS_ONLY",
                "Factorization preserves declared expectations from the all-zero initial "
                "state; full-program conversion retains the source IR.",
                severity="info",
            ),
            Diagnostic(
                "EXACT_EXECUTION_PORTFOLIO",
                f"Shared {self.factor_count} local readouts across {len(self.groups)} "
                "source circuits; equal dense functions reuse execution and matching "
                "templates pack parameter lanes.",
                severity="info",
            ),
            Diagnostic(
                "STATIC_COST_SELECTION",
                "Engine selection compares estimated scalar work; runtime speed depends "
                "on backend and batch shape.",
                severity="info",
            ),
        )

    def to_dict(self):
        return {
            "semantics": "declared_expectations",
            "initial_state": "all_zero",
            "source_resources": asdict(circuit_stats(self.circuit)),
            "cost_model": "static_scalar_work",
            "estimated_work": _program_work(self._coefficient_nodes)
            + len(self._nodes)
            + sum(
                len(batch.parameter_slots) * batch.estimated_work_per_lane
                for batch in self.dense_batches
            ),
            "planning_work": self.planning_work,
            "working_bytes_per_row": self.working_bytes_per_row,
            "factor_count": self.factor_count,
            "unique_factors": len(set(self.factor_aliases)),
            "factor_aliases": list(self.factor_aliases),
            "contraction_nodes": len(self._nodes),
            "shared_coefficient_nodes": len(self._coefficient_nodes),
            "shared_coefficient_outputs": len(self._coefficient_roots),
            "active_parameters": list(self.active_parameters),
            "removed_parameters": sorted(self._source_parameters - set(self.active_parameters)),
            "groups": [g.to_dict() for g in self.groups],
            "dense_batches": [batch.to_dict() for batch in self.dense_batches],
            "dense_unique_lanes": sum(len(b.parameter_slots) for b in self.dense_batches),
            "limits": {
                n: getattr(self, n)
                for n in (
                    "max_groups",
                    "max_dense_qubits",
                    "max_planning_work",
                    "max_terms",
                    "max_nodes",
                    "max_work",
                    "max_operations",
                    "max_memory_mb",
                    "max_dense_batch",
                )
            },
            "diagnostics": [asdict(d) for d in self.diagnostics],
        }

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def evaluate(self, bindings=None, *, backend="numpy", like=None, batch_shape=None):
        """Evaluate exact local factors and a division-free real contraction DAG.

        The working budget includes retained factor arrays, contraction values
        and the largest local primal workspace. Caller data, requested outputs
        and framework autodiff retention are excluded. Batch chunks can be set
        on native layers; public calls reject batches exceeding this budget.
        """
        ops, local, shape = _prepare_bindings(
            bindings, backend, like, batch_shape, self._source_parameters, self.active_parameters
        )
        if self.working_bytes_per_row * max(1, math.prod(shape)) > self._memory_bytes:
            raise ResourceError(
                "Planned expectation working arrays exceed max_memory_mb; "
                "reduce the batch or use batch_chunk_size."
            )
        factors = [None] * self.factor_count
        context = like if like is not None else ops.array(0.0, real_value=True)
        coefficients = (
            _evaluate_program(self._coefficient_nodes, self._coefficient_roots, local, ops, shape)
            if self._coefficient_roots
            else None
        )
        for group in self.groups:
            if group.pauli_plan is not None:
                for slot, factor in zip(group.program_indices, group.factor_indices, strict=True):
                    factors[factor] = coefficients[..., slot]
        _evaluate_dense(self, coefficients, factors, ops, shape, context)
        return _evaluate_program(self._nodes, self._roots, local, ops, shape, factors)

    def verify(
        self,
        bindings=None,
        *,
        reference_backend="numpy",
        atol=1e-10,
        rtol=1e-10,
        max_reference_qubits=12,
        max_reference_work=10_000_000,
    ):
        """Check one scalar binding against bounded, unfactored input IR execution.

        Reference engines are NumPy, Qiskit, Cirq and PennyLane. Full source
        bindings are required, including eliminated parameters. This reports
        numerical agreement at the supplied point, retaining no model values.
        """
        import numpy as np

        from .simulator import expectations

        if not isinstance(reference_backend, str) or reference_backend not in {
            "numpy",
            "qiskit",
            "cirq",
            "pennylane",
        }:
            raise ValidationError("Unknown reference verification backend.")
        limit = integer(max_reference_qubits, "max_reference_qubits", 1)
        if self.circuit.num_qubits > limit:
            raise ResourceError("Source verification exceeds max_reference_qubits.")
        readout_work = self.circuit.num_qubits * sum(
            len(ob.terms) if isinstance(ob, Hamiltonian) else 1 for ob in self.observables
        )
        _reference_work(
            self.circuit, readout_work, integer(max_reference_work, "max_reference_work", 1)
        )
        atol, rtol = real(atol), real(rtol)
        if min(atol, rtol) < 0:
            raise ValidationError("Verification tolerances must be nonnegative.")
        bindings = {} if bindings is None else bindings
        if not isinstance(bindings, Mapping) or set(bindings) != self._source_parameters:
            raise ValidationError("Verification requires all source parameter bindings.")
        values = {name: real(value) for name, value in bindings.items()}
        expected = expectations(
            self.circuit,
            self.observables,
            values,
            backend=reference_backend,
            max_qubits=limit,
            max_memory_mb=self.max_memory_mb,
        )
        actual = self.evaluate(values)
        errors = np.abs(actual - expected)
        mismatches = tuple(
            int(i) for i in np.flatnonzero(~np.isclose(actual, expected, atol=atol, rtol=rtol))
        )
        return ExpectationCheck(
            not mismatches, reference_backend, float(errors.max()), mismatches, atol, rtol
        )
