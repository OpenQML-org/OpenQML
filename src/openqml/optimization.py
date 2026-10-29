"""Opt-in logical optimization of immutable quantum/classical IR.

Only classical facts are propagated: unwritten storage is zero, while every
measurement is unknown regardless of its numerical probability. No quantum
state, binding, tensor or approximate threshold is consulted.
"""

from dataclasses import dataclass, replace

from .compiler import CircuitStats, _sum_angles, circuit_stats, simplify
from .errors import Diagnostic, ValidationError, conversion_error
from .ir import Circuit, ClassicalCondition, Operation, integer


class _Facts:
    """Branch-local overlay; fork cost is independent of register/history width."""

    def __init__(self, parent=None):
        self.parent, self.values = parent, {}

    def get(self, bit):
        scope = self
        while scope is not None:
            if bit in scope.values:
                return scope.values[bit]
            scope = scope.parent
        return 0

    def merge(self, left, right):
        for bit in left.values.keys() | right.values.keys():
            a, b = left.get(bit), right.get(bit)
            self.values[bit] = a if a == b else None


def _canonical(condition):
    pairs = sorted((bit, (condition.value >> i) & 1) for i, bit in enumerate(condition.bits))
    return ClassicalCondition(
        tuple(bit for bit, _ in pairs), sum(value << i for i, (_, value) in enumerate(pairs))
    )


def _writes_guard(operation):
    bits = set(operation.condition.bits)
    return any(
        bits.intersection(op.clbits)
        for block in operation.blocks
        for _, op in block.walk_operations()
    )


@dataclass
class _DecisionRun:
    operation: Operation
    operations: list
    phases: list
    widths: list
    writes_guard: bool

    @classmethod
    def start(cls, op, writes_guard):
        return cls(
            op,
            [list(b.operations) for b in op.blocks],
            [[b.global_phase] for b in op.blocks],
            [b.num_clbits for b in op.blocks],
            writes_guard,
        )

    def extend(self, op, writes_guard):
        for i, block in enumerate(op.blocks):
            self.operations[i].extend(block.operations)
            self.phases[i].append(block.global_phase)
            self.widths[i] = max(self.widths[i], block.num_clbits)
        self.writes_guard = writes_guard

    def materialize(self):
        blocks = tuple(
            Circuit(
                self.operation.blocks[i].num_qubits,
                tuple(self.operations[i]),
                self.widths[i],
                _sum_angles(self.phases[i]),
            )
            for i in range(2)
        )
        return replace(self.operation, blocks=blocks)


@dataclass(frozen=True)
class OptimizationResult:
    circuit: Circuit
    before: CircuitStats
    after: CircuitStats
    folded_conditionals: int
    merged_conditionals: int
    removed_parameters: tuple[str, ...]
    passes: int
    converged: bool

    @property
    def diagnostics(self):
        notes = [
            Diagnostic(
                "EXACT_LOGICAL_OPTIMIZATION",
                f"Folded {self.folded_conditionals} and merged "
                f"{self.merged_conditionals} decisions.",
                severity="info",
            )
        ]
        notes.extend(
            Diagnostic(
                "PARAMETER_ELIMINATED",
                f"{name!r} no longer affects this optimized program.",
                "parameters",
                severity="info",
            )
            for name in self.removed_parameters
        )
        if not self.converged:
            notes.append(
                Diagnostic(
                    "OPTIMIZATION_PASS_LIMIT",
                    "Returned a valid result before reaching a fixed point.",
                    severity="info",
                )
            )
        return tuple(notes)


@dataclass(frozen=True)
class CertifiedOptimization:
    optimization: OptimizationResult
    proof: object

    @property
    def circuit(self):
        return self.optimization.circuit


def optimize_verified(circuit, *, proof_options=None, **optimization_options):
    """Return an optimization only when its exact IR semantics are proved.

    Unproved numeric angle identities and exhausted proof budgets are explicit.
    This opt-in check does not change the ordinary optimization API or certify
    hardware-dependent gate/noise context.
    """
    from .semantics import prove_equivalence

    optimized = optimize(circuit, **optimization_options)
    proof = prove_equivalence(circuit, optimized.circuit, **dict(proof_options or {}))
    proof.raise_for_errors()
    return CertifiedOptimization(optimized, proof)


def optimize(circuit, *, max_operations=100_000, max_depth=64, max_passes=4):
    """Fold known guards, merge stable adjacent decisions, then simplify gates.

    Exact logical semantics and explicit IR channels/measurement writes are
    preserved. Gate-dependent provider noise must be rechecked after changing
    physical gate sequences. This pass is never enabled implicitly by a backend.
    Removed symbols are reported; callers must update bindings/layer ownership.
    No numeric probability pruning or approximation is performed.
    """
    if not isinstance(circuit, Circuit):
        raise ValidationError("optimize expects an OpenQML Circuit.")
    integer(max_operations, "max_operations")
    integer(max_depth, "max_depth")
    integer(max_passes, "max_passes", 1)
    if max_passes > 32:
        raise ValidationError("max_passes cannot exceed 32.")
    pending, count = [(circuit, 0)], 0
    while pending:
        block, depth = pending.pop()
        if depth > min(max_depth, 128):
            raise conversion_error(
                "OPTIMIZATION_LIMIT", "Control depth exceeds the optimization limit."
            )
        for op in block.operations:
            count += 1
            if count > max_operations:
                raise conversion_error("OPTIMIZATION_LIMIT", "Circuit exceeds max_operations.")
            pending.extend((b, depth + 1) for b in op.blocks)
    before = circuit_stats(circuit)
    folded = merged = 0

    def walk(body, facts):
        nonlocal folded, merged
        output, phases = [], [body.global_phase]

        def append(op):
            nonlocal merged
            if op.name == "if_else":
                writes_guard = _writes_guard(op)
                if (
                    output
                    and isinstance(output[-1], _DecisionRun)
                    and op.condition == output[-1].operation.condition
                    and not output[-1].writes_guard
                ):
                    output[-1].extend(op, writes_guard)
                    merged += 1
                else:
                    output.append(_DecisionRun.start(op, writes_guard))
            else:
                output.append(op)

        for op in body.operations:
            if op.name == "measure":
                facts.values[op.clbits[0]] = None
                append(op)
                continue
            if op.name != "if_else":
                append(op)
                continue
            condition = _canonical(op.condition)
            pairs = [(bit, (condition.value >> i) & 1) for i, bit in enumerate(condition.bits)]
            unknown = [bit for bit, _ in pairs if facts.get(bit) is None]
            mismatch = any(facts.get(bit) not in {None, value} for bit, value in pairs)
            if mismatch or not unknown:
                folded += 1
                selected = walk(op.blocks[1 if mismatch else 0], facts)
                phases.append(selected.global_phase)
                for child in selected.operations:
                    append(child)
                continue
            left, right = _Facts(facts), _Facts(facts)
            left.values.update(pairs)
            if len(unknown) == 1:
                bit = unknown[0]
                right.values[bit] = 1 - dict(pairs)[bit]
            blocks = (walk(op.blocks[0], left), walk(op.blocks[1], right))
            facts.merge(left, right)
            append(replace(op, condition=condition, blocks=blocks))
        return simplify(
            replace(
                body,
                operations=tuple(
                    item.materialize() if isinstance(item, _DecisionRun) else item
                    for item in output
                ),
                global_phase=_sum_angles(phases),
            )
        )

    current, converged, passes = circuit, False, 0
    for _ in range(max_passes):
        passes += 1
        updated = walk(current, _Facts())
        if updated == current:
            converged = True
            break
        current = updated
    return OptimizationResult(
        current,
        before,
        circuit_stats(current),
        folded,
        merged,
        tuple(sorted(set(circuit.parameters) - set(current.parameters))),
        passes,
        converged,
    )
