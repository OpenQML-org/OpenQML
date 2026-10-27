"""Exact expectation compilation by bounded symbolic Pauli propagation.

Background: Rudolph et al., https://arxiv.org/abs/2505.21606 and Monaco et al.,
https://arxiv.org/abs/2512.16674. This implementation uses no term truncation:
budget exhaustion is an explicit failure, never an approximate answer.
"""

import json
import math
from collections.abc import Mapping, Set
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import product

import numpy as np

from .compiler import _check_circuit_limits, circuit_stats, decompose
from .errors import Diagnostic, ResourceError, ValidationError, conversion_error
from .ir import Circuit, integer
from .observables import Hamiltonian, PauliString
from .parameters import real
from .simulator import _FIXED, _I, _X, _Y, _Z, _TensorOps

_CLIFFORD = frozenset("id x y z h s sdg sx sxdg cx cy cz swap iswap".split())
_ROTATIONS = {"rx": "X", "ry": "Y", "rz": "Z", "p": "Z", "rxx": "X", "ryy": "Y", "rzz": "Z"}
_BASIS = _CLIFFORD | _ROTATIONS.keys()
_UNARY = frozenset({"sin", "cos", "tan", "exp", "log", "sqrt"})


def _limit(code, message, location="circuit"):
    return conversion_error(
        code, message, location, "Increase the explicit budget or use dense execution."
    )


@lru_cache(maxsize=len(_CLIFFORD))
def _clifford_table(name):
    """Small, fixed conjugation tables; no circuit values or state arrays cached."""
    unitary = _FIXED[name]
    width = int(math.log2(len(unitary)))
    # Tables use I, X, Z, Y digits.
    matrices = []
    for digits in product(range(4), repeat=width):
        matrix = np.asarray([[1]], dtype=complex)
        for digit in digits:
            matrix = np.kron(matrix, (_I, _X, _Z, _Y)[digit])
        matrices.append(matrix)
    result = []
    for matrix in matrices:
        transformed = unitary.conj().T @ matrix @ unitary
        overlaps = np.real([np.trace(p @ transformed) / len(unitary) for p in matrices])
        index = int(np.argmax(np.abs(overlaps)))
        sign = 1 if overlaps[index] > 0 else -1
        # This checks known constant Clifford identities, not user gate data.
        if not np.allclose(transformed, sign * matrices[index], atol=1e-12, rtol=0):
            raise ValidationError("Invalid internal Clifford conjugation rule.")
        result.append((index, sign))
    return tuple(result)


def _clifford(x, z, qubits, table):
    word = 0
    for q in qubits:
        word = (word << 2) | ((x >> q) & 1) | (((z >> q) & 1) << 1)
    mapped, sign = table[word]
    for q in reversed(qubits):
        digit, mapped = mapped & 3, mapped >> 2
        bit = 1 << q
        x = (x & ~bit) | ((digit & 1) << q)
        z = (z & ~bit) | (((digit >> 1) & 1) << q)
    return (x, z), sign


def _masks(paulis):
    return (
        sum(1 << q for q, p in enumerate(paulis) if p in "XY"),
        sum(1 << q for q, p in enumerate(paulis) if p in "YZ"),
    )


def _rotation_sign(gx, gz, x, z):
    # Canonical Hermitian words are i**popcount(x&z) X**x Z**z.
    exponent = (gx & gz).bit_count() + (x & z).bit_count()
    exponent -= ((gx ^ x) & (gz ^ z)).bit_count()
    exponent += 2 * (gz & x).bit_count() + 1  # i * generator * observable
    return 1 if exponent % 4 == 0 else -1


class _Arena:
    """Intern real scalar arithmetic, then remove unreachable coefficient paths."""

    def __init__(self, max_nodes):
        self.nodes, self.index, self.max_nodes = [], {}, max_nodes
        self.expressions = {}
        self.location = "circuit"
        self.zero, self.one = self.make(("const", 0.0)), self.make(("const", 1.0))

    def make(self, key):
        if key in self.index:
            return self.index[key]
        if len(self.nodes) >= self.max_nodes:
            raise _limit("PAULI_NODE_LIMIT", "Coefficient graph exceeds max_nodes.", self.location)
        index = len(self.nodes)
        self.nodes.append(key)
        self.index[key] = index
        return index

    def const(self, value):
        return self.make(("const", real(value)))

    def angle(self, value):
        pending = [(value, False)]
        while pending:
            if len(pending) > 2 * self.max_nodes:
                raise _limit(
                    "PAULI_NODE_LIMIT",
                    "Parameter expression traversal exceeds max_nodes.",
                    self.location,
                )
            current, ready = pending.pop()
            identity = id(current)
            if identity in self.expressions:
                continue
            if len(self.expressions) >= self.max_nodes:
                raise _limit(
                    "PAULI_NODE_LIMIT",
                    "Parameter expression graph exceeds max_nodes.",
                    self.location,
                )
            if current.op is None:
                result = self.const(current.constant)
                for name, coefficient in current.terms:
                    term = self.binary("mul", self.const(coefficient), self.make(("param", name)))
                    result = self.binary("add", result, term)
            elif not ready:
                pending.append((current, True))
                pending.extend((arg, False) for arg in reversed(current.args))
                continue
            else:
                args = tuple(self.expressions[id(arg)] for arg in current.args)
                result = (
                    self.trig(current.op, args[0])
                    if current.op in _UNARY
                    else self.binary(current.op, *args)
                )
            self.expressions[identity] = result
        return self.expressions[id(value)]

    def trig(self, name, value):
        node = self.nodes[value]
        if node[0] != "const":
            return self.make((name, value))
        try:
            return self.const(getattr(math, name)(node[1]))
        except (ArithmeticError, ValueError) as exc:
            raise ValidationError(
                "Compiled parameter expression is undefined over finite reals."
            ) from exc

    def binary(self, name, left, right):
        if name == "mul":
            if self.zero in (left, right):
                return self.zero
            if left == self.one:
                return right
            if right == self.one:
                return left
        elif name == "add":
            if left == self.zero:
                return right
            if right == self.zero:
                return left
        a, b = self.nodes[left], self.nodes[right]
        if a[0] == b[0] == "const":
            try:
                if name == "mul":
                    value = a[1] * b[1]
                elif name == "add":
                    value = a[1] + b[1]
                elif name == "div":
                    value = a[1] / b[1]
                else:
                    value = a[1] ** b[1]
                return self.const(value)
            except (ArithmeticError, ValueError) as exc:
                raise ValidationError(
                    "Compiled parameter expression is undefined over finite reals."
                ) from exc
        indices = sorted((left, right)) if name in {"add", "mul"} else (left, right)
        return self.make((name, *indices))

    def signed(self, value, sign):
        return value if sign == 1 else self.binary("mul", self.const(-1), value)

    def total(self, values):
        values = list(values)
        while len(values) > 1:
            values = [
                self.binary("add", values[i], values[i + 1]) for i in range(0, len(values) - 1, 2)
            ] + (values[-1:] if len(values) % 2 else [])
        return values[0] if values else self.zero

    def compact(self, roots):
        reached, pending = set(), list(roots)
        while pending:
            index = pending.pop()
            if index not in reached:
                reached.add(index)
                node = self.nodes[index]
                if node[0] not in {"const", "param"}:
                    pending.extend(node[1:])
        remap, nodes = {}, []
        for index in sorted(reached):
            node = self.nodes[index]
            remap[index] = len(nodes)
            nodes.append(
                node if node[0] in {"const", "param"} else (node[0], *(remap[i] for i in node[1:]))
            )
        return tuple(nodes), tuple(remap[i] for i in roots)


def _compile(circuit, observables, max_terms, max_nodes, max_work, max_operations):
    _check_circuit_limits(circuit, max_operations=max_operations)
    if circuit.has_effects:
        raise conversion_error(
            "PAULI_EFFECTS_UNSUPPORTED",
            "Pauli compilation requires a unitary circuit without effects.",
        )
    if isinstance(observables, (str, bytes, Mapping, Set)):
        raise ValidationError("Pauli observables require an ordered iterable.")
    try:
        iterator = iter(observables)
    except TypeError as exc:
        raise ValidationError("Pauli observables require an ordered iterable.") from exc
    labels, collected, source_terms = {}, [], 0
    for ob in iterator:
        if len(collected) >= max_nodes:
            raise _limit("PAULI_NODE_LIMIT", "Output count exceeds max_nodes.")
        collected.append(ob)
        for term in ob.terms if isinstance(ob, Hamiltonian) else (ob,):
            source_terms += 1
            if source_terms > max_nodes:
                raise _limit("PAULI_NODE_LIMIT", "Source term count exceeds max_nodes.")
            if not isinstance(term, PauliString) or len(term.paulis) != circuit.num_qubits:
                raise ValidationError("Pauli observable width must match the circuit.")
            if term.coefficient:
                labels.setdefault(term.paulis, None)
    observables = tuple(collected)
    if not observables:
        raise ValidationError("Pauli compilation requires observables.")
    lowered = decompose(circuit, basis=_BASIS, max_operations=max_operations)
    arena, work, peak = _Arena(max_nodes), 0, 0
    prepared = []
    for op in lowered.operations:
        if op.name == "barrier":
            prepared.append(None)
        elif op.name in _CLIFFORD:
            prepared.append(("clifford", op.qubits, _clifford_table(op.name)))
        else:
            axis = _ROTATIONS[op.name]
            mask = sum(1 << q for q in op.qubits)
            prepared.append(
                ("rotation", mask if axis != "Z" else 0, mask if axis != "X" else 0, op.params[0])
            )
    for label in labels:
        word = _masks(label)
        if word == (0, 0):
            labels[label] = arena.one
            continue
        terms = {word: arena.one}
        peak = max(peak, 1)
        for index in range(len(prepared) - 1, -1, -1):
            op = prepared[index]
            if op is None:
                continue
            arena.location = f"prepared.operations[{index}]"
            if len(terms) > max_work - work:
                raise _limit("PAULI_WORK_LIMIT", "Propagation exceeds max_work.", arena.location)
            work += len(terms)
            output = {}

            def emit(word, value, output=output):
                if value == arena.zero:
                    return
                if word in output:
                    output[word] = arena.binary("add", output[word], value)
                    if output[word] == arena.zero:
                        del output[word]
                else:
                    if len(output) >= max_terms:
                        raise _limit(
                            "PAULI_TERM_LIMIT", "Propagation exceeds max_terms.", arena.location
                        )
                    output[word] = value

            for (x, z), coefficient in terms.items():
                if op[0] == "clifford":
                    word, sign = _clifford(x, z, op[1], op[2])
                    emit(word, arena.signed(coefficient, sign))
                else:
                    _, gx, gz, angle = op
                    if ((gx & z).bit_count() + (gz & x).bit_count()) % 2 == 0:
                        emit((x, z), coefficient)
                    else:
                        leaf = arena.angle(angle)
                        emit((x, z), arena.binary("mul", coefficient, arena.trig("cos", leaf)))
                        second = arena.binary("mul", coefficient, arena.trig("sin", leaf))
                        emit((x ^ gx, z ^ gz), arena.signed(second, _rotation_sign(gx, gz, x, z)))
            terms = output
            peak = max(peak, len(terms))
        # Only I/Z words have a nonzero expectation in the all-zero state.
        labels[label] = arena.total(value for (x, _), value in terms.items() if x == 0)
    roots = tuple(
        arena.total(
            arena.binary("mul", arena.const(term.coefficient), labels[term.paulis])
            for term in (ob.terms if isinstance(ob, Hamiltonian) else (ob,))
            if term.coefficient
        )
        for ob in observables
    )
    nodes, roots = arena.compact(roots)
    return observables, nodes, roots, work, peak, len(arena.nodes), len(lowered.operations)


@dataclass(frozen=True, eq=False)
class PauliPlan:
    """Reusable real arithmetic for declared expectations from the all-zero state.

    No state vector, density matrix, bindings or framework tensors are retained.
    Native Torch/JAX evaluation supports higher derivatives. Compile budgets
    bound frontier terms, all allocated DAG nodes and total Pauli transitions.
    max_nodes also bounds source terms, outputs and expression traversal.
    Exact propagation can grow exponentially; limits raise ConversionError.
    Full-program export uses circuit, never the coefficient graph as a circuit.
    """

    circuit: Circuit
    observables: tuple
    max_terms: int = 4096
    max_nodes: int = 100_000
    max_work: int = 1_000_000
    max_operations: int = 100_000
    max_memory_mb: float = 256
    active_parameters: tuple = field(init=False)
    peak_terms: int = field(init=False)
    propagation_work: int = field(init=False)
    construction_nodes: int = field(init=False)
    prepared_operations: int = field(init=False)
    _nodes: tuple = field(init=False, repr=False)
    _roots: tuple = field(init=False, repr=False)
    _source_parameters: frozenset = field(init=False, repr=False)

    def __post_init__(self):
        for name in ("max_terms", "max_nodes", "max_work", "max_operations"):
            object.__setattr__(
                self, name, integer(getattr(self, name), name, 2 if name == "max_nodes" else 1)
            )
        budget = real(self.max_memory_mb)
        if budget <= 0:
            raise ValidationError("max_memory_mb must be positive.")
        obs, nodes, roots, work, peak, allocated, operations = _compile(
            self.circuit,
            self.observables,
            self.max_terms,
            self.max_nodes,
            self.max_work,
            self.max_operations,
        )
        active = tuple(sorted({n[1] for n in nodes if n[0] == "param"}))
        for name, value in (
            ("observables", obs),
            ("_nodes", nodes),
            ("_roots", roots),
            ("active_parameters", active),
            ("peak_terms", peak),
            ("propagation_work", work),
            ("construction_nodes", allocated),
            ("prepared_operations", operations),
            ("_source_parameters", frozenset(self.circuit.parameters)),
            ("max_memory_mb", budget),
        ):
            object.__setattr__(self, name, value)

    @property
    def diagnostics(self):
        return (
            Diagnostic(
                "PAULI_EXPECTATIONS_ONLY",
                "The coefficient graph preserves declared expectations from the all-zero "
                "initial state; full-program export retains the source IR.",
                severity="info",
            ),
            Diagnostic(
                "PAULI_EXACT_COMPILATION",
                f"Compiled {len(self._nodes)} real arithmetic nodes with peak frontier "
                f"{self.peak_terms}; no approximate term truncation.",
                severity="info",
            ),
        )

    def to_dict(self):
        return {
            "semantics": "declared_expectations",
            "initial_state": "all_zero",
            "source_resources": asdict(circuit_stats(self.circuit)),
            "prepared_operations": self.prepared_operations,
            "outputs": len(self.observables),
            "arithmetic_nodes": len(self._nodes),
            "construction_nodes": self.construction_nodes,
            "peak_terms": self.peak_terms,
            "propagation_work": self.propagation_work,
            "active_parameters": list(self.active_parameters),
            "removed_parameters": sorted(self._source_parameters - set(self.active_parameters)),
            "limits": {
                n: getattr(self, n)
                for n in ("max_terms", "max_nodes", "max_work", "max_operations", "max_memory_mb")
            },
            "diagnostics": [asdict(d) for d in self.diagnostics],
        }

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def evaluate(self, bindings=None, *, backend="numpy", like=None, batch_shape=None):
        """Evaluate named scalar/array bindings with native broadcast and autodiff.

        Active symbols are required; unused source symbols may be omitted.
        The working budget covers conservative primal real-array volume,
        excluding caller data, output arrays and framework autodiff retention.
        No evaluator is compiled implicitly and no numerical values are cached.
        """
        bindings = {} if bindings is None else bindings
        if not isinstance(bindings, Mapping):
            raise ValidationError("Pauli bindings require a name mapping.")
        if set(bindings) - self._source_parameters or set(self.active_parameters) - set(bindings):
            raise ValidationError(
                "Pauli bindings must cover active circuit symbols without unknown names."
            )
        if not isinstance(backend, str) or backend not in {"numpy", "torch", "jax"}:
            raise ValidationError("Pauli evaluation requires numpy, torch or jax.")
        if backend == "numpy":
            clean = {}
            for name, value in bindings.items():
                array = np.asarray(value)
                if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
                    raise ValidationError("Pauli bindings must be finite real scalars or arrays.")
                clean[name] = array.astype(float, copy=True)
            bindings = clean
        if like is None and backend != "numpy":
            like = next((v for v in bindings.values() if hasattr(v, "dtype")), None)
        ops = _TensorOps(backend, like)
        for value in bindings.values():
            dtype = getattr(value, "dtype", None)
            kind = getattr(dtype, "kind", None)
            if (
                isinstance(value, (bool, complex, str, bytes))
                or getattr(dtype, "is_complex", False)
                or (kind is not None and kind not in "iuf")
                or (backend == "torch" and dtype == ops.xp.bool)
            ):
                raise ValidationError("Pauli bindings must be real numeric values.")
        local = {n: ops.array(v, real_value=True) for n, v in bindings.items()}
        try:
            shape = np.broadcast_shapes(*(tuple(v.shape) for v in local.values()))
            if batch_shape is not None:
                target = tuple(integer(n, "batch dimension") for n in batch_shape)
                shape = np.broadcast_shapes(shape, target)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "Pauli parameter batch shapes cannot be broadcast together."
            ) from exc
        if (len(self._nodes) + 8) * 8 * max(1, math.prod(shape)) > self.max_memory_mb * 2**20:
            raise ResourceError(
                "Pauli arithmetic exceeds max_memory_mb; reduce the batch or use batch_chunk_size."
            )
        values = []
        for node in self._nodes:
            name = node[0]
            if name == "const":
                value = ops.array(node[1], real_value=True)
            elif name == "param":
                value = local[node[1]]
            elif name in _UNARY:
                value = getattr(ops.xp, name)(values[node[1]])
            elif name == "mul":
                value = values[node[1]] * values[node[2]]
            elif name == "div":
                value = values[node[1]] / values[node[2]]
            elif name == "pow":
                value = values[node[1]] ** values[node[2]]
            else:
                value = values[node[1]] + values[node[2]]
            values.append(value)
        outputs = tuple(ops.xp.broadcast_to(values[i], shape) for i in self._roots)
        result = (
            ops.xp.stack(outputs, dim=-1) if backend == "torch" else ops.xp.stack(outputs, axis=-1)
        )
        if backend == "numpy" and not np.isfinite(result).all():
            raise ValidationError("Pauli expectations exceed finite real precision.")
        return result
