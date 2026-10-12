"""Composable quantum programs; angles come first in rotation methods."""

from dataclasses import dataclass, replace

import numpy as np

from ._validation import integer, pauli_string, probability, real, wires
from .exceptions import ResourceLimitError, UnsupportedOperationError
from .simulation import H, SimulationBudget, State, X, Y, Z, validate_channel, validate_matrix


def rotation(axis, theta):
    theta = real(theta, "angle")
    return np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * {"x": X, "y": Y, "z": Z}[axis]


@dataclass(frozen=True)
class Operation:
    name: str
    wires: tuple[int, ...]
    controls: tuple[int, ...] = ()
    matrix: object = None
    angle: float | None = None
    key: str | None = None
    basis: str = "Z"
    condition: tuple[str, int] | None = None
    kraus: tuple = ()
    label: str | None = None


class Circuit:
    """Build unitary circuits, noise channels, and measurement feedback.

    Every mutating builder returns self. Measurement conditions refer to earlier
    keys and run in circuit order. Circuits can contain 64 wires; local simulation
    has a smaller explicit SimulationBudget.
    """

    def __init__(self, n_qubits):
        self.n_qubits = integer(n_qubits, "n_qubits", 1, 64)
        self._operations = []

    @property
    def operations(self):
        # Arrays are copied so callers cannot mutate a circuit through its readout.
        return tuple(
            replace(
                op,
                matrix=None if op.matrix is None else op.matrix.copy(),
                kraus=tuple(k.copy() for k in op.kraus),
            )
            for op in self._operations
        )

    def __len__(self):
        return len(self._operations)

    def copy(self):
        circuit = Circuit(self.n_qubits)
        circuit._operations = list(self._operations)
        return circuit

    def _condition(self, condition):
        if condition is None:
            return None
        if (
            not isinstance(condition, tuple)
            or len(condition) != 2
            or not isinstance(condition[0], str)
            or condition[1] not in (0, 1)
        ):
            raise ValueError("condition must be a (measurement_key, 0 or 1) tuple")
        if condition[0] not in {op.key for op in self._operations if op.name == "measure"}:
            raise ValueError("condition must refer to an earlier measurement key")
        return condition

    def _gate(self, name, matrix, targets, controls=(), angle=None, condition=None):
        targets = wires(targets, self.n_qubits, allow_empty=True)
        controls = wires(controls, self.n_qubits, allow_empty=True)
        if len(targets) > 4 or set(targets) & set(controls):
            raise ValueError("at most 4 target wires, disjoint from controls, are supported")
        self._operations.append(
            Operation(
                name,
                targets,
                controls,
                validate_matrix(matrix, 1 << len(targets)),
                angle,
                condition=self._condition(condition),
            )
        )
        return self

    def h(self, q, *, condition=None):
        return self._gate("h", H, [q], condition=condition)

    def x(self, q, *, condition=None):
        return self._gate("x", X, [q], condition=condition)

    def y(self, q, *, condition=None):
        return self._gate("y", Y, [q], condition=condition)

    def z(self, q, *, condition=None):
        return self._gate("z", Z, [q], condition=condition)

    def s(self, q):
        return self._gate("s", np.diag([1, 1j]), [q])

    def sdg(self, q):
        return self._gate("sdg", np.diag([1, -1j]), [q])

    def t(self, q):
        return self._gate("t", np.diag([1, np.exp(1j * np.pi / 4)]), [q])

    def tdg(self, q):
        return self._gate("tdg", np.diag([1, np.exp(-1j * np.pi / 4)]), [q])

    def rx(self, theta, q, *, condition=None):
        return self._gate(
            "rx", rotation("x", theta), [q], angle=real(theta, "angle"), condition=condition
        )

    def ry(self, theta, q, *, condition=None):
        return self._gate(
            "ry", rotation("y", theta), [q], angle=real(theta, "angle"), condition=condition
        )

    def rz(self, theta, q, *, condition=None):
        return self._gate(
            "rz", rotation("z", theta), [q], angle=real(theta, "angle"), condition=condition
        )

    def p(self, theta, q):
        theta = real(theta, "angle")
        return self._gate("p", np.diag([1, np.exp(1j * theta)]), [q], angle=theta)

    def cx(self, control, target, *, condition=None):
        return self._gate("x", X, [target], [control], condition=condition)

    cnot = cx

    def cy(self, control, target):
        return self._gate("y", Y, [target], [control])

    def cz(self, control, target):
        return self._gate("z", Z, [target], [control])

    def cp(self, theta, control, target):
        theta = real(theta, "angle")
        return self._gate("p", np.diag([1, np.exp(1j * theta)]), [target], [control], theta)

    def ccx(self, first, second, target):
        return self.mcx([first, second], target)

    def mcx(self, controls, target):
        return self._gate("x", X, [target], wires(controls, self.n_qubits))

    def swap(self, first, second):
        return self._gate(
            "swap", [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], [first, second]
        )

    def global_phase(self, theta):
        theta = real(theta, "angle")
        return self._gate("global_phase", [[np.exp(1j * theta)]], (), angle=theta)

    def unitary(self, matrix, targets, *, controls=(), label="unitary"):
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a nonempty string")
        targets = wires(targets, self.n_qubits)
        # Keep a reserved operation name; a label must not masquerade as a built-in gate.
        self._gate("unitary", matrix, targets, controls)
        self._operations[-1] = replace(self._operations[-1], label=label)
        return self

    def pauli_rotation(self, theta, pauli):
        theta = real(theta, "angle")
        pauli = pauli_string(pauli, self.n_qubits)
        active = [(q, p) for q, p in enumerate(reversed(pauli)) if p != "I"]
        if not active:
            return self.global_phase(-theta / 2)
        basis = Circuit(self.n_qubits)
        for q, p in active:
            if p == "X":
                basis.h(q)
            elif p == "Y":
                basis.sdg(q).h(q)
        for q, _ in active[:-1]:
            basis.cx(q, active[-1][0])
        return self.compose(basis).rz(theta, active[-1][0]).compose(basis.inverse())

    def qft(self, selected_wires=None, *, inverse=False, swaps=True):
        selected = (
            tuple(range(self.n_qubits))
            if selected_wires is None
            else wires(selected_wires, self.n_qubits)
        )
        block = Circuit(self.n_qubits)
        for j in reversed(range(len(selected))):
            block.h(selected[j])
            for k in reversed(range(j)):
                block.cp(np.pi / (2 ** (j - k)), selected[k], selected[j])
        if swaps:
            for j in range(len(selected) // 2):
                block.swap(selected[j], selected[-1 - j])
        return self.compose(block.inverse() if inverse else block)

    def iqft(self, selected_wires=None, *, swaps=True):
        return self.qft(selected_wires, inverse=True, swaps=swaps)

    def measure(self, q, key=None, *, basis="Z"):
        q = wires([q], self.n_qubits)[0]
        if basis not in ("X", "Y", "Z"):
            raise ValueError("measurement basis must be X, Y, or Z")
        key = f"m{sum(op.name == 'measure' for op in self._operations)}" if key is None else key
        if not isinstance(key, str) or not key or key in {op.key for op in self._operations}:
            raise ValueError("measurement keys must be unique nonempty strings")
        self._operations.append(Operation("measure", (q,), key=key, basis=basis))
        return self

    def channel(self, kraus, targets):
        targets = wires(targets, self.n_qubits)
        if len(targets) > 4:
            raise ValueError("channels support at most 4 target wires")
        self._operations.append(
            Operation("channel", targets, kraus=validate_channel(kraus, 1 << len(targets)))
        )
        return self

    def amplitude_damping(self, p, q):
        p = probability(p)
        return self.channel([[[1, 0], [0, np.sqrt(1 - p)]], [[0, np.sqrt(p)], [0, 0]]], [q])

    def dephasing(self, p, q):
        p = probability(p)
        return self.channel([np.sqrt(1 - p) * np.eye(2), np.sqrt(p) * Z], [q])

    def depolarizing(self, p, q):
        p = probability(p)
        return self.channel(
            [np.sqrt(1 - 3 * p / 4) * np.eye(2)] + [np.sqrt(p / 4) * g for g in (X, Y, Z)], [q]
        )

    def reset(self, q):
        return self.channel([[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [q])

    def compose(self, block, mapping=None):
        if not isinstance(block, Circuit):
            raise ValueError("block must be a Circuit")
        mapping = tuple(range(block.n_qubits)) if mapping is None else wires(mapping, self.n_qubits)
        if len(mapping) != block.n_qubits or any(q >= self.n_qubits for q in mapping):
            raise ValueError("mapping must provide one distinct destination per block wire")
        existing_keys = {op.key for op in self._operations if op.key is not None}
        incoming_keys = {op.key for op in block._operations if op.key is not None}
        if existing_keys & incoming_keys:
            raise ValueError("composed measurement keys must be unique")
        # Take a snapshot: compose(self) must not grow the list while iterating.
        incoming = [
            replace(
                op,
                wires=tuple(mapping[q] for q in op.wires),
                controls=tuple(mapping[q] for q in op.controls),
            )
            for op in block._operations
        ]
        self._operations.extend(incoming)
        return self

    def inverse(self):
        if any(op.matrix is None or op.condition is not None for op in self._operations):
            raise UnsupportedOperationError("only unconditional unitary circuits have an inverse")
        inverse_names = {"s": "sdg", "sdg": "s", "t": "tdg", "tdg": "t"}
        result = Circuit(self.n_qubits)
        result._operations = [
            replace(
                op,
                matrix=validate_matrix(op.matrix.conj().T, op.matrix.shape[0]),
                angle=None if op.angle is None else -op.angle,
                name=inverse_names.get(op.name, op.name),
            )
            for op in reversed(self._operations)
        ]
        return result

    def repeat(self, count):
        count = integer(count, "count", 0)
        if count * len(self) > 100_000:
            raise ResourceLimitError("repeat would create more than 100,000 operations")
        if count > 1 and any(op.key is not None for op in self._operations):
            raise UnsupportedOperationError("repeat requires a circuit without measurement keys")
        result = Circuit(self.n_qubits)
        result._operations = self._operations * count
        return result

    def run(self, *, seed=None, backend="auto", budget=None, initial_state=None):
        budget = budget or (
            initial_state.budget if isinstance(initial_state, State) else SimulationBudget()
        )
        if not isinstance(budget, SimulationBudget):
            raise ValueError("budget must be a SimulationBudget")
        if self.n_qubits > budget.max_qubits:
            raise ResourceLimitError("Circuit width exceeds the simulation budget")
        if len(self) > budget.max_operations:
            raise ResourceLimitError("Circuit exceeds the operation budget")
        if initial_state is None:
            state = State(self.n_qubits, backend=backend, seed=seed, budget=budget)
        else:
            if not isinstance(initial_state, State) or initial_state.n_qubits != self.n_qubits:
                raise ValueError("initial_state must be a State of the same width")
            if initial_state.backend == "density":
                if self.n_qubits > budget.max_density_qubits:
                    raise ResourceLimitError("initial density state exceeds the new budget")
                budget.allocation(1 << (2 * self.n_qubits), copies=6)
            elif initial_state.backend == "dense":
                if self.n_qubits > budget.max_dense_qubits:
                    raise ResourceLimitError("initial dense state exceeds the new budget")
                budget.allocation(1 << self.n_qubits, copies=6)
            elif len(initial_state._sparse) > min(
                budget.max_sparse_states, budget.max_memory_mb * 1024**2 // 384
            ):
                raise ResourceLimitError("initial sparse state exceeds the new budget")
            state = initial_state.clone()
            state.budget = budget
            state._work = 0
            state.classical = {}
            if seed is not None:
                state._rng = np.random.default_rng(seed)
        for op in self._operations:
            if op.condition is not None and state.classical[op.condition[0]] != op.condition[1]:
                continue
            if op.name == "measure":
                state.classical[op.key] = state.measure(op.wires[0], op.basis)
            elif op.name == "channel":
                state.channel(op.kraus, op.wires)
            else:
                state._apply(op.matrix, op.wires, op.controls)
        return state

    def to_qasm(self):
        from .interop import to_qasm

        return to_qasm(self)

    @classmethod
    def from_qasm(cls, source):
        from .interop import from_qasm

        return from_qasm(source)

    def to_qiskit(self):
        from .interop import to_qiskit

        return to_qiskit(self)

    def resources(self):
        depths = [0] * self.n_qubits
        measurement_depths = {}
        for op in self._operations:
            occupied = op.wires + op.controls
            if occupied:
                level = 1 + max(depths[q] for q in occupied)
                if op.condition is not None:
                    level = max(level, measurement_depths[op.condition[0]] + 1)
                for q in occupied:
                    depths[q] = level
                if op.name == "measure":
                    measurement_depths[op.key] = level
        return {
            "n_qubits": self.n_qubits,
            "operations": len(self),
            "depth": max(depths),
            "two_qubit_gates": sum(
                op.matrix is not None and len(op.wires + op.controls) == 2
                for op in self._operations
            ),
            "measurements": sum(op.name == "measure" for op in self._operations),
            "noise_channels": sum(op.name == "channel" for op in self._operations),
        }

    def __repr__(self):
        return f"Circuit(n_qubits={self.n_qubits}, operations={len(self)})"
