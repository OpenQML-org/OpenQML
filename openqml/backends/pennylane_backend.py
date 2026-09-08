"""Optional PennyLane bridge.

Only imported when a ``pennylane.*`` backend is requested. It translates the
OpenQML circuit IR into PennyLane operations, which is also the cleanest place
to look when writing a bridge to Qiskit, Cirq or an in-house simulator.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from ..circuits.circuit import Circuit

_GATE_MAP = {
    "h": "Hadamard", "x": "PauliX", "y": "PauliY", "z": "PauliZ", "s": "S", "t": "T",
    "rx": "RX", "ry": "RY", "rz": "RZ", "phase": "PhaseShift",
    "cnot": "CNOT", "cz": "CZ", "swap": "SWAP",
    "crx": "CRX", "cry": "CRY", "crz": "CRZ", "rzz": "IsingZZ", "rxx": "IsingXX",
}


class PennyLaneSimulator:  # pragma: no cover - requires the optional dependency
    """Same surface as :class:`~openqml.backends.StatevectorSimulator`."""

    name = "pennylane"

    def __init__(self, n_qubits: int, device: str = "default.qubit", seed=None,
                 shots: Optional[int] = None):
        import pennylane as qml

        self._qml = qml
        self.n_qubits = int(n_qubits)
        self.shots = shots
        self._device = qml.device(device, wires=self.n_qubits, shots=shots)
        self._state = None
        self._circuit = None
        self._weights = None
        self._features = None

    def _apply(self, circuit: Circuit, weights, features):
        qml = self._qml
        for gate in circuit.gates:
            params = gate.resolved_params(weights, features)
            if gate.name == "amplitude_embedding":
                vector = np.asarray(features, dtype=complex).reshape(-1)
                qml.StatePrep(vector / np.linalg.norm(vector), wires=range(self.n_qubits))
                continue
            operation = getattr(qml, _GATE_MAP[gate.name])
            operation(*params, wires=list(gate.wires))

    def run(self, circuit: Circuit, weights=None, features=None):
        qml = self._qml

        @qml.qnode(self._device)
        def node():
            self._apply(circuit, weights, features)
            return qml.state()

        self._state = np.asarray(node()).reshape(-1)
        self._circuit, self._weights, self._features = circuit, weights, features
        return self

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def probabilities(self) -> np.ndarray:
        return np.abs(self._state) ** 2

    def expval(self, observable: Mapping[int, str]) -> float:
        qml = self._qml
        terms = [getattr(qml, f"Pauli{letter.upper()}")(int(wire))
                 for wire, letter in observable.items() if letter.upper() != "I"]
        if not terms:
            return 1.0
        product = terms[0]
        for term in terms[1:]:
            product = product @ term

        @qml.qnode(self._device)
        def node():
            self._apply(self._circuit, self._weights, self._features)
            return qml.expval(product)

        return float(node())

    def expval_hamiltonian(self, terms) -> float:
        return float(sum(float(c) * self.expval(word) for c, word in terms))

    def sample(self, shots: int = 1024):
        probs = self.probabilities()
        rng = np.random.default_rng()
        draws = rng.choice(len(probs), size=shots, p=probs / probs.sum())
        counts = {}
        for outcome in draws:
            key = format(int(outcome), f"0{self.n_qubits}b")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
