"""A dependency-free statevector simulator with a batch dimension.

The state is a complex tensor of shape ``(batch, 2, 2, ..., 2)``. Applying a
gate is a reshape plus one matmul, and the batch axis rides along for free --
which matters because the two hot loops in this package (encoding a dataset,
and evaluating parameter-shift gradients) are both "run the same circuit a few
hundred times with different angles".

Angles may vary per batch element, so a single call covers all the shifted
parameter sets a gradient needs.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np

from ..circuits.circuit import Circuit, Gate, Ref

__all__ = ["StatevectorSimulator", "PAULI", "pauli_matrix"]

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex)

PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def pauli_matrix(letter: str) -> np.ndarray:
    return PAULI[letter.upper()]


# -- gate matrices; every builder accepts a scalar or a (batch,) array --------

def _rows(*rows) -> np.ndarray:
    """Stack rows of entries into a (..., n, n) matrix, broadcasting entries."""
    shape = np.broadcast_shapes(*[np.asarray(e).shape for row in rows for e in row])
    size = len(rows)
    out = np.zeros(shape + (size, size), dtype=complex)
    for i, row in enumerate(rows):
        for j, entry in enumerate(row):
            out[..., i, j] = entry
    return out


def _rx(t):
    t = np.asarray(t, dtype=float)
    c, s = np.cos(t / 2), -1j * np.sin(t / 2)
    return _rows([c, s], [s, c])


def _ry(t):
    t = np.asarray(t, dtype=float)
    c, s = np.cos(t / 2), np.sin(t / 2)
    return _rows([c, -s], [s, c])


def _rz(t):
    t = np.asarray(t, dtype=float)
    zero = np.zeros_like(t)
    return _rows([np.exp(-0.5j * t), zero], [zero, np.exp(0.5j * t)])


def _phase(t):
    t = np.asarray(t, dtype=float)
    return _rows([np.ones_like(t), np.zeros_like(t)], [np.zeros_like(t), np.exp(1j * t)])


def _controlled(u: np.ndarray) -> np.ndarray:
    out = np.zeros(u.shape[:-2] + (4, 4), dtype=complex)
    out[..., 0, 0] = 1.0
    out[..., 1, 1] = 1.0
    out[..., 2:, 2:] = u
    return out


def _rzz(t):
    t = np.asarray(t, dtype=float)
    zero = np.zeros_like(t)
    minus, plus = np.exp(-0.5j * t), np.exp(0.5j * t)
    return _rows([minus, zero, zero, zero], [zero, plus, zero, zero],
                 [zero, zero, plus, zero], [zero, zero, zero, minus])


def _rxx(t):
    t = np.asarray(t, dtype=float)
    c, s = np.cos(t / 2) + 0j, -1j * np.sin(t / 2)
    zero = np.zeros_like(c)
    return _rows([c, zero, zero, s], [zero, c, s, zero],
                 [zero, s, c, zero], [s, zero, zero, c])


_CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)

_ONE_QUBIT = {
    "h": lambda p: _H, "x": lambda p: _X, "y": lambda p: _Y, "z": lambda p: _Z,
    "s": lambda p: _S, "t": lambda p: _T,
    "rx": lambda p: _rx(p[0]), "ry": lambda p: _ry(p[0]), "rz": lambda p: _rz(p[0]),
    "phase": lambda p: _phase(p[0]),
}

_TWO_QUBIT = {
    "cnot": lambda p: _CNOT, "cz": lambda p: _CZ, "swap": lambda p: _SWAP,
    "crx": lambda p: _controlled(_rx(p[0])),
    "cry": lambda p: _controlled(_ry(p[0])),
    "crz": lambda p: _controlled(_rz(p[0])),
    "rzz": lambda p: _rzz(p[0]), "rxx": lambda p: _rxx(p[0]),
}


class StatevectorSimulator:
    """Exact simulation of a :class:`~openqml.circuits.Circuit`, batched."""

    name = "statevector"
    supports_batch = True

    def __init__(self, n_qubits: int, seed=None, shots: Optional[int] = None):
        if n_qubits > 24:
            raise ValueError("this simulator is capped at 24 qubits (16M amplitudes)")
        self.n_qubits = int(n_qubits)
        self.shots = shots
        self._rng = np.random.default_rng(seed)
        self._sign_cache: Dict[int, np.ndarray] = {}
        self.reset()

    # -- state ------------------------------------------------------------
    def reset(self, batch: int = 1) -> "StatevectorSimulator":
        self._batch = int(batch)
        self._state = np.zeros((self._batch,) + (2,) * self.n_qubits, dtype=complex)
        self._state[(slice(None),) + (0,) * self.n_qubits] = 1.0
        return self

    def set_state(self, vector) -> "StatevectorSimulator":
        vector = np.asarray(vector, dtype=complex)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.shape[-1] != 2 ** self.n_qubits:
            raise ValueError(f"expected {2 ** self.n_qubits} amplitudes, got {vector.shape[-1]}")
        norms = np.linalg.norm(vector, axis=-1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("cannot normalise the zero vector")
        self._batch = vector.shape[0]
        self._state = (vector / norms).reshape((self._batch,) + (2,) * self.n_qubits)
        return self

    @property
    def batch_size(self) -> int:
        return self._batch

    @property
    def states(self) -> np.ndarray:
        """All statevectors as ``(batch, 2**n)``; qubit 0 is most significant."""
        return self._state.reshape(self._batch, -1).copy()

    @property
    def state(self) -> np.ndarray:
        """The single statevector, for an unbatched run."""
        if self._batch != 1:
            raise ValueError(f"this run holds {self._batch} states; use .states instead")
        return self._state.reshape(-1).copy()

    # -- gates ------------------------------------------------------------
    def apply_1q(self, matrix: np.ndarray, wire: int) -> None:
        axis = int(wire) + 1
        moved = np.moveaxis(self._state, axis, -1)
        shape = moved.shape
        flat = moved.reshape(shape[0], -1, 2)
        out = flat @ np.swapaxes(np.asarray(matrix), -1, -2)
        self._state = np.moveaxis(out.reshape(shape), -1, axis)

    def apply_2q(self, matrix: np.ndarray, wire_a: int, wire_b: int) -> None:
        axes = [int(wire_a) + 1, int(wire_b) + 1]
        moved = np.moveaxis(self._state, axes, [-2, -1])
        shape = moved.shape
        flat = moved.reshape(shape[0], -1, 4)
        out = flat @ np.swapaxes(np.asarray(matrix), -1, -2)
        self._state = np.moveaxis(out.reshape(shape), [-2, -1], axes)

    def _resolve(self, param, weights, features):
        if not isinstance(param, Ref):
            return float(param)
        source = weights if param.kind == "weights" else features
        if source is None:
            raise ValueError(f"circuit needs a value for {param.kind}[{param.index}]")
        if source.ndim == 1:
            if param.index >= source.shape[0]:
                raise ValueError(f"{param.kind}[{param.index}] is out of range")
            return float(np.real(source[param.index])) * param.scale + param.offset
        if param.index >= source.shape[1]:
            raise ValueError(f"{param.kind}[{param.index}] is out of range")
        return np.real(source[:, param.index]) * param.scale + param.offset

    def apply_gate(self, gate: Gate, weights=None, features=None) -> None:
        if gate.name == "amplitude_embedding":
            self.set_state(np.asarray(features, dtype=complex))
            return
        params = tuple(self._resolve(p, weights, features) for p in gate.params)
        if gate.name in _ONE_QUBIT:
            self.apply_1q(_ONE_QUBIT[gate.name](params), gate.wires[0])
        elif gate.name in _TWO_QUBIT:
            self.apply_2q(_TWO_QUBIT[gate.name](params), gate.wires[0], gate.wires[1])
        else:
            raise ValueError(f"gate {gate.name!r} is not supported by {self.name}")

    # -- execution --------------------------------------------------------
    @staticmethod
    def _prepare(array, batch: int):
        if array is None:
            return None
        array = np.asarray(array)
        if array.ndim == 2 and array.shape[0] == 1 and batch > 1:
            array = np.repeat(array, batch, axis=0)
        return array

    def run_batch(self, circuit: Circuit, weights=None, features=None) -> "StatevectorSimulator":
        """Run one circuit for a whole batch of weight/feature rows.

        ``weights`` and ``features`` may be 1-D (shared by the batch) or 2-D
        (one row per batch element); the batch size is taken from whichever is
        2-D. Mixing the two is the case that makes gradients cheap.
        """
        batch = 1
        for array in (weights, features):
            if array is not None and np.ndim(array) == 2:
                batch = max(batch, np.shape(array)[0])
        weights = self._prepare(weights, batch)
        features = self._prepare(features, batch)
        self.reset(batch)
        for gate in circuit.gates:
            self.apply_gate(gate, weights, features)
        return self

    def run(self, circuit: Circuit, weights=None, features=None) -> "StatevectorSimulator":
        """Single-shot convenience wrapper around :meth:`run_batch`."""
        weights = None if weights is None else np.asarray(weights).reshape(-1)
        features = None if features is None else np.asarray(features).reshape(-1)
        return self.run_batch(circuit, weights, features)

    # -- measurement ------------------------------------------------------
    def probabilities(self) -> np.ndarray:
        """``(batch, 2**n)`` outcome probabilities."""
        return np.abs(self._state.reshape(self._batch, -1)) ** 2

    def _z_signs(self, wire: int) -> np.ndarray:
        """(-1)**bit_of_wire over all basis states, cached per wire."""
        if wire not in self._sign_cache:
            indices = np.arange(2 ** self.n_qubits)
            bits = (indices >> (self.n_qubits - 1 - int(wire))) & 1
            self._sign_cache[wire] = 1.0 - 2.0 * bits
        return self._sign_cache[wire]

    def _shot_noise(self, exact):
        """Replace exact expectation values by their finite-sample estimates."""
        exact = np.asarray(exact, dtype=float)
        p_plus = np.clip((exact + 1.0) / 2.0, 0.0, 1.0)
        counts = self._rng.binomial(self.shots, p_plus)
        return 2.0 * counts / self.shots - 1.0

    def z_expvals(self, wires: Sequence[int]) -> np.ndarray:
        """``(batch, len(wires))`` values of <Z> -- read straight off the probabilities."""
        probs = self.probabilities()
        out = np.column_stack([probs @ self._z_signs(int(q)) for q in wires])
        return self._shot_noise(out) if self.shots else out

    def expval_batch(self, observable: Mapping[int, str]) -> np.ndarray:
        """``(batch,)`` expectation values of one Pauli word."""
        letters = {int(w): l.upper() for w, l in observable.items() if str(l).upper() != "I"}
        if not letters:
            return np.ones(self._batch)
        if all(letter == "Z" for letter in letters.values()):
            signs = np.ones(2 ** self.n_qubits)
            for wire in letters:
                signs = signs * self._z_signs(wire)
            values = self.probabilities() @ signs
        else:
            phi = self._state
            for wire, letter in letters.items():
                moved = np.moveaxis(phi, wire + 1, -1)
                shape = moved.shape
                flat = moved.reshape(shape[0], -1, 2)
                phi = np.moveaxis((flat @ pauli_matrix(letter).T).reshape(shape), -1, wire + 1)
            psi = self._state.reshape(self._batch, -1)
            values = np.real(np.einsum("bi,bi->b", psi.conj(), phi.reshape(self._batch, -1)))
        return self._shot_noise(values) if self.shots else values

    def expval(self, observable: Mapping[int, str]) -> float:
        values = self.expval_batch(observable)
        if len(values) != 1:
            raise ValueError(f"this run holds {self._batch} states; use expval_batch()")
        return float(values[0])

    def energies(self, terms: Sequence) -> np.ndarray:
        """``(batch,)`` energies of a Hamiltonian given as ``[(coeff, word), ...]``."""
        total = np.zeros(self._batch)
        for coefficient, word in terms:
            total = total + float(coefficient) * self.expval_batch(word)
        return total

    def expval_hamiltonian(self, terms: Sequence) -> float:
        return float(self.energies(terms)[0])

    def sample(self, shots: int = 1024) -> Dict[str, int]:
        probs = self.probabilities()[0]
        draws = self._rng.choice(len(probs), size=int(shots), p=probs / probs.sum())
        counts: Dict[str, int] = {}
        for outcome in draws:
            key = format(int(outcome), f"0{self.n_qubits}b")
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def __repr__(self) -> str:
        return (f"<StatevectorSimulator qubits={self.n_qubits} batch={self._batch} "
                f"shots={self.shots}>")
