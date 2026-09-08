"""A dependency-free statevector simulator with a batch dimension.

The state is a complex array of shape ``(batch, 2**n)``. Applying a gate never
permutes or copies that array: the amplitudes a gate mixes are already
contiguous slices of it, so reshaping to ``(batch, left, 2, right)`` -- a free
view -- exposes the two halves the gate combines, and the gate is a handful of
elementwise operations written straight back in place.

That matters because the two hot loops in this package (encoding a dataset, and
evaluating parameter-shift gradients) are both "run the same circuit a few
hundred times with different angles". The batch axis rides along for free, and
angles may vary per batch element, so a single call covers all the shifted
parameter sets a gradient needs.

Diagonal gates (``rz``, ``phase``, ``rzz``, ``cz``, ...) and permutation gates
(``x``, ``cnot``, ``swap``) never build a matrix at all -- they are a scaling or
a swap of those slices, which is where most of the gates in the bundled feature
maps and ansaetze land.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

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

MAX_QUBITS = 24
_INV_SQRT2 = 1.0 / math.sqrt(2)


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
    return _rows(*_rx_entries((t,)))


def _ry(t):
    return _rows(*_ry_entries((t,)))


def _rz(t):
    d0, d1 = _rz_diagonal((t,))
    zero = np.zeros_like(d0)
    return _rows([d0, zero], [zero, d1])


def _phase(t):
    d0, d1 = _phase_diagonal((t,))
    zero = np.zeros_like(d1)
    return _rows([np.ones_like(d1), zero], [zero, d1])


def _controlled(u: np.ndarray) -> np.ndarray:
    out = np.zeros(u.shape[:-2] + (4, 4), dtype=complex)
    out[..., 0, 0] = 1.0
    out[..., 1, 1] = 1.0
    out[..., 2:, 2:] = u
    return out


def _rzz(t):
    minus, plus = _rz_diagonal((t,))
    zero = np.zeros_like(minus)
    return _rows([minus, zero, zero, zero], [zero, plus, zero, zero],
                 [zero, zero, plus, zero], [zero, zero, zero, minus])


def _rxx(t):
    c, s = _rxx_entries((t,))
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


# -- gate entries: what the fast paths use instead of a matrix ----------------
#
# Every builder below returns the *non-trivial entries* of a gate rather than a
# dense matrix. For a per-batch angle that is the difference between a handful
# of ``(batch,)`` vectors and a full ``(batch, 2, 2)`` array of mostly zeros,
# and it lets the applier skip the structural zeros entirely.

def _rx_entries(p):
    t = np.asarray(p[0], dtype=float)
    c, s = np.cos(t / 2) + 0j, -1j * np.sin(t / 2)
    return [c, s], [s, c]


def _ry_entries(p):
    t = np.asarray(p[0], dtype=float)
    c, s = np.cos(t / 2), np.sin(t / 2)
    return [c, -s], [s, c]


def _unit_phase(angle):
    """``exp(1j * angle)`` built from one cos/sin pair.

    ``np.exp`` on a complex argument is markedly slower than the two real
    transcendentals it reduces to here, and these are called once per gate per
    batch element -- the innermost loop in the package.
    """
    angle = np.asarray(angle, dtype=float)
    if angle.ndim == 0:
        return complex(math.cos(angle), math.sin(angle))
    out = np.empty(angle.shape, dtype=complex)
    out.real = np.cos(angle)
    out.imag = np.sin(angle)
    return out


def _rz_diagonal(p):
    plus = _unit_phase(0.5 * np.asarray(p[0], dtype=float))
    return np.conj(plus), plus


def _phase_diagonal(p):
    return 1.0, _unit_phase(p[0])


def _rzz_diagonal(p):
    minus, plus = _rz_diagonal(p)
    return minus, plus, plus, minus


def _crz_diagonal(p):
    minus, plus = _rz_diagonal(p)
    return 1.0, 1.0, minus, plus


def _rxx_entries(p):
    t = np.asarray(p[0], dtype=float)
    return np.cos(t / 2) + 0j, -1j * np.sin(t / 2)


#: name -> the two diagonal entries of a single-qubit gate
_DIAGONAL_1Q = {
    "z": lambda p: (1.0, -1.0),
    "s": lambda p: (1.0, 1j),
    "t": lambda p: (1.0, np.exp(1j * math.pi / 4)),
    "rz": _rz_diagonal,
    "phase": _phase_diagonal,
}

#: name -> the four diagonal entries of a two-qubit gate
_DIAGONAL_2Q = {
    "cz": lambda p: (1.0, 1.0, 1.0, -1.0),
    "rzz": _rzz_diagonal,
    "crz": _crz_diagonal,
}

#: name -> the four entries [[m00, m01], [m10, m11]] of a single-qubit gate
_DENSE_1Q = {
    "y": lambda p: ([0.0, -1j], [1j, 0.0]),
    "rx": _rx_entries,
    "ry": _ry_entries,
}

#: name -> the entries of the single-qubit gate applied when the control is set
_CONTROLLED_1Q = {"crx": _rx_entries, "cry": _ry_entries}

#: name -> (a, b) index pair whose amplitude blocks the gate exchanges
_PERMUTATION_2Q = {"cnot": (2, 3), "swap": (1, 2)}


def _conjugate(entries, dagger: bool):
    """The diagonal (or symmetric-matrix entries) of the inverse gate."""
    return tuple(np.conj(e) for e in entries) if dagger else entries


def _transpose(entries, dagger: bool):
    """``[[m00, m01], [m10, m11]]`` conjugate-transposed, for the inverse gate."""
    if not dagger:
        return entries
    (m00, m01), (m10, m11) = entries
    return [np.conj(m00), np.conj(m10)], [np.conj(m01), np.conj(m11)]


def _broadcast(value, ndim: int):
    """Shape a scalar or per-batch factor so it broadcasts over a block."""
    array = np.asarray(value)
    if array.ndim == 0:
        return array[()]
    return array.reshape((array.shape[0],) + (1,) * (ndim - 1))


class StatevectorSimulator:
    """Exact simulation of a :class:`~openqml.circuits.Circuit`, batched."""

    name = "statevector"
    supports_batch = True

    def __init__(self, n_qubits: int, seed=None, shots: Optional[int] = None):
        if n_qubits > MAX_QUBITS:
            raise ValueError(f"this simulator is capped at {MAX_QUBITS} qubits (16M amplitudes)")
        self.n_qubits = int(n_qubits)
        self.dimension = 1 << self.n_qubits
        self.shots = shots
        self._rng = np.random.default_rng(seed)
        self._sign_cache: Dict[int, np.ndarray] = {}
        self._sign_matrix_cache: Dict[tuple, np.ndarray] = {}
        self._diagonal_cache: Dict[tuple, tuple] = {}
        self.reset()

    # -- state ------------------------------------------------------------
    def reset(self, batch: int = 1) -> "StatevectorSimulator":
        self._batch = int(batch)
        self._run_batch = self._batch
        self._state = np.zeros((self._batch, self.dimension), dtype=complex)
        self._state[:, 0] = 1.0
        return self

    def set_state(self, vector) -> "StatevectorSimulator":
        vector = np.asarray(vector, dtype=complex)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        if vector.shape[-1] != self.dimension:
            raise ValueError(f"expected {self.dimension} amplitudes, got {vector.shape[-1]}")
        norms = np.linalg.norm(vector, axis=-1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("cannot normalise the zero vector")
        self._batch = vector.shape[0]
        self._state = np.ascontiguousarray(vector / norms)
        return self

    @property
    def batch_size(self) -> int:
        return self._batch

    @property
    def states(self) -> np.ndarray:
        """All statevectors as ``(batch, 2**n)``; qubit 0 is most significant."""
        return self._state.copy()

    @property
    def state(self) -> np.ndarray:
        """The single statevector, for an unbatched run."""
        if self._batch != 1:
            raise ValueError(f"this run holds {self._batch} states; use .states instead")
        return self._state[0].copy()

    # -- amplitude blocks -------------------------------------------------
    #
    # Qubit 0 is the most significant bit, so the amplitudes that differ only in
    # the value of ``wire`` sit a fixed stride apart: reshaping the state to
    # (batch, left, 2, right) is a view, never a copy, and the axis of length 2
    # is exactly the subspace a single-qubit gate acts on.

    def _split_1q(self, wire: int, state=None) -> np.ndarray:
        """``(batch, left, 2, right)`` view of the state, split on ``wire``."""
        wire = int(wire)
        if not 0 <= wire < self.n_qubits:
            raise ValueError(f"wire {wire} out of range for {self.n_qubits} qubits")
        state = self._state if state is None else state
        right = 1 << (self.n_qubits - wire - 1)
        return state.reshape(state.shape[0], 1 << wire, 2, right)

    def _split_2q(self, wire_a: int, wire_b: int, state=None):
        """The four amplitude blocks a two-qubit gate mixes, in gate order.

        Block ``2 * bit(wire_a) + bit(wire_b)`` matches the row/column ordering
        of a 4x4 gate matrix, whichever way round the two wires happen to be.
        """
        wire_a, wire_b = int(wire_a), int(wire_b)
        if wire_a == wire_b:
            raise ValueError("a two-qubit gate needs two distinct wires")
        for wire in (wire_a, wire_b):
            if not 0 <= wire < self.n_qubits:
                raise ValueError(f"wire {wire} out of range for {self.n_qubits} qubits")
        state = self._state if state is None else state
        low, high = min(wire_a, wire_b), max(wire_a, wire_b)
        view = state.reshape(state.shape[0], 1 << low, 2, 1 << (high - low - 1), 2,
                             1 << (self.n_qubits - high - 1))
        ordered = wire_a < wire_b
        blocks = [None] * 4
        for bit_low in (0, 1):
            for bit_high in (0, 1):
                index = 2 * bit_low + bit_high if ordered else 2 * bit_high + bit_low
                blocks[index] = view[:, :, bit_low, :, bit_high, :]
        return blocks

    # -- gates ------------------------------------------------------------
    def apply_1q(self, matrix: np.ndarray, wire: int) -> None:
        """Apply a dense 2x2 (or batched ``(batch, 2, 2)``) gate to ``wire``."""
        matrix = np.asarray(matrix)
        self._apply_entries_1q(([matrix[..., 0, 0], matrix[..., 0, 1]],
                                [matrix[..., 1, 0], matrix[..., 1, 1]]), wire)

    def apply_2q(self, matrix: np.ndarray, wire_a: int, wire_b: int) -> None:
        """Apply a dense 4x4 (or batched ``(batch, 4, 4)``) gate to two wires."""
        matrix = np.asarray(matrix)
        blocks = self._split_2q(wire_a, wire_b)
        updated = []
        for row in range(4):
            total = None
            for column in range(4):
                entry = matrix[..., row, column]
                if not entry.any():
                    continue
                term = _broadcast(entry, 4) * blocks[column]
                total = term if total is None else total + term
            updated.append(total)
        for block, value in zip(blocks, updated):
            if value is None:
                block[...] = 0.0
            else:
                block[...] = value

    def _apply_entries_1q(self, entries, wire: int, state=None) -> None:
        """Apply ``[[m00, m01], [m10, m11]]`` without assembling a matrix."""
        view = self._split_1q(wire, state)
        lower, upper = view[:, :, 0, :], view[:, :, 1, :]
        (m00, m01), (m10, m11) = entries
        new_lower = _broadcast(m00, 3) * lower + _broadcast(m01, 3) * upper
        new_upper = _broadcast(m10, 3) * lower + _broadcast(m11, 3) * upper
        lower[...] = new_lower
        upper[...] = new_upper

    def _apply_diagonal_1q(self, diagonal, wire: int, state=None) -> None:
        """Scale each half of the ``wire`` axis -- no temporaries, in place."""
        view = self._split_1q(wire, state)
        for bit, entry in enumerate(diagonal):
            if np.ndim(entry) == 0 and entry == 1.0:
                continue
            view[:, :, bit, :] *= _broadcast(entry, 3)

    def _apply_diagonal_2q(self, diagonal, wire_a: int, wire_b: int, state=None) -> None:
        blocks = self._split_2q(wire_a, wire_b, state)
        for block, entry in zip(blocks, diagonal):
            if np.ndim(entry) == 0 and entry == 1.0:
                continue
            block *= _broadcast(entry, 4)

    def _apply_controlled_1q(self, entries, control: int, target: int, state=None) -> None:
        """A single-qubit gate on the ``control == 1`` half of the register."""
        blocks = self._split_2q(control, target, state)
        lower, upper = blocks[2], blocks[3]
        (m00, m01), (m10, m11) = entries
        new_lower = _broadcast(m00, 4) * lower + _broadcast(m01, 4) * upper
        new_upper = _broadcast(m10, 4) * lower + _broadcast(m11, 4) * upper
        lower[...] = new_lower
        upper[...] = new_upper

    @staticmethod
    def _exchange(first: np.ndarray, second: np.ndarray) -> None:
        held = first.copy()
        first[...] = second
        second[...] = held

    def _apply_named(self, name: str, params, wires, state=None, dagger: bool = False) -> bool:
        """Run a gate through its structure-aware path; False if there is none.

        ``dagger`` applies the inverse. Every gate here is either self-inverse,
        diagonal (conjugate the diagonal) or given by four entries (conjugate
        transpose them), so the inverse costs nothing extra -- which is what
        makes the backward sweep of :meth:`z_jacobian` affordable.
        """
        builder = _DIAGONAL_1Q.get(name)
        if builder is not None:
            self._apply_diagonal_1q(_conjugate(builder(params), dagger), wires[0], state)
            return True
        builder = _DENSE_1Q.get(name)
        if builder is not None:
            self._apply_entries_1q(_transpose(builder(params), dagger), wires[0], state)
            return True
        if name == "h":
            # sum and difference, then one scaling each: four passes over the
            # state where the generic 2x2 needs six. Every feature map opens
            # with a layer of these, so it is worth the special case.
            view = self._split_1q(wires[0], state)
            lower, upper = view[:, :, 0, :], view[:, :, 1, :]
            total = lower + upper
            total *= _INV_SQRT2
            difference = lower - upper
            difference *= _INV_SQRT2
            lower[...] = total
            upper[...] = difference
            return True
        if name == "x":
            view = self._split_1q(wires[0], state)
            self._exchange(view[:, :, 0, :], view[:, :, 1, :])
            return True
        builder = _DIAGONAL_2Q.get(name)
        if builder is not None:
            self._apply_diagonal_2q(_conjugate(builder(params), dagger), wires[0], wires[1], state)
            return True
        builder = _CONTROLLED_1Q.get(name)
        if builder is not None:
            self._apply_controlled_1q(_transpose(builder(params), dagger),
                                      wires[0], wires[1], state)
            return True
        pair = _PERMUTATION_2Q.get(name)
        if pair is not None:
            blocks = self._split_2q(wires[0], wires[1], state)
            self._exchange(blocks[pair[0]], blocks[pair[1]])
            return True
        if name == "rxx":
            blocks = self._split_2q(wires[0], wires[1], state)
            # symmetric, so the inverse is the elementwise conjugate
            cosine, sine = (_broadcast(e, 4) for e in _conjugate(_rxx_entries(params), dagger))
            b0, b1, b2, b3 = blocks
            new = (cosine * b0 + sine * b3, cosine * b1 + sine * b2,
                   sine * b1 + cosine * b2, sine * b0 + cosine * b3)
            for block, value in zip(blocks, new):
                block[...] = value
            return True
        return False

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
            self._embed(features)
            return
        params = tuple(self._resolve(p, weights, features) for p in gate.params)
        self._apply_resolved(gate, params)

    def _embed(self, features) -> None:
        """Load features into the amplitudes, keeping the run's batch size.

        One feature vector shared by a batch of weight rows is the ordinary case
        for a gradient, and it must not collapse the batch the way a bare
        :meth:`set_state` would.
        """
        self.set_state(np.asarray(features, dtype=complex))
        batch = getattr(self, "_run_batch", 1)
        if self._batch == 1 and batch > 1:
            self._state = np.repeat(self._state, batch, axis=0)
            self._batch = batch

    def _apply_resolved(self, gate: Gate, params, state=None, dagger: bool = False) -> None:
        """Apply a gate whose angles are already numbers (or per-batch arrays)."""
        if self._apply_named(gate.name, params, gate.wires, state, dagger):
            return
        if dagger:  # pragma: no cover - every supported name has a fast path
            raise ValueError(f"gate {gate.name!r} cannot be inverted by {self.name}")
        if gate.name in _ONE_QUBIT:  # pragma: no cover - every name has a fast path
            self.apply_1q(_ONE_QUBIT[gate.name](params), gate.wires[0])
        elif gate.name in _TWO_QUBIT:  # pragma: no cover - every name has a fast path
            self.apply_2q(_TWO_QUBIT[gate.name](params), gate.wires[0], gate.wires[1])
        else:
            raise ValueError(f"gate {gate.name!r} is not supported by {self.name}")

    # -- generators, for differentiation -----------------------------------
    #
    # Every parameterised gate here is ``exp(-i * theta * G / 2)`` for a
    # Hermitian ``G`` built from Paulis (``phase`` included, with ``G = Z - I``,
    # and the controlled rotations with ``G = |1><1| (x) P``). The adjoint
    # method needs ``G|psi>`` and nothing else about the gate.

    #: name -> the Pauli letter its generator applies to each of its wires
    _GENERATORS = {
        "rx": "X", "ry": "Y", "rz": "Z",
        "rzz": "ZZ", "rxx": "XX",
        "crx": "X", "cry": "Y", "crz": "Z",
    }

    def is_differentiable(self, name: str) -> bool:
        """True when the adjoint method knows this gate's generator."""
        return name in self._GENERATORS or name == "phase"

    def _apply_generator(self, name: str, wires, state: np.ndarray) -> np.ndarray:
        """``G |psi>`` in place, for the generator of a parameterised gate."""
        if name == "phase":  # G = Z - I = diag(0, -2)
            view = self._split_1q(wires[0], state)
            view[:, :, 0, :] = 0.0
            view[:, :, 1, :] *= -2.0
            return state
        letters = self._GENERATORS.get(name)
        if letters is None:
            raise ValueError(f"gate {name!r} has no generator registered for differentiation")
        if name in _CONTROLLED_1Q or name == "crz":  # G = |1><1| (x) P
            blocks = self._split_2q(wires[0], wires[1], state)
            blocks[0][...] = 0.0
            blocks[1][...] = 0.0
            self._apply_pauli_pair(blocks[2], blocks[3], letters)
            return state
        return self._apply_pauli_word(state, dict(zip(wires, letters)))

    @staticmethod
    def _apply_pauli_pair(lower: np.ndarray, upper: np.ndarray, letter: str) -> None:
        """One Pauli acting on the two halves of a wire, in place."""
        if letter == "Z":
            upper *= -1.0
        elif letter == "X":
            StatevectorSimulator._exchange(lower, upper)
        elif letter == "Y":
            held = lower.copy()
            lower[...] = -1j * upper
            upper[...] = 1j * held
        else:
            raise ValueError(f"unknown Pauli letter {letter!r}")

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
        self._run_batch = batch
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
        state = self._state
        return state.real ** 2 + state.imag ** 2

    def _z_signs(self, wire: int) -> np.ndarray:
        """(-1)**bit_of_wire over all basis states, cached per wire."""
        signs = self._sign_cache.get(wire)
        if signs is None:
            block = np.empty(1 << (self.n_qubits - wire), dtype=float)
            half = block.size // 2
            block[:half] = 1.0
            block[half:] = -1.0
            signs = np.tile(block, 1 << wire)
            self._sign_cache[wire] = signs
        return signs

    def _z_sign_matrix(self, wires: Sequence[int]) -> np.ndarray:
        """``(2**n, len(wires))`` sign columns, so <Z> is a single matmul."""
        key = tuple(int(q) for q in wires)
        matrix = self._sign_matrix_cache.get(key)
        if matrix is None:
            matrix = np.column_stack([self._z_signs(q) for q in key]) if key \
                else np.zeros((self.dimension, 0))
            self._sign_matrix_cache[key] = matrix
        return matrix

    def _shot_noise(self, exact):
        """Replace exact expectation values by their finite-sample estimates."""
        exact = np.asarray(exact, dtype=float)
        p_plus = np.clip((exact + 1.0) / 2.0, 0.0, 1.0)
        counts = self._rng.binomial(self.shots, p_plus)
        return 2.0 * counts / self.shots - 1.0

    def z_expvals(self, wires: Sequence[int]) -> np.ndarray:
        """``(batch, len(wires))`` values of <Z> -- read straight off the probabilities."""
        out = self.probabilities() @ self._z_sign_matrix(wires)
        return self._shot_noise(out) if self.shots else out

    @staticmethod
    def _letters_of(observable: Mapping[int, str]) -> Dict[int, str]:
        return {int(w): l.upper() for w, l in observable.items() if str(l).upper() != "I"}

    def _apply_pauli_word(self, state: np.ndarray, letters: Mapping[int, str]) -> np.ndarray:
        """``P |psi>`` for a Pauli word, in place on a state we already own."""
        for wire, letter in letters.items():
            view = self._split_1q(wire, state)
            self._apply_pauli_pair(view[:, :, 0, :], view[:, :, 1, :], letter)
        return state

    def expval_batch(self, observable: Mapping[int, str]) -> np.ndarray:
        """``(batch,)`` expectation values of one Pauli word."""
        letters = self._letters_of(observable)
        if not letters:
            return np.ones(self._batch)
        if all(letter == "Z" for letter in letters.values()):
            signs = np.ones(self.dimension)
            for wire in letters:
                signs = signs * self._z_signs(wire)
            values = self.probabilities() @ signs
        else:
            phi = self._apply_pauli_word(self._state.copy(), letters)
            values = np.einsum("bi,bi->b", self._state.real, phi.real)
            values += np.einsum("bi,bi->b", self._state.imag, phi.imag)
        return self._shot_noise(values) if self.shots else values

    def expval(self, observable: Mapping[int, str]) -> float:
        values = self.expval_batch(observable)
        if len(values) != 1:
            raise ValueError(f"this run holds {self._batch} states; use expval_batch()")
        return float(values[0])

    def _grouped_terms(self, terms: Sequence):
        """Split a Pauli sum into one diagonal vector plus the remaining words.

        Every all-Z term is diagonal in the computational basis, so the whole
        group collapses into a single ``2**n`` vector and one matmul against the
        probabilities -- rather than one pass over the state per term. The
        result is cached, because a VQE run asks for the same Hamiltonian once
        per optimiser step.
        """
        key = tuple((float(c), tuple(sorted(self._letters_of(word).items()))) for c, word in terms)
        grouped = self._diagonal_cache.get(key)
        if grouped is None:
            constant, diagonal, rest = 0.0, None, []
            for coefficient, letters in key:
                if not letters:
                    constant += coefficient
                elif all(letter == "Z" for _, letter in letters):
                    signs = np.full(self.dimension, coefficient)
                    for wire, _ in letters:
                        signs *= self._z_signs(wire)
                    diagonal = signs if diagonal is None else diagonal + signs
                else:
                    rest.append((coefficient, dict(letters)))
            grouped = (constant, diagonal, tuple(rest))
            self._diagonal_cache[key] = grouped
        return grouped

    def energies(self, terms: Sequence) -> np.ndarray:
        """``(batch,)`` energies of a Hamiltonian given as ``[(coeff, word), ...]``."""
        if self.shots:
            # each term is a separate measurement, so it carries its own noise
            total = np.zeros(self._batch)
            for coefficient, word in terms:
                total = total + float(coefficient) * self.expval_batch(word)
            return total
        constant, diagonal, rest = self._grouped_terms(terms)
        total = np.full(self._batch, constant)
        if diagonal is not None:
            total += self.probabilities() @ diagonal
        for coefficient, word in rest:
            total += coefficient * self.expval_batch(word)
        return total

    def expval_hamiltonian(self, terms: Sequence) -> float:
        return float(self.energies(terms)[0])

    # -- adjoint differentiation -------------------------------------------
    #
    # The parameter-shift rule costs two *whole circuit* evaluations per gate
    # parameter, so a gradient is O(P) runs of an O(P)-gate circuit: O(P**2)
    # gate applications. On an exact simulator the same numbers come out of a
    # single backward sweep.
    #
    # Writing f(theta) = <psi|O|psi> with |psi> = U_P...U_1|0>, and carrying
    # both |psi_j> = U_j...U_1|0> and |b_j> = (U_j+1...U_P)^dag O |psi>
    # backwards from j = P, the derivative for gate j is just
    #
    #     df/dtheta_j = Im( <b_j| G_j |psi_j> )
    #
    # for the gate's generator G_j. Each step un-applies one gate from |psi>
    # and from each |b> -- both cheap, because every gate here has a
    # closed-form inverse -- so the whole gradient is O(P) state passes.
    #
    # This is exact, not an approximation: it agrees with the shift rule to
    # machine precision (the test suite pins that). It is *analytic*, though,
    # so it has no meaning under shot noise, and unlike the two-term shift rule
    # it stays correct when one weight drives several gates.

    def _adjoint_jacobian(self, circuit: Circuit, adjoints, resolved, features,
                          n_parameters: int) -> np.ndarray:
        """``(batch, n_observables, n_parameters)`` from one backward sweep."""
        psi = self._state
        jacobian = np.zeros((self._batch, len(adjoints), n_parameters))
        for position in range(len(circuit.gates) - 1, -1, -1):
            gate = circuit.gates[position]
            if gate.name == "amplitude_embedding":
                break  # state preparation; nothing before it carries a weight
            params = resolved[position]
            weights_here = [p for p in gate.params
                            if isinstance(p, Ref) and p.kind == "weights"]
            if weights_here:
                derivative = self._apply_generator(gate.name, gate.wires, psi.copy())
                for column, adjoint in enumerate(adjoints):
                    overlap = (np.einsum("bi,bi->b", adjoint.real, derivative.imag)
                               - np.einsum("bi,bi->b", adjoint.imag, derivative.real))
                    for ref in weights_here:
                        # the gate angle is scale * w[index] + offset
                        jacobian[:, column, ref.index] += ref.scale * overlap
            self._apply_resolved(gate, params, psi, dagger=True)
            for adjoint in adjoints:
                self._apply_resolved(gate, params, adjoint, dagger=True)
        return jacobian

    def _forward(self, circuit: Circuit, weights, features):
        """Run the circuit, keeping every gate's resolved angles for the sweep."""
        if self.shots:
            raise ValueError("adjoint differentiation is analytic; it has no "
                             "meaning on a device with shot noise")
        batch = 1
        for array in (weights, features):
            if array is not None and np.ndim(array) == 2:
                batch = max(batch, np.shape(array)[0])
        weights = self._prepare(weights, batch)
        features = self._prepare(features, batch)
        self._run_batch = batch
        self.reset(batch)
        resolved = []
        for gate in circuit.gates:
            if gate.name == "amplitude_embedding":
                resolved.append(())
                self._embed(features)
                continue
            params = tuple(self._resolve(p, weights, features) for p in gate.params)
            resolved.append(params)
            self._apply_resolved(gate, params)
        return resolved, features

    def z_jacobian(self, circuit: Circuit, wires: Sequence[int], weights=None, features=None):
        """``(<Z>, d<Z>/dweights)`` shaped ``(batch, W)`` and ``(batch, W, P)``."""
        resolved, features = self._forward(circuit, weights, features)
        outputs = self.probabilities() @ self._z_sign_matrix(wires)
        adjoints = []
        for wire in wires:
            adjoint = self._state.copy()
            self._split_1q(int(wire), adjoint)[:, :, 1, :] *= -1.0
            adjoints.append(adjoint)
        jacobian = self._adjoint_jacobian(circuit, adjoints, resolved, features,
                                          circuit.n_parameters)
        return outputs, jacobian

    def energy_jacobian(self, circuit: Circuit, terms: Sequence, weights=None, features=None):
        """``(energy, dE/dweights)`` shaped ``(batch,)`` and ``(batch, P)``."""
        resolved, features = self._forward(circuit, weights, features)
        energy = self.energies(terms)
        constant, diagonal, rest = self._grouped_terms(terms)
        adjoint = self._state * constant if constant else np.zeros_like(self._state)
        if diagonal is not None:
            adjoint += self._state * diagonal
        for coefficient, word in rest:
            adjoint += coefficient * self._apply_pauli_word(self._state.copy(), word)
        jacobian = self._adjoint_jacobian(circuit, [adjoint], resolved, features,
                                          circuit.n_parameters)
        return energy, jacobian[:, 0, :]

    def sample(self, shots: int = 1024) -> Dict[str, int]:
        probs = self.probabilities()[0]
        draws = self._rng.choice(len(probs), size=int(shots), p=probs / probs.sum())
        tally = np.bincount(draws, minlength=len(probs))
        width = self.n_qubits
        counts = {format(int(outcome), f"0{width}b"): int(count)
                  for outcome, count in enumerate(tally) if count}
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def __repr__(self) -> str:
        return (f"<StatevectorSimulator qubits={self.n_qubits} batch={self._batch} "
                f"shots={self.shots}>")
