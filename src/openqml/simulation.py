"""Exact dense/sparse pure states and small-register density matrices.

q0 is the least significant bit. Matrix wire lists are ordered least significant
local bit first; printed count strings use the conventional q[n-1]...q0 order.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from ._validation import integer, pauli_string, probability, wires
from .exceptions import ResourceLimitError


@dataclass(frozen=True)
class SimulationBudget:
    """Conservative allocation and arithmetic limits, checked before dense reads.

    Sparse simulation can reach 40 qubits only when support and work stay small.
    This is a capacity limit, not a claim that arbitrary 40-qubit states fit.
    """

    max_qubits: int = 40
    max_dense_qubits: int = 18
    max_sparse_states: int = 262_144
    max_density_qubits: int = 8
    max_memory_mb: int = 256
    max_work: int = 20_000_000
    max_operations: int = 10_000

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            integer(getattr(self, name), name, 1)
        if self.max_qubits > 40 or self.max_density_qubits > 10:
            raise ValueError("max_qubits is capped at 40; max_density_qubits is capped at 10")
        if self.max_dense_qubits > self.max_qubits:
            raise ValueError("max_dense_qubits must not exceed max_qubits")

    def allocation(self, elements, *, complex_values=True, copies=1):
        size = elements * (16 if complex_values else 8) * copies
        if size > self.max_memory_mb * 1024**2:
            raise ResourceLimitError(
                f"Readout/allocation needs about {size / 1024**2:.1f} MiB; "
                f"budget is {self.max_memory_mb} MiB. Request fewer wires."
            )


H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)


def validate_matrix(matrix, size, *, unitary=True):
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError(f"matrix must be a finite {size} by {size} array")
    if unitary and not np.allclose(matrix.conj().T @ matrix, np.eye(size), atol=1e-10, rtol=1e-10):
        raise ValueError("matrix must be unitary (U†U = I)")
    result = matrix.copy()
    result.flags.writeable = False
    return result


def validate_channel(kraus, size):
    operators = tuple(validate_matrix(k, size, unitary=False) for k in kraus)
    if not operators or not np.allclose(
        sum((k.conj().T @ k for k in operators), np.zeros((size, size), complex)),
        np.eye(size),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError("Kraus operators must satisfy sum(K†K) = I")
    return operators


def _indices(n_qubits, targets, controls=()):
    mask = sum(1 << q for q in targets)
    control_mask = sum(1 << q for q in controls)
    base = np.arange(1 << n_qubits)
    base = base[((base & mask) == 0) & ((base & control_mask) == control_mask)]
    local = np.array(
        [sum(((j >> i) & 1) << q for i, q in enumerate(targets)) for j in range(1 << len(targets))]
    )
    return local[:, None] | base[None, :]


def _apply_dense(array, matrix, targets, controls=()):
    n_qubits = array.shape[0].bit_length() - 1
    indices = _indices(n_qubits, targets, controls)
    result = array.copy()
    result[indices] = np.einsum("ij,j...->i...", matrix, array[indices])
    return result


def _pauli_action(index, pauli):
    target, phase = index, 1.0 + 0j
    for q, p in enumerate(reversed(pauli)):
        bit = (index >> q) & 1
        if p in "XY":
            target ^= 1 << q
        if p == "Y":
            phase *= -1j if bit else 1j
        elif p == "Z" and bit:
            phase *= -1
    return target, phase


class State:
    """A mutable quantum state. Circuit.run() always returns a fresh instance."""

    def __init__(self, n_qubits, *, backend="auto", seed=None, budget=None):
        self.budget = budget or SimulationBudget()
        if not isinstance(self.budget, SimulationBudget):
            raise ValueError("budget must be a SimulationBudget")
        self.n_qubits = integer(n_qubits, "n_qubits", 1, self.budget.max_qubits)
        if backend not in ("auto", "dense", "sparse"):
            raise ValueError("backend must be 'auto', 'dense', or 'sparse'")
        if backend == "auto":
            backend = "dense" if n_qubits <= self.budget.max_dense_qubits else "sparse"
        if backend == "dense":
            self._check_vector(n_qubits)
            self._dense = np.zeros(1 << n_qubits, complex)
            self._dense[0] = 1
            self._sparse = None
        else:
            self._dense = None
            self._sparse = {0: 1.0 + 0j}
        self._rho = None
        self._rng = np.random.default_rng(seed)
        self._work = 0
        self.classical = {}

    @classmethod
    def from_vector(cls, amplitudes, *, seed=None, budget=None):
        values = np.asarray(amplitudes, dtype=complex)
        if values.ndim != 1 or values.size < 2 or values.size & (values.size - 1):
            raise ValueError("amplitudes must have a power-of-two length of at least 2")
        if not np.isfinite(values).all() or not np.isclose(
            np.vdot(values, values), 1, atol=1e-10, rtol=1e-10
        ):
            raise ValueError("amplitudes must be finite and normalized")
        state = cls(values.size.bit_length() - 1, backend="dense", seed=seed, budget=budget)
        state._dense = values.copy()
        return state

    @property
    def backend(self):
        return (
            "density"
            if self._rho is not None
            else ("dense" if self._dense is not None else "sparse")
        )

    @property
    def work(self):
        """Estimated scalar gate work consumed by this state."""
        return self._work

    def clone(self):
        return deepcopy(self)

    def _check_vector(self, count):
        if count > self.budget.max_dense_qubits:
            raise ResourceLimitError(
                f"Complete vectors/probability arrays are limited to "
                f"{self.budget.max_dense_qubits} wires; request a marginal readout."
            )
        self.budget.allocation(1 << count, copies=6)

    def _consume(self, work):
        if self._work + work > self.budget.max_work:
            raise ResourceLimitError("Simulation work budget exceeded; use a smaller circuit")
        self._work += work

    def _density(self):
        if self._rho is None:
            if self.n_qubits > self.budget.max_density_qubits:
                raise ResourceLimitError(
                    f"Exact noise/reset needs a density matrix; limited to "
                    f"{self.budget.max_density_qubits} qubits."
                )
            self.budget.allocation(1 << (2 * self.n_qubits), copies=6)
            amplitudes = self.statevector()
            self._rho = np.outer(amplitudes, amplitudes.conj())
            self._dense = self._sparse = None
        return self._rho

    def _apply(self, matrix, targets, controls=()):
        if self._rho is not None:
            self._consume(self._rho.size * (1 << len(targets)) * 2)
            left = _apply_dense(self._rho, matrix, targets, controls)
            self._rho = _apply_dense(left.T, matrix.conj(), targets, controls).T
        elif self._dense is not None:
            self._consume(self._dense.size * (1 << len(targets)))
            self._dense = _apply_dense(self._dense, matrix, targets, controls)
        else:
            self._consume(len(self._sparse) * (1 << len(targets)))
            target_mask = sum(1 << q for q in targets)
            control_mask = sum(1 << q for q in controls)
            result = {}
            support_limit = min(
                self.budget.max_sparse_states, self.budget.max_memory_mb * 1024**2 // 384
            )
            for index, amplitude in self._sparse.items():
                if index & control_mask != control_mask:
                    result[index] = result.get(index, 0j) + amplitude
                    if len(result) > support_limit:
                        raise ResourceLimitError("Sparse support/memory budget exceeded")
                    continue
                local = sum(((index >> q) & 1) << j for j, q in enumerate(targets))
                base = index & ~target_mask
                for row in np.flatnonzero(matrix[:, local]):
                    output = base | sum(((int(row) >> j) & 1) << q for j, q in enumerate(targets))
                    result[output] = result.get(output, 0j) + matrix[row, local] * amplitude
                    if len(result) > support_limit:
                        raise ResourceLimitError(
                            "Sparse support budget exceeded; fewer qubits or gates are needed. "
                            "No amplitudes were truncated."
                        )
            self._sparse = {k: v for k, v in result.items() if v != 0}
        return self

    def unitary(self, matrix, targets, *, controls=()):
        targets = wires(targets, self.n_qubits, allow_empty=True)
        controls = wires(controls, self.n_qubits, allow_empty=True)
        if len(targets) > 4 or set(targets) & set(controls):
            raise ValueError("custom gates support at most 4 targets, disjoint from controls")
        return self._apply(validate_matrix(matrix, 1 << len(targets)), targets, controls)

    def h(self, q):
        return self.unitary(H, [q])

    def x(self, q):
        return self.unitary(X, [q])

    def y(self, q):
        return self.unitary(Y, [q])

    def z(self, q):
        return self.unitary(Z, [q])

    def rx(self, theta, q):
        from .circuit import rotation

        return self.unitary(rotation("x", theta), [q])

    def ry(self, theta, q):
        from .circuit import rotation

        return self.unitary(rotation("y", theta), [q])

    def rz(self, theta, q):
        from .circuit import rotation

        return self.unitary(rotation("z", theta), [q])

    def cx(self, control, target):
        return self.unitary(X, [target], controls=[control])

    def cz(self, control, target):
        return self.unitary(Z, [target], controls=[control])

    def statevector(self):
        if self._rho is not None:
            raise ValueError("a mixed state has no unique statevector; use reduced_density()")
        self._check_vector(self.n_qubits)
        if self._dense is not None:
            return self._dense.copy()
        result = np.zeros(1 << self.n_qubits, complex)
        for index, amplitude in self._sparse.items():
            result[index] = amplitude
        return result

    def sparse_amplitudes(self):
        """Return a copy of occupied amplitudes without a dense allocation."""
        if self._rho is not None:
            raise ValueError("mixed states have no unique amplitudes")
        if self._sparse is not None:
            return self._sparse.copy()
        if np.count_nonzero(self._dense) * 192 > self.budget.max_memory_mb * 1024**2:
            raise ResourceLimitError("Sparse amplitude readout exceeds the memory budget")
        return {int(i): complex(self._dense[i]) for i in np.flatnonzero(self._dense)}

    def probabilities(self, selected_wires=None):
        selected = (
            tuple(range(self.n_qubits))
            if selected_wires is None
            else wires(selected_wires, self.n_qubits)
        )
        self._check_vector(len(selected))
        result = np.zeros(1 << len(selected))
        if self._rho is not None:
            values = np.real(np.diag(self._rho))
        elif self._dense is not None:
            values = np.abs(self._dense) ** 2
        else:
            for index, amplitude in self._sparse.items():
                local = sum(((index >> q) & 1) << j for j, q in enumerate(selected))
                result[local] += abs(amplitude) ** 2
            return result
        indices = np.arange(values.size)
        local = sum(((indices >> q) & 1) << j for j, q in enumerate(selected))
        np.add.at(result, local, values)
        return np.maximum(result, 0)

    def sample(self, shots=1024, *, seed=None, selected_wires=None):
        """Born-rule counts; does not collapse the state or create a full 2**n array."""
        shots = integer(shots, "shots", 1)
        self.budget.allocation(shots, complex_values=False, copies=4)
        selected = (
            tuple(range(self.n_qubits))
            if selected_wires is None
            else wires(selected_wires, self.n_qubits)
        )
        rng = self._rng if seed is None else np.random.default_rng(seed)
        if self._sparse is not None:
            labels = np.array(list(self._sparse), dtype=np.int64)
            weights = np.array([abs(a) ** 2 for a in self._sparse.values()])
            sampled = rng.choice(labels, size=shots, p=weights / weights.sum())
            local = sum(((sampled >> q) & 1) << j for j, q in enumerate(selected))
        else:
            values = self.probabilities(selected)
            local = rng.choice(values.size, size=shots, p=values / values.sum())
        labels, counts = np.unique(local, return_counts=True)
        return {format(int(k), f"0{len(selected)}b"): int(v) for k, v in zip(labels, counts)}

    def expectation(self, observable):
        """Exact Pauli or Hamiltonian expectation; strings put q0 at the right."""
        from .hamiltonian import Hamiltonian

        if isinstance(observable, Hamiltonian):
            if observable.n_qubits != self.n_qubits:
                raise ValueError("Hamiltonian width does not match the state")
            return float(sum(c * self.expectation(p) for p, c in observable.terms))
        pauli = pauli_string(observable, self.n_qubits)
        if self._sparse is not None:
            self._consume(len(self._sparse) * self.n_qubits)
            value = 0j
            for index, amplitude in self._sparse.items():
                output, phase = _pauli_action(index, pauli)
                value += np.conj(self._sparse.get(output, 0j)) * phase * amplitude
        else:
            size = 1 << self.n_qubits
            self._consume(size * self.n_qubits)
            indices = np.arange(size)
            outputs = indices.copy()
            phases = np.ones(size, complex)
            for q, p in enumerate(reversed(pauli)):
                bits = (indices >> q) & 1
                if p in "XY":
                    outputs ^= 1 << q
                if p == "Y":
                    phases *= 1j * (1 - 2 * bits)
                elif p == "Z":
                    phases *= 1 - 2 * bits
            value = (
                np.sum(phases * self._rho[indices, outputs])
                if self._rho is not None
                else (np.vdot(self._dense[outputs], phases * self._dense))
            )
        return float(np.real(value))

    def reduced_density(self, selected_wires=None):
        selected = (
            tuple(range(self.n_qubits))
            if selected_wires is None
            else wires(selected_wires, self.n_qubits)
        )
        if len(selected) > self.budget.max_density_qubits:
            raise ResourceLimitError("Density readout exceeds max_density_qubits")
        size = 1 << len(selected)
        self.budget.allocation(size * size, copies=3)
        self._consume(
            size * size * (len(self._sparse) if self._sparse is not None else (1 << self.n_qubits))
        )
        rest = tuple(q for q in range(self.n_qubits) if q not in selected)
        if self._sparse is not None:
            groups = {}
            for index, amplitude in self._sparse.items():
                local = sum(((index >> q) & 1) << j for j, q in enumerate(selected))
                env = sum(((index >> q) & 1) << j for j, q in enumerate(rest))
                groups.setdefault(env, {})[local] = amplitude
            result = np.zeros((size, size), complex)
            for group in groups.values():
                labels = list(group)
                values = np.array(list(group.values()))
                result[np.ix_(labels, labels)] += np.outer(values, values.conj())
            return result
        axes = [self.n_qubits - 1 - q for q in reversed(selected + rest)]
        if self._rho is None:
            # Rest bits first in tensor axes: transpose to a local-by-environment matrix.
            data = self._dense.reshape((2,) * self.n_qubits).transpose(axes)
            data = data.reshape(1 << len(rest), size).T
            return data @ data.conj().T
        result = np.zeros((size, size), complex)
        for env in range(1 << len(rest)):
            base = sum(((env >> j) & 1) << q for j, q in enumerate(rest))
            labels = [
                base | sum(((k >> j) & 1) << q for j, q in enumerate(selected)) for k in range(size)
            ]
            result += self._rho[np.ix_(labels, labels)]
        return result

    def bloch_vector(self, q):
        rho = self.reduced_density([q])
        return np.array([np.trace(rho @ p).real for p in (X, Y, Z)])

    def entropy(self, selected_wires=None):
        values = np.linalg.eigvalsh(self.reduced_density(selected_wires))
        values = values[values > 1e-15]
        return float(-np.sum(values * np.log2(values)))

    def measure(self, q, basis="Z"):
        q = wires([q], self.n_qubits)[0]
        if basis not in ("X", "Y", "Z"):
            raise ValueError("measurement basis must be X, Y, or Z")
        change = H if basis == "X" else (np.diag([1, 1j]) @ H if basis == "Y" else None)
        if change is not None:
            self._apply(change.conj().T, (q,))
        probs = self.probabilities([q])
        bit = int(self._rng.random() < probs[1] / probs.sum())
        norm = probs[bit]
        if self._sparse is not None:
            self._sparse = {
                k: a / np.sqrt(norm) for k, a in self._sparse.items() if (k >> q) & 1 == bit
            }
        else:
            mask = ((np.arange(1 << self.n_qubits) >> q) & 1) == bit
            if self._rho is not None:
                self._rho *= mask[:, None] * mask[None, :]
                self._rho /= norm
            else:
                self._dense *= mask
                self._dense /= np.sqrt(norm)
        if change is not None:
            self._apply(change, (q,))
        return bit

    def channel(self, kraus, targets):
        targets = wires(targets, self.n_qubits)
        if len(targets) > 4:
            raise ValueError("channels support at most 4 target wires")
        operators = validate_channel(kraus, 1 << len(targets))
        self._consume((1 << (2 * self.n_qubits)) * len(operators) * (1 << len(targets)) * 2)
        rho = self._density()
        result = np.zeros_like(rho)
        for k in operators:
            left = _apply_dense(rho, k, targets)
            result += _apply_dense(left.T, k.conj(), targets).T
        self._rho = result
        return self

    def amplitude_damping(self, p, q):
        p = probability(p)
        return self.channel([[[1, 0], [0, np.sqrt(1 - p)]], [[0, np.sqrt(p)], [0, 0]]], [q])

    def dephasing(self, p, q):
        """Phase-flip channel (1-p)rho + p Z rho Z; p is the flip probability."""
        p = probability(p)
        return self.channel([np.sqrt(1 - p) * np.eye(2), np.sqrt(p) * Z], [q])

    def depolarizing(self, p, q):
        """With probability p replace the selected qubit by I/2."""
        p = probability(p)
        return self.channel(
            [np.sqrt(1 - 3 * p / 4) * np.eye(2)] + [np.sqrt(p / 4) * gate for gate in (X, Y, Z)],
            [q],
        )

    def reset(self, q):
        """Exact unconditional reset channel; other qubits may become mixed."""
        return self.channel([[[1, 0], [0, 0]], [[0, 1], [0, 0]]], [q])
