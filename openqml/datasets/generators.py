"""Deterministic generators for the bundled datasets.

Nothing is downloaded: every bundled dataset is regenerated from a seed, so a
fresh install is reproducible bit-for-bit and works offline. Datasets you add
yourself are stored as arrays instead.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

__all__ = [
    "make_moons", "make_circles", "make_blobs", "make_parity", "make_periodic",
    "make_tfim_ground_states", "make_two_qubit_hamiltonians", "make_bars_and_stripes",
    "get_generator",
]


def make_moons(n_samples: int = 200, noise: float = 0.12, seed: int = 0):
    """Two interleaving half-circles; the canonical non-linear toy problem."""
    rng = np.random.default_rng(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    theta_out = np.linspace(0, math.pi, n_out)
    theta_in = np.linspace(0, math.pi, n_in)
    X = np.vstack([
        np.column_stack([np.cos(theta_out), np.sin(theta_out)]),
        np.column_stack([1 - np.cos(theta_in), 0.5 - np.sin(theta_in)]),
    ])
    X += rng.normal(0, noise, X.shape)
    y = np.hstack([np.zeros(n_out, dtype=int), np.ones(n_in, dtype=int)])
    order = rng.permutation(n_samples)
    return X[order], y[order], ["x0", "x1"]


def make_circles(n_samples: int = 200, noise: float = 0.06, factor: float = 0.5, seed: int = 1):
    """A small circle inside a large one: not linearly separable at all."""
    rng = np.random.default_rng(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    t_out = np.linspace(0, 2 * math.pi, n_out, endpoint=False)
    t_in = np.linspace(0, 2 * math.pi, n_in, endpoint=False)
    X = np.vstack([
        np.column_stack([np.cos(t_out), np.sin(t_out)]),
        factor * np.column_stack([np.cos(t_in), np.sin(t_in)]),
    ])
    X += rng.normal(0, noise, X.shape)
    y = np.hstack([np.zeros(n_out, dtype=int), np.ones(n_in, dtype=int)])
    order = rng.permutation(n_samples)
    return X[order], y[order], ["x0", "x1"]


def make_blobs(n_samples: int = 150, n_features: int = 4, centers: int = 3, spread: float = 0.6,
               seed: int = 2):
    """Isotropic Gaussian clusters -- a sanity check any model should pass."""
    rng = np.random.default_rng(seed)
    means = rng.uniform(-1.5, 1.5, size=(centers, n_features))
    y = rng.integers(0, centers, size=n_samples)
    X = means[y] + rng.normal(0, spread, size=(n_samples, n_features))
    return X, y.astype(int), [f"x{i}" for i in range(n_features)]


def make_parity(n_bits: int = 4):
    """Every ``n_bits`` bitstring labelled by its parity.

    Parity is the standard example of a target no local/low-degree model can
    shortcut: you either represent the global correlation or you do not.
    """
    values = np.arange(2 ** n_bits)
    X = ((values[:, None] >> np.arange(n_bits - 1, -1, -1)) & 1).astype(float)
    y = (X.sum(axis=1) % 2).astype(int)
    return X, y, [f"b{i}" for i in range(n_bits)]


def make_periodic(n_samples: int = 120, frequency: float = 3.0, noise: float = 0.02, seed: int = 3):
    """y = sin(frequency * x) on [-1, 1].

    High-frequency periodic targets are where a data re-uploading circuit has a
    structural edge -- it *is* a truncated Fourier series -- and where a plain
    ridge model on the raw feature has none.
    """
    rng = np.random.default_rng(seed)
    X = np.linspace(-1.0, 1.0, n_samples).reshape(-1, 1)
    y = np.sin(frequency * math.pi * X[:, 0]) + rng.normal(0, noise, n_samples)
    return X, y, ["x0"]


def _tfim_hamiltonian(n_qubits: int, h: float) -> np.ndarray:
    """Transverse-field Ising model, open chain: H = -sum ZZ - h sum X."""
    identity = np.eye(2)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]])

    def embed(op, wire):
        out = np.array([[1.0]])
        for q in range(n_qubits):
            out = np.kron(out, op if q == wire else identity)
        return out

    dimension = 2 ** n_qubits
    hamiltonian = np.zeros((dimension, dimension))
    for q in range(n_qubits - 1):
        hamiltonian -= embed(pauli_z, q) @ embed(pauli_z, q + 1)
    for q in range(n_qubits):
        hamiltonian -= h * embed(pauli_x, q)
    return hamiltonian


def make_tfim_ground_states(n_qubits: int = 6, n_samples: int = 40, gap: float = 0.2):
    """Exact ground states of the 1-D TFIM, labelled by phase.

    Labels come from the model's known critical point (h = 1 for the infinite
    chain); field strengths within ``gap`` of it are skipped because the finite
    chain has no sharp transition there.
    """
    fields = np.concatenate([
        np.linspace(0.1, 1.0 - gap, n_samples // 2),
        np.linspace(1.0 + gap, 2.5, n_samples - n_samples // 2),
    ])
    states = np.zeros((len(fields), 2 ** n_qubits), dtype=complex)
    for i, h in enumerate(fields):
        values, vectors = np.linalg.eigh(_tfim_hamiltonian(n_qubits, float(h)))
        states[i] = vectors[:, 0]
    labels = (fields > 1.0).astype(int)
    names = [f"amp{i}" for i in range(2 ** n_qubits)]
    return states, labels, names, {"transverse_field": fields.tolist(), "n_qubits": n_qubits}


def make_two_qubit_hamiltonians(n_points: int = 12, r_min: float = 0.4, r_max: float = 2.6):
    """A family of two-qubit Hamiltonians with an exactly known ground state.

    The coefficients are a smooth *surrogate* chosen to produce a bound-state
    energy curve with a minimum and a flat dissociation tail. They are not the
    output of an electronic-structure calculation and should not be read as
    chemistry; what makes the family useful is that the reference energies below
    are exact diagonalisations of the very same operators a VQE run optimises.
    """
    separations = np.linspace(r_min, r_max, n_points)
    records = []
    for r in separations:
        decay = math.exp(-1.4 * (r - 0.74))
        g0 = -1.05 + 0.62 * (1.0 - math.exp(-1.7 * (r - 0.74))) ** 2
        g1 = 0.18 * decay
        g2 = 0.18 * decay
        g3 = 0.13 * decay
        g4 = 0.20 * decay
        terms = [
            (g0, {}),
            (g1, {0: "Z"}),
            (g2, {1: "Z"}),
            (g3, {0: "Z", 1: "Z"}),
            (g4, {0: "X", 1: "X"}),
            (g4, {0: "Y", 1: "Y"}),
        ]
        records.append({"separation": float(r), "terms": terms})
    return records


def make_bars_and_stripes(size: int = 3, seed: int = 4):
    """All bar/stripe patterns of an ``size x size`` grid, flattened to bits."""
    patterns = []
    for value in range(2 ** size):
        bits = [(value >> i) & 1 for i in range(size)]
        patterns.append(np.repeat(np.array(bits, dtype=float)[:, None], size, axis=1).reshape(-1))
        patterns.append(np.repeat(np.array(bits, dtype=float)[None, :], size, axis=0).reshape(-1))
    X = np.unique(np.array(patterns), axis=0)
    y = (X.reshape(len(X), size, size)[:, 0, :].std(axis=1) == 0).astype(int)
    return X, y, [f"p{i}" for i in range(size * size)]


_GENERATORS = {
    "make_moons": make_moons,
    "make_circles": make_circles,
    "make_blobs": make_blobs,
    "make_parity": make_parity,
    "make_periodic": make_periodic,
    "make_tfim_ground_states": make_tfim_ground_states,
    "make_two_qubit_hamiltonians": make_two_qubit_hamiltonians,
    "make_bars_and_stripes": make_bars_and_stripes,
}


def get_generator(name: str):
    try:
        return _GENERATORS[name]
    except KeyError:
        raise ValueError(f"unknown generator {name!r}") from None
