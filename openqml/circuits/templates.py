"""Standard feature maps and variational ansaetze.

Feature maps consume ``x(i)`` references, ansaetze consume ``w(i)`` references,
so any pair composes into a trainable model circuit.
"""

from __future__ import annotations

import math
from typing import Optional

from .circuit import Circuit, w, x

__all__ = [
    "angle_embedding",
    "amplitude_embedding",
    "zz_feature_map",
    "iqp_feature_map",
    "data_reuploading",
    "hardware_efficient_ansatz",
    "strongly_entangling_ansatz",
    "real_amplitudes_ansatz",
    "get_feature_map",
    "get_ansatz",
]


def angle_embedding(n_qubits: int, n_features: Optional[int] = None,
                    rotation: str = "y", scale: float = math.pi) -> Circuit:
    """One rotation per qubit; features beyond ``n_qubits`` wrap around."""
    n_features = n_features or n_qubits
    circuit = Circuit(n_qubits, name="angle_embedding")
    for q in range(n_qubits):
        circuit.h(q)
    for i in range(n_features):
        circuit.append(f"r{rotation}", i % n_qubits, x(i, scale=scale))
    return circuit


def amplitude_embedding(n_qubits: int) -> Circuit:
    """Load a 2**n dimensional (normalised) vector into the amplitudes."""
    return Circuit(n_qubits, name="amplitude_embedding").amplitude_embedding()


def zz_feature_map(n_qubits: int, n_features: Optional[int] = None, reps: int = 2,
                   scale: float = 2.0) -> Circuit:
    """The ZZ feature map of Havlicek et al. (2019), classically hard to sample."""
    n_features = n_features or n_qubits
    circuit = Circuit(n_qubits, name="zz_feature_map")
    for _ in range(reps):
        for q in range(n_qubits):
            circuit.h(q)
        for i in range(n_features):
            circuit.rz(x(i, scale=scale), i % n_qubits)
        for q in range(n_qubits - 1):
            a, b = q % n_qubits, (q + 1) % n_qubits
            circuit.cnot(a, b)
            circuit.rz(x(q % n_features, scale=scale), b)
            circuit.cnot(a, b)
    return circuit


def iqp_feature_map(n_qubits: int, n_features: Optional[int] = None, reps: int = 1) -> Circuit:
    """Instantaneous-quantum-polynomial style map: H layer, then diagonal phases."""
    n_features = n_features or n_qubits
    circuit = Circuit(n_qubits, name="iqp_feature_map")
    for _ in range(reps):
        for q in range(n_qubits):
            circuit.h(q)
        for i in range(n_features):
            circuit.phase(x(i, scale=math.pi), i % n_qubits)
        for q in range(n_qubits - 1):
            circuit.rzz(x(q % n_features, scale=math.pi / 2), q, q + 1)
    return circuit


def data_reuploading(n_qubits: int, n_features: int, layers: int = 2,
                     weight_offset: int = 0) -> Circuit:
    """Alternating encode/trainable blocks (Perez-Salinas et al., 2020).

    Re-uploading makes the model an explicit Fourier series in the inputs, which
    is why it does well on periodic targets and poorly on unstructured ones.
    """
    circuit = Circuit(n_qubits, name="data_reuploading")
    index = weight_offset
    for _ in range(layers):
        # one encoding gate per qubit per layer: the reachable Fourier spectrum
        # grows with the number of encoding gates, not with the parameter count
        # loop over features, not qubits: with more features than qubits the
        # other way round never reaches the trailing columns at all
        for f in range(n_features):
            circuit.ry(x(f, scale=math.pi), f % n_qubits)
        for q in range(n_qubits):
            circuit.rz(w(index), q); index += 1
            circuit.ry(w(index), q); index += 1
        for q in range(n_qubits - 1):
            circuit.cnot(q, q + 1)
    return circuit


def hardware_efficient_ansatz(n_qubits: int, layers: int = 2, weight_offset: int = 0,
                              entangler: str = "cnot") -> Circuit:
    """RY-RZ rotations plus a linear entangling chain, repeated ``layers`` times."""
    circuit = Circuit(n_qubits, name="hardware_efficient_ansatz")
    index = weight_offset
    for _ in range(layers):
        for q in range(n_qubits):
            circuit.ry(w(index), q); index += 1
            circuit.rz(w(index), q); index += 1
        for q in range(n_qubits - 1):
            circuit.append(entangler, (q, q + 1))
        if n_qubits > 2:
            circuit.append(entangler, (n_qubits - 1, 0))
    for q in range(n_qubits):
        circuit.ry(w(index), q); index += 1
    return circuit


def strongly_entangling_ansatz(n_qubits: int, layers: int = 2, weight_offset: int = 0) -> Circuit:
    """Three rotations per qubit and a range-shifted CNOT ring per layer."""
    circuit = Circuit(n_qubits, name="strongly_entangling_ansatz")
    index = weight_offset
    for layer in range(layers):
        for q in range(n_qubits):
            circuit.rz(w(index), q); index += 1
            circuit.ry(w(index), q); index += 1
            circuit.rz(w(index), q); index += 1
        step = (layer % max(1, n_qubits - 1)) + 1
        if n_qubits > 1:
            for q in range(n_qubits):
                circuit.cnot(q, (q + step) % n_qubits)
    return circuit


def real_amplitudes_ansatz(n_qubits: int, layers: int = 2, weight_offset: int = 0) -> Circuit:
    """RY-only ansatz: real amplitudes, half the parameters, no complex phases."""
    circuit = Circuit(n_qubits, name="real_amplitudes_ansatz")
    index = weight_offset
    for _ in range(layers):
        for q in range(n_qubits):
            circuit.ry(w(index), q); index += 1
        for q in range(n_qubits - 1):
            circuit.cnot(q, q + 1)
    for q in range(n_qubits):
        circuit.ry(w(index), q); index += 1
    return circuit


_FEATURE_MAPS = {
    "angle": angle_embedding,
    "amplitude": lambda n_qubits, n_features=None, **kw: amplitude_embedding(n_qubits),
    "zz": zz_feature_map,
    "iqp": iqp_feature_map,
}

_ANSAETZE = {
    "hardware_efficient": hardware_efficient_ansatz,
    "strongly_entangling": strongly_entangling_ansatz,
    "real_amplitudes": real_amplitudes_ansatz,
}


def get_feature_map(name: str, n_qubits: int, n_features: Optional[int] = None, **kwargs) -> Circuit:
    """Look up a feature map by name (``angle``, ``amplitude``, ``zz``, ``iqp``)."""
    if isinstance(name, Circuit):
        return name
    try:
        factory = _FEATURE_MAPS[name]
    except KeyError:
        raise ValueError(f"unknown feature map {name!r}; available: {sorted(_FEATURE_MAPS)}") from None
    return factory(n_qubits, n_features, **kwargs)


def get_ansatz(name: str, n_qubits: int, layers: int = 2, weight_offset: int = 0, **kwargs) -> Circuit:
    """Look up an ansatz by name."""
    if isinstance(name, Circuit):
        return name
    try:
        factory = _ANSAETZE[name]
    except KeyError:
        raise ValueError(f"unknown ansatz {name!r}; available: {sorted(_ANSAETZE)}") from None
    return factory(n_qubits, layers, weight_offset, **kwargs)
