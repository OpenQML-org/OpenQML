"""Execution backends and the batched helpers the models are built on.

``default.statevector`` is the pure-NumPy simulator that ships with the
package. ``pennylane.default.qubit`` is used when PennyLane is installed.
Backends that cannot batch still work: the helpers below fall back to a loop.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..circuits.circuit import Circuit
from ..exceptions import BackendNotAvailableError
from .statevector import StatevectorSimulator

__all__ = [
    "StatevectorSimulator", "get_backend", "list_backends", "register_backend",
    "simulate", "expectations", "z_expectations", "statevectors",
    "batched_states", "batched_z", "batched_energies", "chunk_limit",
    "batched_z_jacobian", "batched_energy_jacobian", "supports_adjoint", "prefers_adjoint",
    "ADJOINT_MIN_WORK",
    "DEFAULT_BACKEND", "MAX_BATCH_ELEMENTS", "MAX_BATCH_AMPLITUDES",
]

DEFAULT_BACKEND = "default.statevector"

#: Cap on how many statevectors are held in memory at once. Batches larger than
#: this are split, so callers can hand over a whole gradient without thinking
#: about the qubit count.
MAX_BATCH_ELEMENTS = 4096

#: ...and a cap on the total amplitudes in flight: a chunk holds
#: ``batch * 2**n_qubits`` complex numbers, and a gate is memory-bound, so the
#: throughput optimum is the chunk that keeps that working set inside L2 rather
#: than the largest one that fits in RAM. Measured across 4 to 10 qubits the
#: best chunk sits at roughly 2**16 amplitudes (1 MB) either way.
MAX_BATCH_AMPLITUDES = 1 << 16

_BACKENDS: Dict[str, Callable] = {
    DEFAULT_BACKEND: lambda n_qubits, **kw: StatevectorSimulator(n_qubits, **kw),
}


def register_backend(name: str, factory: Callable) -> None:
    """Register ``factory(n_qubits, seed=..., shots=...) -> simulator``."""
    _BACKENDS[name] = factory


def list_backends() -> List[str]:
    names = list(_BACKENDS)
    try:  # pragma: no cover - depends on the environment
        import pennylane  # noqa: F401

        names.append("pennylane.default.qubit")
    except ImportError:
        pass
    return sorted(set(names))


def get_backend(name: str = DEFAULT_BACKEND, n_qubits: int = 1, **kwargs):
    """Instantiate a backend by name."""
    if name.startswith("pennylane."):  # pragma: no cover - optional dependency
        try:
            import pennylane  # noqa: F401
        except ImportError:
            raise BackendNotAvailableError(
                "PennyLane is not installed; pip install openqml[pennylane]"
            ) from None
        from .pennylane_backend import PennyLaneSimulator

        return PennyLaneSimulator(n_qubits, device=name.split("pennylane.", 1)[1], **kwargs)
    try:
        factory = _BACKENDS[name]
    except KeyError:
        raise BackendNotAvailableError(
            f"unknown backend {name!r}; available: {list_backends()}"
        ) from None
    return factory(n_qubits, **kwargs)


# -- single-run helpers -------------------------------------------------------

def simulate(circuit: Circuit, weights=None, features=None, backend: str = DEFAULT_BACKEND,
             seed=None, shots: Optional[int] = None):
    """Run a circuit once and return the simulator holding the final state."""
    device = get_backend(backend, circuit.n_qubits, seed=seed, shots=shots)
    return device.run(circuit, weights, features)


def expectations(circuit: Circuit, observables: Sequence[Mapping[int, str]], weights=None,
                 features=None, backend: str = DEFAULT_BACKEND, seed=None,
                 shots: Optional[int] = None) -> np.ndarray:
    """Expectation values of several Pauli words after one circuit execution."""
    device = simulate(circuit, weights, features, backend, seed, shots)
    return np.array([device.expval(obs) for obs in observables], dtype=float)


def z_expectations(circuit: Circuit, wires: Sequence[int], weights=None, features=None,
                   backend: str = DEFAULT_BACKEND, seed=None,
                   shots: Optional[int] = None) -> np.ndarray:
    """<Z> on the given wires -- the usual read-out for a variational classifier."""
    return expectations(circuit, [{int(q): "Z"} for q in wires], weights, features,
                        backend, seed, shots)


# -- batched helpers ----------------------------------------------------------

def _batch_size(*arrays) -> int:
    sizes = [np.shape(a)[0] for a in arrays if a is not None and np.ndim(a) == 2]
    return max(sizes) if sizes else 1


def _chunk_limit_for(chunk, n_qubits: int) -> int:
    """Resolve an explicit ``chunk`` argument, rejecting the useless values."""
    if chunk is None:
        return chunk_limit(n_qubits)
    if int(chunk) < 1:
        raise ValueError(f"chunk must be at least 1, got {chunk}")
    return int(chunk)


def _slice(array, start: int, stop: int):
    """Take one chunk's rows, leaving a shared array alone.

    ``StatevectorSimulator._prepare`` treats both a 1-D array and a 2-D array
    with a single row as "shared by the whole batch". Slicing the second form
    like a batch hands every chunk after the first an empty array.
    """
    if array is None or np.ndim(array) == 1 or np.shape(array)[0] == 1:
        return array
    return array[start:stop]


def _resolve_device(device, circuit, backend, seed, shots):
    return device if device is not None else get_backend(backend, circuit.n_qubits,
                                                         seed=seed, shots=shots)


def chunk_limit(n_qubits: int) -> int:
    """How many circuits one batched call may hold for a register this wide."""
    return max(1, min(MAX_BATCH_ELEMENTS, MAX_BATCH_AMPLITUDES >> int(n_qubits)))


def _chunked(circuit, weights, features, device, collect, chunk: Optional[int]):
    """Run a batch in memory-safe pieces, falling back to a loop if needed."""
    total = _batch_size(weights, features)
    if not getattr(device, "supports_batch", False):  # pragma: no cover - optional backends
        return np.concatenate([
            collect(device.run(circuit,
                               _slice(weights, i, i + 1) if np.ndim(weights) == 2 else weights,
                               _slice(features, i, i + 1) if np.ndim(features) == 2 else features))
            for i in range(total)
        ], axis=0)
    limit = _chunk_limit_for(chunk, circuit.n_qubits)
    if total <= limit:
        return collect(device.run_batch(circuit, weights, features))
    pieces = []
    for start in range(0, total, limit):
        stop = min(start + limit, total)
        pieces.append(collect(device.run_batch(circuit, _slice(weights, start, stop),
                                               _slice(features, start, stop))))
    return np.concatenate(pieces, axis=0)


def batched_states(circuit: Circuit, features=None, weights=None, device=None,
                   backend: str = DEFAULT_BACKEND, seed=None,
                   chunk: Optional[int] = None) -> np.ndarray:
    """``(batch, 2**n)`` statevectors for a batch of feature/weight rows."""
    device = _resolve_device(device, circuit, backend, seed, None)
    collect = lambda sim: (sim.states if getattr(sim, "supports_batch", False)
                           else sim.state.reshape(1, -1))
    return _chunked(circuit, weights, features, device, collect, chunk)


def batched_z(circuit: Circuit, wires: Sequence[int], weights=None, features=None, device=None,
              backend: str = DEFAULT_BACKEND, seed=None, shots: Optional[int] = None,
              chunk: Optional[int] = None) -> np.ndarray:
    """``(batch, len(wires))`` values of <Z>, for a batch of weights and/or features."""
    device = _resolve_device(device, circuit, backend, seed, shots)
    wires = [int(q) for q in wires]
    if getattr(device, "supports_batch", False):
        collect = lambda sim: sim.z_expvals(wires)
    else:  # pragma: no cover - optional backends
        collect = lambda sim: np.array([[sim.expval({q: "Z"}) for q in wires]])
    return _chunked(circuit, weights, features, device, collect, chunk)


def batched_energies(circuit: Circuit, terms: Sequence, weights=None, features=None, device=None,
                     backend: str = DEFAULT_BACKEND, seed=None, shots: Optional[int] = None,
                     chunk: Optional[int] = None) -> np.ndarray:
    """``(batch,)`` energies of one Hamiltonian over a batch of parameter rows."""
    device = _resolve_device(device, circuit, backend, seed, shots)
    if getattr(device, "supports_batch", False):
        collect = lambda sim: sim.energies(terms)
    else:  # pragma: no cover - optional backends
        collect = lambda sim: np.array([sim.expval_hamiltonian(terms)])
    return _chunked(circuit, weights, features, device, collect, chunk)


# -- analytic gradients -------------------------------------------------------

def supports_adjoint(device) -> bool:
    """True when this device can differentiate a circuit by the adjoint method."""
    return getattr(device, "supports_batch", False) and hasattr(device, "z_jacobian")


#: Where the backward sweep starts paying off, in ``amplitudes * parameters``.
#: Calibrated by measuring both rules from 2 to 12 qubits and 6 to 60
#: parameters; the crossover sits near 2e3 and the curve is flat around it, so
#: a wrong call near the boundary costs a few percent either way.
ADJOINT_MIN_WORK = 2_000


def prefers_adjoint(batch: int, n_qubits: int, n_parameters: int,
                    n_observables: int = 1) -> bool:
    """Is the backward sweep actually the cheaper gradient at this size?

    Not always, even though it does asymptotically less work. The shift rule
    stacks all ``2P+1`` parameter sets into *one* batched run, so it issues
    O(gates) NumPy calls on a large array; the sweep issues O(P) calls per
    observable on a small one. Below a few thousand amplitude-parameters the
    per-call overhead is the whole cost and the shift rule wins -- a 6-qubit
    VQE is on that side of the line, a 10-qubit one is 8x the other side.
    """
    work = batch * (1 << int(n_qubits)) * max(int(n_parameters), 1)
    return work >= ADJOINT_MIN_WORK * (1 + max(int(n_observables), 1)) / 2


def _chunked_pair(circuit, weights, features, device, call, chunk: Optional[int]):
    """Same chunking as :func:`_chunked`, for a call returning two arrays."""
    total = _batch_size(weights, features)
    limit = _chunk_limit_for(chunk, circuit.n_qubits)
    if total <= limit:
        return call(weights, features)
    values, jacobians = [], []
    for start in range(0, total, limit):
        stop = min(start + limit, total)
        value, jacobian = call(_slice(weights, start, stop), _slice(features, start, stop))
        values.append(value)
        jacobians.append(jacobian)
    return np.concatenate(values, axis=0), np.concatenate(jacobians, axis=0)


def batched_z_jacobian(circuit: Circuit, wires: Sequence[int], weights=None, features=None,
                       device=None, backend: str = DEFAULT_BACKEND, seed=None,
                       chunk: Optional[int] = None):
    """``(batch, W)`` values of <Z> and their ``(batch, W, P)`` derivatives.

    Exact and analytic, by one backward sweep per batch rather than the ``2P+1``
    forward runs the parameter-shift rule needs. Requires a device that
    implements ``z_jacobian``; ask :func:`supports_adjoint` first.
    """
    device = _resolve_device(device, circuit, backend, seed, None)
    wires = [int(q) for q in wires]
    return _chunked_pair(circuit, weights, features, device,
                         lambda w, f: device.z_jacobian(circuit, wires, w, f), chunk)


def batched_energy_jacobian(circuit: Circuit, terms: Sequence, weights=None, features=None,
                            device=None, backend: str = DEFAULT_BACKEND, seed=None,
                            chunk: Optional[int] = None):
    """``(batch,)`` energies and their ``(batch, P)`` derivatives, analytically."""
    device = _resolve_device(device, circuit, backend, seed, None)
    return _chunked_pair(circuit, weights, features, device,
                         lambda w, f: device.energy_jacobian(circuit, terms, w, f), chunk)


def statevectors(circuit: Circuit, X, weights=None, backend: str = DEFAULT_BACKEND,
                 seed=None) -> np.ndarray:
    """Encode every row of ``X`` and stack the resulting statevectors."""
    return batched_states(circuit, np.asarray(X), weights, backend=backend, seed=seed)
