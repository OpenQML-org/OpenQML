"""Real Pauli observables and sums; characters follow q0, q1, ... order."""

from dataclasses import dataclass

from ..errors import ValidationError
from .parameters import real


def _masks(paulis):
    """Encode X and Z supports with the first character at bit zero."""
    return (
        sum(1 << q for q, p in enumerate(paulis) if p in "XY"),
        sum(1 << q for q, p in enumerate(paulis) if p in "YZ"),
    )


@dataclass(frozen=True)
class PauliString:
    paulis: str
    coefficient: float = 1.0

    def __post_init__(self):
        if not isinstance(self.paulis, str) or not self.paulis or set(self.paulis) - set("IXYZ"):
            raise ValidationError("A Pauli string is a nonempty string over I, X, Y, Z.")
        object.__setattr__(self, "coefficient", real(self.coefficient))


@dataclass(frozen=True)
class Hamiltonian:
    terms: tuple[PauliString, ...]

    def __post_init__(self):
        object.__setattr__(self, "terms", tuple(self.terms))
        if (
            not self.terms
            or any(not isinstance(p, PauliString) for p in self.terms)
            or len({len(p.paulis) for p in self.terms}) != 1
        ):
            raise ValidationError("A Hamiltonian needs Pauli terms of the same width.")


def _basis_projector(width, wire, bit):
    identity = "I" * width
    z = identity[:wire] + "Z" + identity[wire + 1 :]
    return Hamiltonian((PauliString(identity, 0.5), PauliString(z, -0.5 if bit else 0.5)))


@dataclass(frozen=True)
class SparsePauliString:
    """A declared width and nonidentity sites, without width-sized strings."""

    num_qubits: int
    sites: tuple
    coefficient: float = 1.0

    def __post_init__(self):
        from collections.abc import Mapping

        from .core import integer

        width = integer(self.num_qubits, "num_qubits", 1)
        raw = self.sites.items() if isinstance(self.sites, Mapping) else self.sites
        try:
            sites = tuple((integer(q, "observable qubit"), p) for q, p in raw)
        except (TypeError, ValueError) as exc:
            raise ValidationError("Sparse Pauli sites require (qubit, Pauli) pairs.") from exc
        if len({q for q, _ in sites}) != len(sites) or any(
            q >= width or p not in ("I", "X", "Y", "Z") for q, p in sites
        ):
            raise ValidationError("Sparse Pauli sites need unique valid wires and I/X/Y/Z labels.")
        object.__setattr__(self, "num_qubits", width)
        object.__setattr__(self, "sites", tuple(sorted((q, p) for q, p in sites if p != "I")))
        object.__setattr__(self, "coefficient", real(self.coefficient))


@dataclass(frozen=True)
class SparseHamiltonian:
    terms: tuple[SparsePauliString, ...]

    def __post_init__(self):
        terms = tuple(self.terms)
        if (
            not terms
            or any(not isinstance(p, SparsePauliString) for p in terms)
            or len({p.num_qubits for p in terms}) != 1
        ):
            raise ValidationError(
                "A sparse Hamiltonian needs sparse Pauli terms of the same width."
            )
        object.__setattr__(self, "terms", terms)

    @property
    def num_qubits(self):
        return self.terms[0].num_qubits


def z_observables(num_qubits: int) -> tuple[PauliString, ...]:
    return tuple(PauliString("I" * q + "Z" + "I" * (num_qubits - q - 1)) for q in range(num_qubits))


def _linear_observable(observables, features, offset=0.0):
    """Combine weighted output terms before allocating simulation or error to them."""
    import math

    first = observables[0]
    width = len((first.terms[0] if isinstance(first, Hamiltonian) else first).paulis)
    identity = "I" * width
    summands = {identity: [offset]}
    for index, coefficient in features:
        if coefficient:
            ob = observables[index]
            for term in ob.terms if isinstance(ob, Hamiltonian) else (ob,):
                summands.setdefault(term.paulis, []).append(coefficient * term.coefficient)
    try:
        terms = tuple(
            PauliString(word, s) for word, values in summands.items() if (s := math.fsum(values))
        )
    except (OverflowError, ValueError) as exc:
        raise ValidationError("Combined readout coefficients exceed finite precision.") from exc
    return Hamiltonian(terms or (PauliString(identity, 0.0),))
