"""Real Pauli observables and sums; characters follow q0, q1, ... order."""

from dataclasses import dataclass

from .errors import ValidationError
from .parameters import real


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


@dataclass(frozen=True)
class SparsePauliString:
    """A declared width and nonidentity sites, without width-sized strings."""

    num_qubits: int
    sites: tuple
    coefficient: float = 1.0

    def __post_init__(self):
        from collections.abc import Mapping

        from .ir import integer

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
