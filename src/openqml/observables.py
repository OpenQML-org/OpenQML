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


def z_observables(num_qubits: int) -> tuple[PauliString, ...]:
    return tuple(PauliString("I" * q + "Z" + "I" * (num_qubits - q - 1)) for q in range(num_qubits))
