"""Lossless executable transport for formats with a narrower control model."""

import json
from dataclasses import dataclass

from ..errors import ValidationError
from ..ir._serialization import load_json
from ..ir.core import Circuit, _require_circuit, integer

_TARGETS = frozenset({"cirq", "pennylane", "openqasm2", "openqasm3", "qiskit"})


@dataclass(frozen=True)
class PortableProgram:
    """An OpenQML runtime program with an explicit target SDK or dialect."""

    circuit: Circuit
    target_format: str

    def __post_init__(self):
        _require_circuit(self.circuit)
        if not isinstance(self.target_format, str) or self.target_format not in _TARGETS:
            raise ValidationError("A portable program needs a native target format.")

    def to_json(self, *, max_bytes=2_000_000):
        max_bytes = integer(max_bytes, "max_bytes", 1)
        header = (
            json.dumps(
                {"schema": "openqml.portable_program", "version": 1, "target": self.target_format}
            )[:-1]
            + ', "circuit": '
        )
        remaining = max_bytes - len(header) - 1
        if remaining < 1:
            raise ValidationError("Portable program exceeds max_bytes.")
        return header + self.circuit.to_json(indent=None, max_bytes=remaining) + "}"

    @classmethod
    def from_json(cls, text, *, max_bytes=2_000_000, **limits):
        payload = load_json(text, max_bytes, label="Portable program")
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema", "version", "target", "circuit"}
            or payload["schema"] != "openqml.portable_program"
            or type(payload["version"]) is not int
            or payload["version"] != 1
        ):
            raise ValidationError("Invalid portable program schema or version.")
        return cls(Circuit.from_dict(payload["circuit"], **limits), payload["target"])

    def execute(self, bindings=None, **options):
        from ..sim.dense import execute

        return execute(self.circuit, bindings, **options)

    def to_native(self, **options):
        from .conversion import convert

        return convert(self.circuit, target=self.target_format, **options).value
