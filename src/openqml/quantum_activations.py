"""Data-reuploading quantum univariate activations with trainable frequencies."""

from dataclasses import dataclass, field

from .errors import ValidationError
from .ir import CircuitBuilder, integer
from .layers import LayerSpec
from .observables import PauliString
from .parameters import Parameter


@dataclass(frozen=True)
class QuantumActivationPlan:
    depth: int = 3
    num_outputs: int = 3
    max_operations: int = 100_000
    spec: LayerSpec = field(init=False, repr=False)

    def __post_init__(self):
        depth = integer(self.depth, "depth", 1)
        outputs = integer(self.num_outputs, "num_outputs", 1)
        limit = integer(self.max_operations, "max_operations", 1)
        if outputs > 3 or 3 * depth > limit:
            raise ValidationError(
                "A quantum activation requires 1..3 readouts and 3*depth operations."
            )
        for name, value in (("depth", depth), ("num_outputs", outputs), ("max_operations", limit)):
            object.__setattr__(self, name, value)
        builder = CircuitBuilder(1)
        names = tuple(f"activation_{i}" for i in range(6 * depth))
        for layer in range(depth):
            for axis, gate in enumerate(("rx", "ry", "rz")):
                i = 6 * layer + 2 * axis
                builder.gate(
                    gate,
                    0,
                    params=(
                        Parameter(names[i]) * Parameter("activation_input")
                        + Parameter(names[i + 1]),
                    ),
                )
        object.__setattr__(
            self,
            "spec",
            LayerSpec(
                builder.build(),
                ("activation_input",),
                names,
                tuple(PauliString(p) for p in "XYZ"[:outputs]),
            ),
        )

    @property
    def initialization_values(self):
        return tuple((i, 2 ** ((i // 6) % 8)) for i in range(0, 6 * self.depth, 2))

    def to_spec(self):
        return self.spec

    def to_dict(self):
        return {
            "architecture": "quantum_variational_activation",
            "depth": self.depth,
            "readouts": self.num_outputs,
            "parameters": len(self.spec.weight_parameters),
            "frequencies": "trainable",
        }
