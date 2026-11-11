"""Deep quantum residual channels with reusable ancilla and identity growth.

Each block implements (1-p) rho + p U rho U†, with p=sin²(alpha/2).
The data register persists across blocks. Reset discards the ancilla without
postselection. Local readouts and identity initialization provide explicit
controls over circuit depth, residual strength and retained-state width.
"""

import math
from dataclasses import dataclass, field, replace

import numpy as np

from ..errors import ResourceError, ValidationError
from ..ir import CircuitBuilder, integer
from ..ir.observables import PauliString
from ..ir.parameters import Parameter, real
from .layers import LayerCheckpoint, LayerSpec


@dataclass(frozen=True)
class DeepQuantumPlan:
    num_qubits: int
    depth: int = 4
    reupload: bool = True
    initial_mixing: float = 0.1
    max_operations: int = 100_000
    spec: LayerSpec = field(init=False, repr=False)
    initialization_pairs: tuple = field(init=False, repr=False)
    initialization_values: tuple = field(init=False, repr=False)

    def __post_init__(self):
        n = integer(self.num_qubits, "num_qubits", 1)
        depth = integer(self.depth, "depth", 1)
        integer(self.max_operations, "max_operations", 1)
        if type(self.reupload) is not bool:
            raise ValidationError("reupload must be a boolean.")
        p = real(self.initial_mixing)
        if not 0 < p < 1:
            raise ValidationError("initial_mixing must lie strictly between zero and one.")
        per_block = 4 * n + 6 * (n - 1) + 2 + (2 * n if self.reupload else 0)
        if n + depth * per_block > self.max_operations:
            raise ResourceError("Deep quantum construction exceeds max_operations.")
        # q0 is the resettable ancilla; q1..qn hold the persistent quantum data.
        builder = CircuitBuilder(n + 1)
        inputs = tuple(f"deep_x_{i}" for i in range(n))
        for i, name in enumerate(inputs, 1):
            builder.ry(Parameter(name), i)
        gates = tuple((axis, (i,)) for i in range(1, n + 1) for axis in ("ry", "rz"))
        gates += tuple(("rzz", (i, i + 1)) for i in range(1, n))
        weights, pairs, constants = [], [], []

        def controlled(gate, wires, angle):
            if gate == "rzz":
                a, b = wires
                # Controlled RZZ via CX - controlled RZ - CX; control=0
                # cancels both CXs, so the bypass is exactly the identity.
                builder.cx(a, b).crz(angle, 0, b).cx(a, b)
            else:
                builder.gate("c" + gate, 0, *wires, params=(angle,))

        for layer in range(depth):
            if self.reupload:
                for i, name in enumerate(inputs, 1):
                    builder.rz(Parameter(name), i)
            constants.append((len(weights), 2 * math.asin(math.sqrt(p))))
            mix = f"deep_{layer}_mix"
            weights.append(mix)
            builder.ry(Parameter(mix), 0)
            forward = []
            for index, (gate, wires) in enumerate(gates):
                name = f"deep_{layer}_f_{index}"
                forward.append(len(weights))
                weights.append(name)
                controlled(gate, wires, Parameter(name))
            for index in reversed(range(len(gates))):
                name = f"deep_{layer}_r_{index}"
                pairs.append((forward[index], len(weights)))
                weights.append(name)
                gate, wires = gates[index]
                controlled(gate, wires, Parameter(name))
            builder.reset(0)
            if self.reupload:
                for i, name in reversed(tuple(enumerate(inputs, 1))):
                    builder.rz(-Parameter(name), i)
        observables = tuple(PauliString("I" * i + "Z" + "I" * (n - i)) for i in range(1, n + 1))
        object.__setattr__(
            self, "spec", LayerSpec(builder.build(), inputs, tuple(weights), observables)
        )
        object.__setattr__(self, "initialization_pairs", tuple(pairs))
        object.__setattr__(self, "initialization_values", tuple(constants))

    @property
    def num_parameters(self):
        return len(self.spec.weight_parameters)

    def to_spec(self):
        return self.spec

    def initial_weights(self, *, seed=None, scale=0.1):
        if real(scale) < 0:
            raise ValidationError("scale must be nonnegative.")
        values = np.random.default_rng(seed).normal(0, scale, self.num_parameters)
        for forward, reverse in self.initialization_pairs:
            values[reverse] = -values[forward]
        for index, value in self.initialization_values:
            values[index] = value
        return values

    def grow(self, checkpoint, *, add_blocks=1, seed=None, scale=0.1):
        """Return a deeper plan/checkpoint with exactly preserved initial readouts.

        Existing named parameters are copied bit-for-bit. New blocks start at
        identity. Framework optimizer resizing remains an explicit caller step.
        """
        if not isinstance(checkpoint, LayerCheckpoint) or checkpoint.spec != self.spec:
            raise ValidationError("Growth requires a checkpoint of this exact deep quantum plan.")
        added = integer(add_blocks, "add_blocks", 1)
        plan = replace(self, depth=self.depth + added)
        values = plan.initial_weights(seed=seed, scale=scale).astype(checkpoint.dtype)
        values[: self.num_parameters] = np.asarray(checkpoint.weights, dtype=checkpoint.dtype)
        return plan, LayerCheckpoint(plan.spec, tuple(values), checkpoint.dtype)

    def to_dict(self):
        return {
            "architecture": "deep_quantum_residual_channels",
            "data_qubits": self.num_qubits,
            "physical_qubits": self.num_qubits + 1,
            "depth": self.depth,
            "parameters": self.num_parameters,
            "reupload": self.reupload,
            "channel": "(1-p)*rho + p*U*rho*U_dagger; p=sin(alpha/2)^2",
            "postselection": False,
            "growth": "identity_initialized_independent_parameters",
            "simulation": "exact_mixed_state; exponential_cost_can_remain",
        }
