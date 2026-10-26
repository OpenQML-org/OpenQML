"""Cirq feedback -> IR -> hybrid Torch training and core OpenQASM roundtrip."""

import cirq
import numpy as np
import sympy
import torch

from openqml import CirqDensityBackend, convert, density_matrix, sample
from openqml.interfaces.torch import QuantumLayer

q = cirq.LineQubit.range(2)
feature, weight = sympy.symbols("feature weight")
native = cirq.Circuit(
    cirq.ry(feature)(q[0]),
    cirq.measure(q[0], key="syndrome"),
    cirq.ry(weight**2)(q[1]).with_classical_controls("syndrome"),
)
result = convert(native, target="ir", source_options={"measurement_map": {"syndrome": (0,)}})
circuit = result.value
layer = QuantumLayer(circuit, ["feature"], ["weight"], dtype=torch.float64)
model = torch.nn.Sequential(layer, torch.nn.Linear(2, 1, dtype=torch.float64))
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
inputs = torch.tensor([[0.2], [0.7], [1.3]], dtype=torch.float64)
targets = torch.tensor([[0.5], [0.1], [-0.2]], dtype=torch.float64)
for _ in range(20):
    optimizer.zero_grad()
    loss = ((model(inputs) - targets) ** 2).mean()
    loss.backward()
    optimizer.step()

qasm = convert(circuit, target="openqasm3").value
restored = convert(
    qasm, source="openqasm3", target="ir", source_options={"parser": "builtin"}
).value
bindings = {"feature": 0.7, "weight": float(layer.weights.detach()[0])}
np.testing.assert_allclose(density_matrix(restored, bindings), density_matrix(circuit, bindings))
print("hybrid loss:", float(loss.detach()))
print(
    "native feedback counts:",
    sample(circuit.measure(1, 1), 128, bindings=bindings, backend=CirqDensityBackend(), seed=9),
)
