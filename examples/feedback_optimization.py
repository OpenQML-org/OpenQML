"""Compile repeated feedback before constructing a trainable quantum layer."""

import torch

from openqml import Circuit, ClassicalCondition, Parameter, convert, optimize
from openqml.interfaces.torch import QuantumLayer

x, w = Parameter("feature"), Parameter("weight")
circuit = Circuit(2).ry(x, 0).measure(0, 0)
for _ in range(32):
    circuit = circuit.if_else(
        ClassicalCondition((0,), 1), Circuit(2).ry(w * w, 1), Circuit(2).rx(w / 8, 1)
    )
report = optimize(circuit)
assert report.circuit.parameters == circuit.parameters
assert report.before.conditionals == 32 and report.after.conditionals == 1
print("gates:", report.before.gates, "->", report.after.gates)

quantum = QuantumLayer(report.circuit, ("feature",), ("weight",))
model = torch.nn.Sequential(torch.nn.Linear(3, 1), quantum, torch.nn.Linear(2, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
inputs = torch.ones(8, 3)
for _ in range(10):
    optimizer.zero_grad()
    loss = (model(inputs) - 0.2).square().mean()
    loss.backward()
    optimizer.step()
print("hybrid loss:", float(loss.detach()))

qasm = convert(circuit, target="openqasm3", optimize=True)
assert qasm.optimization.after.conditionals == 1
print("QASM lines:", len(qasm.value.splitlines()))
