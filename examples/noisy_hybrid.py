"""One optimizer trains classical layers, feedback rotations and channel probability."""

import torch

from openqml import Circuit, ClassicalCondition, Parameter, PauliString
from openqml.interfaces.torch import QuantumLayer

torch.manual_seed(37)
x, w, noise = Parameter("x"), Parameter("w"), Parameter("noise")
p = 1 / (1 + (-noise).exp())
circuit = (
    Circuit(2)
    .ry(x, 0)
    .measure(0, 0)
    .if_else(ClassicalCondition((0,), 1), Circuit(2).ry(w, 1), Circuit(2).rx(w / 3, 1))
    .amplitude_damping(p, 1)
    .bit_flip(p / 4, 0)
)
quantum = QuantumLayer(circuit, ("x",), ("w", "noise"), (PauliString("ZI"), PauliString("IZ")))
model = torch.nn.Sequential(torch.nn.Linear(2, 1), quantum, torch.nn.Linear(2, 1))
inputs = torch.rand(16, 2) * 2 - 1
targets = inputs[:, :1] * 0.7 - inputs[:, 1:] * 0.2
optimizer = torch.optim.Adam(model.parameters(), lr=0.06)
losses = []
for _ in range(55):
    optimizer.zero_grad()
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
assert losses[-1] < losses[0] * 0.12
print(f"Adaptive noisy hybrid loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
print("Learned damping probability:", torch.sigmoid(quantum.weights[1]).item())
