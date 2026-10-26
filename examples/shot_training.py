"""One Adam optimizer trains a classical encoder, shot-based QNN and classical head."""

import torch

from openqml import Circuit, Hamiltonian, PauliString, ShotBackend, estimate
from openqml.interfaces.torch import QuantumLayer

torch.manual_seed(31)
quantum = QuantumLayer.hardware_efficient(2, 1, backend=ShotBackend(shots=512, seed=7))
model = torch.nn.Sequential(torch.nn.Linear(2, 2), quantum, torch.nn.Linear(2, 1))
x = torch.rand(8, 2)
target = x[:, :1] - 0.4 * x[:, 1:]
optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
losses = []
for _ in range(35):
    optimizer.zero_grad()
    loss = (model(x) - target).square().mean()
    loss.backward()
    optimizer.step()
    losses.append(float(loss.detach()))
print(f"Sampled hybrid regression: {losses[0]:.6f} -> {losses[-1]:.6f}")
assert losses[-1] < losses[0] * 0.3
assert quantum.weights.grad is not None

observables = (
    PauliString("ZI"),
    PauliString("IZ"),
    Hamiltonian((PauliString("ZI", 2), PauliString("IZ", -0.5))),
)
report = estimate(Circuit(2).h(0).cx(0, 1), observables, shots=1000, seed=19)
print("Shared readouts:", report.values)
print("Standard errors:", report.standard_errors)
print("Shots per original circuit:", report.shots_per_circuit)
assert len(report.plan.groups) == 1
