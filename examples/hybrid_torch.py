"""Train classical and quantum weights together. Requires openqml[torch]."""

import torch

from openqml.interfaces.torch import QuantumLayer

torch.manual_seed(7)
inputs = torch.rand(20, 2)
targets = inputs[:, :1] - 0.4 * inputs[:, 1:]
model = torch.nn.Sequential(
    torch.nn.Linear(2, 2),
    QuantumLayer.hardware_efficient(2, depth=1),
    torch.nn.Linear(2, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
history = []
for step in range(60):
    optimizer.zero_grad()
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()
    history.append(loss.item())
    if step % 20 == 0 or step == 59:
        print(f"step={step:02d} loss={loss.item():.6f}")
assert history[-1] < history[0]
assert model[1].weights.grad is not None
