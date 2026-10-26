"""A residual classical/quantum MLP importing real Qiskit library circuits."""

import torch
from sdk_programs import build_qiskit_program

from openqml.interfaces.torch import QuantumLayer

torch.set_num_threads(1)
torch.manual_seed(23)
circuit, inputs, weights = build_qiskit_program()


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(8, 12), torch.nn.GELU(), torch.nn.Linear(12, 4)
        )
        self.quantum = QuantumLayer.from_program(circuit, inputs, weights)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.GELU(), torch.nn.Linear(8, 1)
        )

    def forward(self, x):
        features = torch.tanh(self.encoder(x))
        quantum = self.quantum(features)
        return self.head(torch.cat((features, quantum), dim=-1))


model = Model()
x = torch.randn(32, 8)
target = 0.4 * x[:, :1] - 0.2 * x[:, 1:2] + 0.1 * x[:, 2:3] ** 2
optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
losses = []
for step in range(60):
    optimizer.zero_grad()
    loss = (model(x) - target).square().mean()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if step % 20 == 0 or step == 59:
        print(f"step={step:02d} loss={loss.item():.7f}")
assert losses[-1] < 0.2 * losses[0]
assert model.quantum.weights.grad is not None
print("Qiskit circuit, nonlinear feature map, encoder and quantum weights trained together.")
