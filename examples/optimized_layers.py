"""Import QASM, optimize a trainable layer, and reuse its original checkpoint."""

import torch

from openqml import Circuit, Parameter, convert
from openqml.interfaces.torch import QuantumLayer

x, unused_x = Parameter("feature"), Parameter("unused_input")
w, unused_w = Parameter("weight"), Parameter("unused_weight")
source = Circuit(1).ry(unused_x, 0).ry(-unused_x, 0).rx(unused_w, 0).rx(-unused_w, 0).ry(x * w, 0)
qasm = convert(source, target="openqasm3").value
imported = convert(
    qasm, source="openqasm3", target="ir", source_options={"parser": "builtin"}
).value
names = (("unused_input", "feature"), ("unused_weight", "weight"))
original = QuantumLayer(imported, *names)
optimized = QuantumLayer(imported, *names, optimize=True)
optimized.load_state_dict(original.state_dict())
inputs = torch.tensor([[0.2, 0.4], [0.3, 0.8]])
torch.testing.assert_close(optimized(inputs), original(inputs))
assert optimized.weights.shape == original.weights.shape == (2,)

model = torch.nn.Sequential(torch.nn.Linear(3, 2), optimized, torch.nn.Linear(1, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for _ in range(5):
    optimizer.zero_grad()
    loss = (model(torch.ones(4, 3)) - 0.3).square().mean()
    loss.backward()
    assert optimized.weights.grad[0] == 0
    optimizer.step()

report = optimized.optimization
print("executable gates:", report.before.gates, "->", report.after.gates)
print("zero-gradient parameters:", report.removed_parameters)
print("hybrid loss:", float(loss.detach()))
print(convert(optimized.circuit, target="openqasm3").value)
