"""Build a custom trainable layer with shared parameters and Hamiltonian output."""

import torch

from openqml import Circuit, Hamiltonian, Parameter, PauliString
from openqml.interfaces.torch import QuantumLayer

x, w = Parameter("input_angle"), Parameter("shared_weight")
circuit = Circuit(2).ry(x, 0).rx(w, 1).cx(0, 1).ry(2 * w, 0)
observable = Hamiltonian((PauliString("ZI", 0.7), PauliString("XX", -0.2)))
layer = QuantumLayer(circuit, ("input_angle",), ("shared_weight",), (observable,))
inputs = torch.tensor([[0.2], [0.5]], requires_grad=True)
prediction = layer(inputs)
prediction.square().mean().backward()
print("Output:", prediction.detach())
print("Input gradient:", inputs.grad)
print("Shared weight gradient:", layer.weights.grad)
