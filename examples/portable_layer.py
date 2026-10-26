"""Reuse a QASM-derived layer definition and weights across Torch, JAX and Flax."""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from openqml import Circuit, Hamiltonian, LayerSpec, Parameter, PauliString, convert
from openqml.interfaces.flax import QuantumLayer as FlaxLayer
from openqml.interfaces.jax import QuantumLayer as JaxLayer
from openqml.interfaces.torch import QuantumLayer as TorchLayer

x, w = Parameter("feature"), Parameter("weight")
qasm = convert(Circuit(2).ry(x, 0).ry(w, 1).cx(0, 1), target="openqasm3").value
readouts = (PauliString("ZI"), Hamiltonian((PauliString("IZ", 0.7), PauliString("XX", 0.2))))
torch_layer = TorchLayer.from_program(
    qasm,
    ("feature",),
    framework="openqasm3",
    source_options={"parser": "builtin"},
    observables=readouts,
    dtype=torch.float64,
    optimize=True,
)
spec = LayerSpec.from_json(torch_layer.to_spec().to_json())
inputs = torch.tensor([[0.2], [0.5]], dtype=torch.float64, requires_grad=True)
expected = torch_layer(inputs)
expected.sum().backward()

with jax.enable_x64():
    weights = jnp.asarray(torch_layer.weights.detach().numpy())
    features = jnp.asarray(inputs.detach().numpy())
    jax_layer = JaxLayer.from_spec(spec, optimize=True)
    np.testing.assert_allclose(
        jax.jit(jax_layer)(weights, features), expected.detach().numpy(), atol=1e-12
    )
    gradient = jax.jit(jax.grad(lambda w: jax_layer(w, features).sum()))(weights)
    np.testing.assert_allclose(gradient, torch_layer.weights.grad.numpy(), atol=1e-12)
    flax_layer = FlaxLayer.from_spec(spec, dtype=jnp.float64, optimize=True)
    params = {"params": {"weights": weights}}
    np.testing.assert_allclose(
        flax_layer.apply(params, features), expected.detach().numpy(), atol=1e-12
    )

print(
    "Portable layer:", len(spec.input_parameters), "inputs,", len(spec.weight_parameters), "weights"
)
print("Torch / JAX / Flax outputs and Torch / JAX gradients agree.")
