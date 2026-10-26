"""Directional sensitivities without a full Jacobian, including external engines."""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from openqml import Circuit, Parameter, PauliString, parameter_shift_jvp
from openqml.interfaces.jax import QuantumLayer as JaxLayer
from openqml.interfaces.torch import QuantumLayer as TorchLayer

feature, weight = Parameter("feature"), Parameter("weight")
circuit = Circuit(1).ry(feature * weight, 0)
rows = [{"feature": 0.4, "weight": 0.7}, {"feature": 0.9, "weight": 0.7}]
directions = np.array([[0.2, -0.1], [0.2, -0.1]])  # circuit.parameters order
print("numeric JVP:", parameter_shift_jvp(circuit, [PauliString("Z")], rows, directions))

layer = TorchLayer(circuit, ["feature"], ["weight"], backend="numpy", dtype=torch.float64)
with torch.no_grad():
    layer.weights.fill_(0.7)
x = torch.tensor([[0.4], [0.9]], dtype=torch.float64)
with torch.autograd.forward_ad.dual_level():
    dual_x = torch.autograd.forward_ad.make_dual(x, torch.full_like(x, 0.2))
    dual_w = torch.autograd.forward_ad.make_dual(
        layer.weights, torch.full_like(layer.weights, -0.1)
    )
    dual_y = torch.func.functional_call(layer, {"weights": dual_w}, (dual_x,))
    primal, tangent = torch.autograd.forward_ad.unpack_dual(dual_y)
    print("Torch external JVP:", tangent)

jax_layer = JaxLayer(circuit, ["feature"], ["weight"], backend="numpy")
w, xj = jnp.array([0.7], dtype=jnp.float32), jnp.asarray(x.numpy(), dtype=jnp.float32)
primal, tangent = jax.jit(jax_layer.jvp)(w, xj, jnp.full_like(w, -0.1), jnp.full_like(xj, 0.2))
print("JAX external JVP:", tangent)
