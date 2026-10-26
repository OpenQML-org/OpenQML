"""Train in Torch and restore the quantum layer in JAX and Flax."""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from openqml import LayerCheckpoint, convert
from openqml.interfaces.flax import QuantumLayer as FlaxLayer
from openqml.interfaces.jax import QuantumLayer as JaxLayer
from openqml.interfaces.torch import QuantumLayer as TorchLayer

torch.manual_seed(7)
layer = TorchLayer.hardware_efficient(2, 1, dtype=torch.float64, optimize=True)
features = torch.tensor([[0.2, -0.3], [0.7, 0.4]], dtype=torch.float64)
optimizer = torch.optim.Adam(layer.parameters(), lr=0.03)
for _ in range(5):
    optimizer.zero_grad()
    loss = (layer(features) - 0.2).square().mean()
    loss.backward()
    optimizer.step()

checkpoint = LayerCheckpoint.from_json(layer.checkpoint().to_json())
reference = layer(features).detach().numpy()
restored = TorchLayer.from_checkpoint(checkpoint, optimize=True)
torch.testing.assert_close(restored(features), layer(features))

with jax.enable_x64():
    jax_layer, weights = JaxLayer.from_checkpoint(checkpoint, optimize=True)
    jax_features = jnp.asarray(features.numpy())
    np.testing.assert_allclose(jax.jit(jax_layer)(weights, jax_features), reference, atol=1e-12)
    flax_layer, variables = FlaxLayer.from_checkpoint(checkpoint, optimize=True)
    np.testing.assert_allclose(
        jax.jit(flax_layer.apply)(variables, jax_features), reference, atol=1e-12
    )
    gradient = jax.jit(jax.grad(lambda w: jnp.square(jax_layer(w, jax_features)).mean()))(weights)
    assert np.isfinite(gradient).all()

print("Restored named weights:", len(checkpoint.bindings))
print("Torch, JAX and Flax predictions agree.")
print(convert(restored.circuit, target="openqasm3").value)
