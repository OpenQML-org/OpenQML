"""Linen ownership of the same canonical parameters and model program as JAX."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..architecture import HybridArchitecture
from ..training import StagewiseOptimizer
from ._jax_network import HybridNetwork as FunctionalNetwork
from ._jax_training import make_train_step


class HybridNetwork(nn.Module):
    architecture: HybridArchitecture
    active_depth: int | None = None
    backend: object = None
    dtype: object = jnp.float32

    @nn.compact
    def __call__(self, inputs, *, state=None, return_state=False):
        parameters = {}
        for spec in self.architecture.parameters:

            def initialize(key, shape, dtype, name=spec.name):
                return self.architecture.initialize_one(
                    name,
                    lambda s: jax.random.normal(key, s, dtype=dtype),
                    lambda s, v: jnp.full(s, v, dtype=dtype),
                    backend="jax",
                )

            parameters[spec.name] = self.param(spec.name, initialize, spec.shape, self.dtype)
        model = FunctionalNetwork(
            self.architecture, active_depth=self.active_depth, backend=self.backend
        )
        return model(parameters, inputs, state=state, return_state=return_state)

    def optimizer(self, optimizer):
        return StagewiseOptimizer(
            self.architecture, optimizer, self.architecture.active(self.active_depth)
        )

    def sample_tokens(self, variables, key, steps, *, prompt=None, batch_size=1):
        return FunctionalNetwork(
            self.architecture, active_depth=self.active_depth, backend=self.backend
        ).sample_tokens(variables["params"], key, steps, prompt=prompt, batch_size=batch_size)

    def make_train_step(self, optimizer, loss_function, **options):
        """Optimize the flat variables['params'] tree; loss(params,batch) is per-example."""
        options.setdefault("gradient_metrics", self.architecture.gradient_metrics)
        return make_train_step(self.optimizer(optimizer), loss_function, **options)

    def trainer(self, optimizer, loss_function, *, variables=None, key=None, **options):
        model = FunctionalNetwork(
            self.architecture, active_depth=self.active_depth, backend=self.backend
        )
        return model.trainer(
            optimizer,
            loss_function,
            parameters=None if variables is None else variables["params"],
            key=key,
            **options,
        )
