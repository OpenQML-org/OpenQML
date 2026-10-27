"""Train a hybrid model with damped Gauss-Newton products, without a Jacobian.

The numerical quantum backend supplies first-order JVPs and VJPs. At a fixed
parameter point, JAX linearizes the complete encoder/quantum/head model and
transposes that linear map. The resulting normal operator is a Gauss-Newton
approximation, not an exact loss Hessian. Nonlinear higher derivatives require
a native tensor backend.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from openqml.interfaces.jax import QuantumLayer


def normal_product(push, pull, direction):
    product = pull(push(direction))[0]
    return jax.tree_util.tree_map(lambda a, v: a + 0.1 * v, product, direction)


def main():
    layer = QuantumLayer.data_reuploading(
        2, depth=2, backend="numpy", gradient_method="adjoint", batch_chunk_size=8
    )
    key = jax.random.key(71)
    inputs = jax.random.normal(key, (12, 3))
    targets = jnp.sin(inputs[:, 0]) + 0.2 * inputs[:, 1]
    params = {
        "encoder": jax.random.normal(jax.random.fold_in(key, 1), (3, 2)) * 0.3,
        "bias": jnp.zeros(2),
        "quantum": layer.init(jax.random.fold_in(key, 2)),
        "head": jnp.array([0.4, -0.3]),
        "offset": jnp.zeros(()),
    }

    def residual(p):
        encoded = jnp.tanh(inputs @ p["encoder"] + p["bias"])
        prediction = layer(p["quantum"], encoded) @ p["head"] + p["offset"]
        return (prediction - targets) / jnp.sqrt(inputs.shape[0])

    loss = jax.jit(lambda p: jnp.vdot(residual(p), residual(p)).real / 2)
    initial = float(loss(params))
    for _ in range(4):
        values, push = jax.linearize(residual, params)
        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        pull = jax.linear_transpose(push, zeros)

        normal = jax.jit(partial(normal_product, push, pull))
        rhs = jax.tree_util.tree_map(jnp.negative, pull(values)[0])
        step, _ = cg(normal, rhs, tol=1e-5, maxiter=20)
        before = float(loss(params))
        for scale in (1.0, 0.5, 0.25, 0.125):
            candidate = jax.tree_util.tree_map(lambda p, d, s=scale: p + s * d, params, step)
            if float(loss(candidate)) < before:
                params = candidate
                break
    final = float(loss(params))
    assert final < initial
    print(f"Hybrid Gauss-Newton loss: {initial:.6f} -> {final:.6f}")


if __name__ == "__main__":
    main()
