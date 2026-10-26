"""Conditional quantum weights must differentiate back into their classical generator."""

import numpy as np
import pytest

from openqml import Circuit, ClassicalCondition, Parameter, ValidationError


def rotation():
    return Circuit(1).ry(Parameter("x") + Parameter("w"), 0)


@pytest.mark.parametrize("backend", ["torch", "numpy"])
@pytest.mark.parametrize("chunk_size", [None, 2])
@pytest.mark.parametrize("shared", ["weights", "inputs", "both", "empty"])
def test_torch_broadcast_values_and_pullbacks(backend, chunk_size, shared):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    layer = QuantumLayer(
        rotation(),
        ["x"],
        ["w"],
        backend=backend,
        batch_chunk_size=chunk_size,
        dtype=torch.float64,
    )
    w = torch.tensor([0.2, 0.4], dtype=torch.float64).reshape(2, 1, 1)
    x = torch.tensor([0.1, 0.3, 0.5], dtype=torch.float64).reshape(3, 1)
    if shared == "weights":
        w = w[0, 0]
    elif shared == "inputs":
        x = x[0]
    elif shared == "empty":
        w = w[:0]
    w, x = w.requires_grad_(), x.requires_grad_()
    before = layer.weights.detach().clone()
    actual, expected = layer.functional(w, x), torch.cos(w + x)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), (w, x))
    expected_grad = torch.autograd.grad(expected.square().sum(), (w, x))
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(layer.weights, before)
    assert layer.weights.grad is None


@pytest.mark.parametrize("backend", ["jax", "numpy"])
@pytest.mark.parametrize("chunk_size", [None, 2])
def test_jax_broadcast_jit_vmap_pullbacks_and_directional_derivatives(backend, chunk_size):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    layer = QuantumLayer(rotation(), ["x"], ["w"], backend=backend, batch_chunk_size=chunk_size)
    w, x = jnp.array([0.2, 0.4]).reshape(2, 1, 1), jnp.array([0.1, 0.3, 0.5]).reshape(3, 1)
    expected = jnp.cos(w + x)
    np.testing.assert_allclose(jax.jit(layer)(w, x), expected, atol=3e-7)
    actual_grad = jax.jit(jax.grad(lambda a, b: jnp.square(layer(a, b)).sum(), argnums=(0, 1)))(
        w, x
    )
    expected_grad = jax.grad(lambda a, b: jnp.square(jnp.cos(a + b)).sum(), argnums=(0, 1))(w, x)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        np.testing.assert_allclose(a, b, atol=8e-7)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(layer, in_axes=(0, None)))(w, x), expected, atol=3e-7
    )
    dw, dx = jnp.full_like(w, 0.3), jnp.full_like(x, -0.1)
    _, tangent = jax.jit(layer.jvp)(w, x, dw, dx)
    np.testing.assert_allclose(tangent, -jnp.sin(w + x) * (dw + dx), atol=5e-7)
    empty = jax.jit(layer)(w[:0], x)
    assert empty.shape == (0, 3, 1)
    gradients = jax.grad(lambda a: layer(a, x).sum())(w[:0])
    assert gradients.shape == w[:0].shape


def test_torch_generated_weights_joint_optimizer_and_higher_derivatives():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    torch.manual_seed(12)
    layer = QuantumLayer(rotation(), ["x"], ["w"], batch_chunk_size=2, dtype=torch.float64)
    generator = torch.nn.Linear(2, 1, dtype=torch.float64)
    encoder = torch.nn.Linear(2, 1, dtype=torch.float64)
    data = torch.ones(5, 2, dtype=torch.float64)
    optimizer = torch.optim.Adam([*generator.parameters(), *encoder.parameters()], lr=0.01)
    before = [p.detach().clone() for p in (*generator.parameters(), *encoder.parameters())]
    generated, inputs = generator(data), encoder(data)
    output = layer.functional(generated, inputs)
    first = torch.autograd.grad(output.sum(), generated, create_graph=True, retain_graph=True)[0]
    second = torch.autograd.grad(first.sum(), generated, retain_graph=True)[0]
    torch.testing.assert_close(second, -torch.cos(generated + inputs))
    output.square().mean().backward()
    optimizer.step()
    assert all(
        not torch.equal(a, b)
        for a, b in zip(before, (*generator.parameters(), *encoder.parameters()), strict=True)
    )


def test_jax_conditional_hessian_and_flax_single_parameter_tree():
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import flax.linen as nn
    import jax.numpy as jnp
    import optax

    from openqml.interfaces.flax import QuantumLayer as FlaxLayer
    from openqml.interfaces.jax import QuantumLayer

    layer = QuantumLayer(rotation(), ["x"], ["w"], batch_chunk_size=2)
    w, x = jnp.array([[0.2], [0.4], [0.6]]), jnp.array([0.3])
    hessian = jax.jit(jax.hessian(lambda a: layer(a, x).sum()))(w)
    np.testing.assert_allclose(hessian.reshape(3, 3), np.diag(-np.cos(w[:, 0] + 0.3)), atol=3e-7)

    class Model(nn.Module):
        @nn.compact
        def __call__(self, data):
            inputs = nn.Dense(1, name="encoder")(data)
            weights = nn.Dense(1, name="generator")(data)
            return FlaxLayer(rotation(), ("x",), ("w",), batch_chunk_size=2)(
                inputs, weights=weights
            )

    model, data = Model(), jnp.ones((5, 2))
    params = model.init(jax.random.key(2), data)["params"]
    assert set(params) == {"encoder", "generator"}

    def loss(p):
        return jnp.square(model.apply({"params": p}, data)).mean()

    gradients = jax.jit(jax.grad(loss))(params)
    optimizer = optax.adam(0.01)
    updates, _ = optimizer.update(gradients, optimizer.init(params), params)
    updated = optax.apply_updates(params, updates)
    assert float(loss(updated)) < float(loss(params))
    assert all(np.isfinite(g).all() for g in jax.tree_util.tree_leaves(gradients))


@pytest.mark.parametrize("interface", ["torch", "jax"])
@pytest.mark.parametrize("backend", ["native", "numpy"])
def test_conditional_density_branches_and_optimized_ownership(interface, backend):
    pytest.importorskip(interface)
    circuit = (
        Circuit(1)
        .ry(Parameter("x"), 0)
        .measure(0, 0)
        .if_else(ClassicalCondition((0,), 1), Circuit(1).ry(Parameter("w"), 0))
        .rz(Parameter("unused"), 0)
        .rz(-Parameter("unused"), 0)
    )
    options = {"backend": interface if backend == "native" else backend, "optimize": True}
    if interface == "torch":
        import torch

        from openqml.interfaces.torch import QuantumLayer

        layer = QuantumLayer(circuit, ["x"], ["w", "unused"], dtype=torch.float64, **options)
        w = torch.tensor([[0.2, 0.4], [0.5, 0.6]], dtype=torch.float64, requires_grad=True)
        x = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
        expected = torch.cos(x / 2).square() - torch.sin(x / 2).square() * torch.cos(w[:, :1])
        actual = layer.functional(w, x)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(
            torch.autograd.grad(actual.sum(), (w, x)),
            torch.autograd.grad(expected.sum(), (w, x)),
            strict=True,
        ):
            torch.testing.assert_close(a, b)
    else:
        import jax
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        layer = QuantumLayer(circuit, ["x"], ["w", "unused"], **options)
        w, x = jnp.array([[0.2, 0.4], [0.5, 0.6]]), jnp.array([0.7])

        def expected(a, b):
            return jnp.cos(b / 2) ** 2 - jnp.sin(b / 2) ** 2 * jnp.cos(a[:, :1])

        np.testing.assert_allclose(layer(w, x), expected(w, x), atol=3e-7)
        for a, b in zip(
            jax.jit(jax.grad(lambda a, b: layer(a, b).sum(), argnums=(0, 1)))(w, x),
            jax.grad(lambda a, b: expected(a, b).sum(), argnums=(0, 1))(w, x),
            strict=True,
        ):
            np.testing.assert_allclose(a, b, atol=3e-7)


@pytest.mark.parametrize("interface", ["torch", "jax"])
def test_conditional_weights_validate_feature_width_and_batch_shapes(interface):
    module = pytest.importorskip(interface)
    if interface == "torch":
        from openqml.interfaces.torch import QuantumLayer

        layer = QuantumLayer(rotation(), ["x"], ["w"])
        call, array = layer.functional, module.tensor
    else:
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        call, array = QuantumLayer(rotation(), ["x"], ["w"]), jnp.array
    for w, x, message in [
        (0.2, [0.3], "weight shape"),
        ([0.2, 0.4], [0.3], "weight shape"),
        ([[0.2], [0.4]], [[0.1], [0.3], [0.5]], "broadcastable"),
    ]:
        with pytest.raises(ValidationError, match=message):
            call(array(w), array(x))


@pytest.mark.parametrize("interface", ["torch", "jax"])
@pytest.mark.parametrize("backend", ["native", "numpy"])
@pytest.mark.parametrize("empty", ["inputs", "weights"])
def test_zero_feature_width_with_broadcasted_weight_or_input_batches(interface, backend, empty):
    pytest.importorskip(interface)
    name = "w" if empty == "inputs" else "x"
    circuit = Circuit(1).ry(Parameter(name), 0)
    inputs, weights = (() if empty == "inputs" else ("x",)), (() if empty == "weights" else ("w",))
    w = np.full((2, 1, len(weights)), 0.3)
    x = np.full((3, len(inputs)), 0.4)
    options = {"backend": interface if backend == "native" else backend, "batch_chunk_size": 2}
    expected = np.full((2, 3, 1), np.cos(0.3 if empty == "inputs" else 0.4))
    if interface == "torch":
        import torch

        from openqml.interfaces.torch import QuantumLayer

        layer = QuantumLayer(circuit, inputs, weights, dtype=torch.float64, **options)
        w, x = torch.tensor(w, requires_grad=True), torch.tensor(x, requires_grad=True)
        actual = layer.functional(w, x)
        np.testing.assert_allclose(actual.detach(), expected)
        dw, dx = torch.autograd.grad(actual.sum(), (w, x))
    else:
        import jax
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        layer = QuantumLayer(circuit, inputs, weights, **options)
        w, x = jnp.asarray(w), jnp.asarray(x)
        np.testing.assert_allclose(jax.jit(layer)(w, x), expected, atol=3e-7)
        dw, dx = jax.jit(jax.grad(lambda a, b: layer(a, b).sum(), argnums=(0, 1)))(w, x)
    assert (dw.shape, dx.shape) == (w.shape, x.shape)
    np.testing.assert_allclose(dw, -3 * np.sin(0.3), atol=3e-7)
    np.testing.assert_allclose(dx, -2 * np.sin(0.4), atol=3e-7)


@pytest.mark.parametrize("backend", ["qiskit", "pennylane", "cirq"])
def test_sdk_execution_receives_conditional_weights_and_accumulates_broadcast_gradients(backend):
    pytest.importorskip(backend)
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    layer = QuantumLayer(rotation(), ["x"], ["w"], backend=backend, dtype=torch.float64)
    w = torch.tensor([[0.2], [0.4], [0.6]], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    output = layer.functional(w, x)
    torch.testing.assert_close(output, torch.cos(w + x))
    dw, dx = torch.autograd.grad(output.sum(), (w, x))
    torch.testing.assert_close(dw, -torch.sin(w + x))
    torch.testing.assert_close(dx, -torch.sin(w + x).sum(0))


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_torch_conditional_weight_forward_ad(backend):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    layer = QuantumLayer(rotation(), ["x"], ["w"], backend=backend, batch_chunk_size=2)
    w, x = torch.tensor([[0.2], [0.4], [0.6]]), torch.tensor([0.3])
    dw, dx = torch.full_like(w, 0.3), torch.full_like(x, -0.1)
    with torch.autograd.forward_ad.dual_level():
        output = layer.functional(
            torch.autograd.forward_ad.make_dual(w, dw),
            torch.autograd.forward_ad.make_dual(x, dx),
        )
        primal, tangent = torch.autograd.forward_ad.unpack_dual(output)
    torch.testing.assert_close(primal, torch.cos(w + x))
    torch.testing.assert_close(tangent, -torch.sin(w + x) * (dw + dx))
