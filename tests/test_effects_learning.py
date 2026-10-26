"""Hybrid optimization and higher derivatives through noisy adaptive ensembles."""

import numpy as np
import pytest

from openqml import Circuit, ClassicalCondition, Parameter, PauliString, expectation


def adaptive_circuit():
    x, w, noise = Parameter("x"), Parameter("w"), Parameter("noise")
    p = 1 / (1 + (-noise).exp())
    return (
        Circuit(2)
        .ry(x, 0)
        .measure(0, 0)
        .if_else(
            ClassicalCondition((0,), 1),
            Circuit(2).ry(w, 1),
            Circuit(2).rx(w / 3, 1),
        )
        .amplitude_damping(p, 1)
        .bit_flip(p / 4, 0)
    )


def test_torch_adaptive_noise_gradcheck_and_gradgradcheck():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    layer = QuantumLayer(
        adaptive_circuit(),
        ("x",),
        ("w", "noise"),
        (PauliString("IX"), PauliString("ZZ")),
        dtype=torch.float64,
    )
    inputs = torch.tensor([[0.37], [-0.41]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([0.6, -0.7], dtype=torch.float64, requires_grad=True)

    def function(x, w):
        return torch.func.functional_call(layer, {"weights": w}, (x,))

    assert torch.autograd.gradcheck(function, (inputs, weights), atol=1e-5)
    assert torch.autograd.gradgradcheck(function, (inputs, weights), atol=1e-5)
    torch.testing.assert_close(
        function(inputs, weights), torch.stack([function(row, weights) for row in inputs])
    )


def test_torch_classical_and_adaptive_quantum_parameters_share_optimizer():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    torch.manual_seed(37)
    layer = QuantumLayer(adaptive_circuit(), ("x",), ("w", "noise"))
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), layer, torch.nn.Linear(2, 1))
    inputs = torch.rand(16, 2) * 2 - 1
    targets = inputs[:, :1] * 0.7 - inputs[:, 1:] * 0.2
    optimizer = torch.optim.Adam(model.parameters(), lr=0.06)
    losses = []
    before = layer.weights.detach().clone()
    for _ in range(55):
        optimizer.zero_grad()
        loss = (model(inputs) - targets).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.12
    assert not torch.equal(before, layer.weights)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_torch_inactive_branch_phase_empty_batch_and_mixed_precision():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    c = Circuit(1).if_else(
        ClassicalCondition((0,), 1),
        Circuit(1).ry(Parameter("w"), 0).with_global_phase(Parameter("phase")),
    )
    layer = QuantumLayer(c, (), ("w", "phase"))
    inputs = torch.empty((3, 0), requires_grad=True)
    output = layer(inputs)
    torch.testing.assert_close(output, torch.ones((3, 1)))
    output.sum().backward()
    torch.testing.assert_close(layer.weights.grad, torch.zeros(2))
    layer.zero_grad()
    inputs = torch.empty((0, 0), dtype=torch.bfloat16, requires_grad=True)
    output = layer(inputs)
    assert output.shape == (0, 1) and output.dtype == torch.bfloat16
    output.sum().backward()
    assert inputs.grad is not None
    torch.testing.assert_close(layer.weights.grad, torch.zeros(2))


def test_external_aer_density_torch_channel_probability_vjp():
    torch = pytest.importorskip("torch")
    pytest.importorskip("qiskit_aer")
    from openqml.interfaces.torch import QuantumLayer

    x, n = Parameter("x"), Parameter("noise")
    c = Circuit(1).ry(x * x, 0).amplitude_damping(1 / (1 + (-n).exp()), 0)
    layer = QuantumLayer(
        c,
        ("x",),
        ("noise",),
        (PauliString("X"), PauliString("Z")),
        dtype=torch.float64,
        backend="aer-density",
    )
    inputs = torch.tensor([[0.27], [0.63]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([-0.37], dtype=torch.float64, requires_grad=True)

    def function(x, w):
        return torch.func.functional_call(layer, {"weights": w}, (x,))

    assert torch.autograd.gradcheck(function, (inputs, weights), atol=1e-5)


def test_jax_adaptive_jit_vmap_vjp_and_hessian_match_closed_form():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from openqml.interfaces.jax import QuantumLayer

    x, w = Parameter("x"), Parameter("w")
    c = (
        Circuit(2)
        .ry(x, 0)
        .measure(0, 0)
        .if_else(
            ClassicalCondition((0,), 1),
            Circuit(2).ry(w, 1),
        )
        .amplitude_damping(0.25, 1)
    )
    layer = QuantumLayer(c, ("x",), ("w",), (PauliString("IZ"),))

    def function(value):
        return layer(value[1:], value[:1])[0]

    def analytic(value):
        return 1 - 0.75 * (1 - jnp.cos(value[1])) * jnp.sin(value[0] / 2) ** 2

    value = jnp.array([0.4, 0.7], dtype=jnp.float32)
    np.testing.assert_allclose(jax.jit(function)(value), analytic(value), atol=3e-7)
    np.testing.assert_allclose(
        jax.jit(jax.grad(function))(value), jax.grad(analytic)(value), atol=4e-7
    )
    np.testing.assert_allclose(
        jax.jit(jax.hessian(function))(value), jax.hessian(analytic)(value), atol=4e-7
    )
    values = jnp.array([[0.2, 0.7], [0.4, 0.7], [-0.6, 0.7]])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(function))(values), jax.vmap(analytic)(values), atol=4e-7
    )
    _, vjp = jax.vjp(function, value)
    np.testing.assert_allclose(vjp(jnp.array(2.0))[0], 2 * jax.grad(analytic)(value), atol=4e-7)


def test_jax_traced_channel_domain_errors_propagate_nan():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    c = Circuit(1).bit_flip(Parameter("p"), 0)

    def function(p):
        return expectation(c, PauliString("Z"), {"p": p}, backend="jax")

    assert np.isnan(jax.jit(function)(jnp.array(1.1)))
    assert np.isfinite(jax.jit(jax.grad(function))(jnp.array(0.0)))


def test_flax_optax_joint_noisy_training_updates_entire_parameter_tree():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    nn = pytest.importorskip("flax.linen")
    optax = pytest.importorskip("optax")
    from openqml.interfaces.flax import QuantumLayer

    class Model(nn.Module):
        @nn.compact
        def __call__(self, inputs):
            x = nn.Dense(1, name="encoder")(inputs)
            x = QuantumLayer(adaptive_circuit(), ("x",), ("w", "noise"), name="quantum")(x)
            return nn.Dense(1, name="readout")(x)

    model = Model()
    x = jax.random.normal(jax.random.key(2), (12, 2))
    y = 0.5 * x[:, :1] - 0.2 * x[:, 1:]
    params = model.init(jax.random.key(37), x)["params"]
    initial_quantum = params["quantum"]["weights"]
    optimizer = optax.adam(0.05)
    state = optimizer.init(params)

    @jax.jit
    def step(params, state):
        loss, grads = jax.value_and_grad(
            lambda p: jnp.mean((model.apply({"params": p}, x) - y) ** 2)
        )(params)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss, grads

    losses = []
    for _ in range(60):
        params, state, loss, grads = step(params, state)
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.15
    assert not np.array_equal(initial_quantum, params["quantum"]["weights"])
    assert all(np.isfinite(g).all() for g in jax.tree_util.tree_leaves(grads))
