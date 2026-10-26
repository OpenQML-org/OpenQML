import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from openqml import Circuit, Parameter, PauliString, expectation
from openqml.interfaces.jax import QuantumLayer


def test_jit_grad_vmap_and_second_derivatives():
    circuit = Circuit(1).ry(Parameter("t"), 0).ry(2 * Parameter("t") + 0.1, 0)

    def function(t):
        return expectation(circuit, PauliString("Z"), {"t": t}, backend="jax")

    theta = jnp.array(0.21, dtype=jnp.float32)
    gradient = jax.jit(jax.grad(function))(theta)
    second = jax.jit(jax.grad(jax.grad(function)))(theta)
    np.testing.assert_allclose(gradient, -3 * np.sin(3 * float(theta) + 0.1), atol=2e-6)
    np.testing.assert_allclose(second, -9 * np.cos(3 * float(theta) + 0.1), atol=3e-6)
    np.testing.assert_allclose(
        jax.vmap(function)(jnp.array([0.1, 0.2])), np.cos(np.array([0.4, 0.7])), atol=1e-6
    )


def test_jax_layer_training_inputs_and_parameter_gradients():
    layer = QuantumLayer.hardware_efficient(2, depth=1)
    weights = layer.init(jax.random.key(17))
    inputs = jnp.array([[0.1, -0.2], [0.5, 0.3]], dtype=jnp.float32)
    output = jax.jit(layer)(weights, inputs)
    assert output.shape == (2, 2)

    def loss(w, x):
        return jnp.mean((layer(w, x) - 0.3) ** 2)

    value, (dw, dx) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(weights, inputs)
    assert dw.shape == weights.shape and dx.shape == inputs.shape
    assert np.isfinite(dw).all() and np.linalg.norm(dw) > 0
    assert np.isfinite(dx).all() and np.linalg.norm(dx) > 0
    train = jax.jit(jax.value_and_grad(loss))
    for _ in range(25):
        _, gradient = train(weights, inputs)
        weights = weights - 0.2 * gradient
    assert float(loss(weights, inputs)) < 0.3 * float(value)
    assert layer(weights, jnp.zeros((3, 4, 2))).shape == (3, 4, 2)
    assert layer(weights, jnp.zeros((0, 2))).shape == (0, 2)


def test_constant_zero_feature_layer():
    layer = QuantumLayer(Circuit(1).h(0), (), ())
    assert layer(jnp.empty(0), jnp.empty((4, 0))).shape == (4, 1)


def test_extended_gate_jit_and_gradients_match_shift_reference():
    from openqml import parameter_shift

    t = Parameter("theta")
    circuit = Circuit(2).h(0).u(t, 0.31, -0.27, 1).crx(2 * t, 0, 1).cp(t, 1, 0)
    observable = PauliString("XZ")

    def loss(theta):
        return expectation(circuit, observable, {"theta": theta}, backend="jax")

    gradient = jax.jit(jax.grad(loss))(jnp.array(0.43, dtype=jnp.float32))
    expected = parameter_shift(circuit, observable, {"theta": 0.43})["theta"]
    np.testing.assert_allclose(gradient, expected, atol=3e-6)
