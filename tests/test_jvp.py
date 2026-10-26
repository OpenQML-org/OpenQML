"""Directional gradients: independent finite differences, AD and bounded work."""

import numpy as np
import pytest

from openqml import (
    BackendCapabilities,
    Circuit,
    ClassicalCondition,
    Parameter,
    PauliString,
    ValidationError,
    expectations_batch,
    parameter_shift_jacobian,
    parameter_shift_jvp,
)


def program():
    x, w, p = Parameter("x"), Parameter("w"), Parameter("p")
    return (
        Circuit(2)
        .ry(x * w, 0)
        .rx(x.sin(), 1)
        .measure(0, 0)
        .if_else(ClassicalCondition((0,), 1), Circuit(2).ry(w * w, 1), Circuit(2).rz(w, 1))
        .amplitude_damping(p, 1)
        .bit_flip(p / 2, 0)
    )


def test_streamed_directional_gradients_match_finite_difference_and_jacobian():
    c = program()
    obs = [PauliString("IZ"), PauliString("IX"), PauliString("ZZ")]
    rows = [{"x": 0.4, "w": 0.8, "p": 0.2}, {"x": 1.1, "w": -0.6, "p": 0.3}]
    names = ("p", "w", "x")
    directions = np.array([[0.1, 0.2, -0.4], [-0.3, 0, 0.8]])
    jac = parameter_shift_jacobian(c, obs, rows, parameters=names, chunk_size=3)
    actual = parameter_shift_jvp(
        c, obs, iter(rows), directions, parameters=iter(names), chunk_size=3
    )
    np.testing.assert_allclose(actual, np.einsum("bmp,bp->bm", jac, directions), atol=1e-12)

    def shifted(sign):
        return [
            {k: row[k] + sign * 1e-5 * directions[b, names.index(k)] for k in names}
            for b, row in enumerate(rows)
        ]

    finite = (
        expectations_batch(c, obs, shifted(1)) - expectations_batch(c, obs, shifted(-1))
    ) / 2e-5
    np.testing.assert_allclose(actual, finite, atol=2e-9)


class Counting:
    name = "direction-counting"
    capabilities = BackendCapabilities(native_expectations=True)

    def __init__(self):
        self.sizes = []

    def expectations_batch(self, circuits, observables, *, options):
        self.sizes.append(len(circuits))
        return expectations_batch(circuits, observables)


def test_zero_and_cancelling_directions_submit_no_jobs_and_other_jobs_are_chunked():
    x, w = Parameter("x"), Parameter("w")
    c = Circuit(1).ry(x - w, 0).rz((x - w) ** 2, 0)
    backend = Counting()
    obs = [PauliString("Z")]
    rows = [{"x": 0.2, "w": 0.3}] * 5
    np.testing.assert_array_equal(
        parameter_shift_jvp(c, obs, rows, np.ones((5, 2)), backend=backend), np.zeros((5, 1))
    )
    assert backend.sizes == []
    directions = np.array([[1, 0], [0, 0], [0, 0], [0, 1], [0, 0]])
    result = parameter_shift_jvp(c, obs, rows, directions, backend=backend, chunk_size=3)
    assert max(backend.sizes) <= 3
    assert sum(backend.sizes) == 8
    np.testing.assert_array_equal(result[[1, 2, 4]], np.zeros((3, 1)))


@pytest.mark.parametrize(
    "tangents", [np.ones((2, 2)), [[np.nan]], [[np.inf]], [[1j]], [[True]], [["1"]]]
)
def test_invalid_directions_never_submit_work(tangents):
    c = Circuit(1).ry(Parameter("x"), 0)
    backend = Counting()
    with pytest.raises(ValidationError, match="Tangents"):
        parameter_shift_jvp(c, [PauliString("Z")], [{"x": 0.3}], tangents, backend=backend)
    assert backend.sizes == []


def test_empty_batches_empty_parameters_and_nonsmooth_zero_direction():
    c = Circuit(1).ry(Parameter("x"), 0)
    assert parameter_shift_jvp(c, [PauliString("Z")], [], np.empty((0, 1))).shape == (0, 1)
    np.testing.assert_array_equal(
        parameter_shift_jvp(c, [PauliString("Z")], [{"x": 0.2}], np.empty((1, 0)), parameters=[]),
        [[0]],
    )
    c = Circuit(1).x(0).amplitude_damping(Parameter("p"), 0)
    np.testing.assert_array_equal(
        parameter_shift_jvp(c, [PauliString("Z")], [{"p": 1}], [[0.0]]), [[0.0]]
    )
    with pytest.raises(ValidationError, match="interior"):
        parameter_shift_jvp(c, [PauliString("Z")], [{"p": 1}], [[1.0]])


def test_torch_external_forward_ad_for_inputs_shared_weights_and_backprop():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    c = program()
    layer = QuantumLayer(c, ["x"], ["p", "w"], backend="numpy", dtype=torch.float64)
    with torch.no_grad():
        layer.weights.copy_(torch.tensor([0.2, 0.8]))
    x = torch.tensor([[[0.4], [0.7]], [[1.1], [-0.2]]], dtype=torch.float64, requires_grad=True)
    dx, dw = torch.full_like(x, 0.3), torch.tensor([-0.1, 0.2], dtype=torch.float64)
    with torch.autograd.forward_ad.dual_level():
        dual_x = torch.autograd.forward_ad.make_dual(x, dx)
        dual_w = torch.autograd.forward_ad.make_dual(layer.weights, dw)
        output = torch.func.functional_call(layer, {"weights": dual_w}, (dual_x,))
        primal, tangent = torch.autograd.forward_ad.unpack_dual(output)
    native = QuantumLayer(c, ["x"], ["p", "w"], dtype=torch.float64)

    def fn(a, b):
        return torch.func.functional_call(native, {"weights": b}, (a,))

    expected = torch.autograd.functional.jvp(fn, (x, layer.weights), (dx, dw))
    torch.testing.assert_close(primal, expected[0])
    torch.testing.assert_close(tangent, expected[1], atol=1e-11, rtol=1e-11)
    layer(x).sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(layer.weights.grad).all()


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_jax_directional_api_jit_vmap_and_existing_reverse_mode(backend):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    layer = QuantumLayer(program(), ["x"], ["p", "w"], backend=backend, gradient_chunk_size=3)
    x = jnp.array([[[0.4], [0.7]], [[1.1], [-0.2]]], dtype=jnp.float32)
    w = jnp.array([0.2, 0.8], dtype=jnp.float32)
    dx, dw = jnp.full_like(x, 0.3), jnp.array([-0.1, 0.2], dtype=jnp.float32)
    native = QuantumLayer(program(), ["x"], ["p", "w"])
    expected = jax.jvp(native, (w, x), (dw, dx))
    actual = jax.jit(layer.jvp)(w, x, dw, dx)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=8e-7)
    mapped = jax.jit(jax.vmap(lambda a, d: layer.jvp(w, a, dw, d)))(x, dx)
    for a, b in zip(mapped, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=8e-7)
    grad = jax.jit(jax.grad(lambda t: layer(t, x).sum()))(w)
    np.testing.assert_allclose(grad, jax.grad(lambda t: native(t, x).sum())(w), atol=2e-6)
    with pytest.raises(ValidationError, match="tangent shapes"):
        layer.jvp(w, x, dw[:1], dx)


def test_external_density_provider_physical_gradient_preparation_is_shared():
    cirq = pytest.importorskip("cirq")
    from openqml import CirqDensityBackend

    c = Circuit(2).h(0).cry(Parameter("x") ** 2, 0, 1).amplitude_damping(0.2, 1)
    backend = CirqDensityBackend(noise=cirq.depolarize(0.04))
    obs = [PauliString("IX"), PauliString("IZ")]
    rows = [{"x": 0.4}, {"x": -0.7}]
    direction = np.array([[0.2], [-0.3]])
    actual = parameter_shift_jvp(c, obs, rows, direction, backend=backend)
    step = 1e-5
    plus = [{"x": r["x"] + step * direction[i, 0]} for i, r in enumerate(rows)]
    minus = [{"x": r["x"] - step * direction[i, 0]} for i, r in enumerate(rows)]
    expected = (
        expectations_batch(c, obs, plus, backend=backend)
        - expectations_batch(c, obs, minus, backend=backend)
    ) / (2 * step)
    np.testing.assert_allclose(actual, expected, atol=2e-10)
