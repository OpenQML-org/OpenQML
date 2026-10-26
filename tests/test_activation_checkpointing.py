"""Recomputation must reduce retained tensors while preserving native autodiff."""

import numpy as np
import pytest

from openqml import Circuit, Parameter, PauliString, ValidationError, statevector


def circuit():
    x, w = Parameter("x"), Parameter("w")
    return Circuit(2).ry(x * w, 0).barrier().crx(w.sin(), 0, 1).rz(x + w, 1)


def test_torch_recomputation_gradients_hessian_and_saved_tensor_memory():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    ordinary = QuantumLayer.data_reuploading(4, 6, dtype=torch.float64)
    recompute = QuantumLayer.from_checkpoint(ordinary.checkpoint(), checkpoint_interval=8)
    x = torch.linspace(0.1, 0.9, 12, dtype=torch.float64).reshape(3, 4).requires_grad_()

    def evaluate(layer):
        saved = []

        def pack(value):
            if value.is_complex():
                saved.append(value.numel() * value.element_size())
            return value

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            output = layer(x)
        gradients = torch.autograd.grad(output.square().sum(), (layer.weights, x))
        return output, gradients, sum(saved)

    left, left_grad, ordinary_bytes = evaluate(ordinary)
    right, right_grad, recompute_bytes = evaluate(recompute)
    torch.testing.assert_close(left, right)
    for a, b in zip(left_grad, right_grad, strict=True):
        torch.testing.assert_close(a, b)
    assert recompute_bytes < ordinary_bytes * 0.6
    assert recompute.to_spec() == ordinary.to_spec()

    layer = QuantumLayer(
        circuit(), ["x"], ["w"], dtype=torch.float64, checkpoint_interval=2, batch_chunk_size=2
    )
    w = torch.tensor([[[0.2]], [[0.4]]], dtype=torch.float64, requires_grad=True)
    inputs = torch.tensor([[0.3], [0.7], [0.9]], dtype=torch.float64, requires_grad=True)
    base = QuantumLayer(circuit(), ["x"], ["w"], dtype=torch.float64)
    for candidate in (layer, base):
        value = candidate.functional(w, inputs).square().sum()
        first = torch.autograd.grad(value, w, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), w)[0]
        if candidate is layer:
            actual = second
        else:
            torch.testing.assert_close(actual, second)
    compiled = torch.compile(layer.functional, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(w, inputs), base.functional(w, inputs))


def test_torch_recomputation_preserves_forward_ad_and_no_grad_inference():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    simple = Circuit(1).ry(Parameter("x") + Parameter("w"), 0)
    layer = QuantumLayer(simple, ["x"], ["w"], checkpoint_interval=1)
    w, x = torch.tensor([0.3]), torch.tensor([[0.2], [0.5]])
    with torch.autograd.forward_ad.dual_level():
        result = layer.functional(torch.autograd.forward_ad.make_dual(w, torch.ones_like(w)), x)
        primal, tangent = torch.autograd.forward_ad.unpack_dual(result)
    torch.testing.assert_close(primal, torch.cos(w + x))
    torch.testing.assert_close(tangent, -torch.sin(w + x))
    with torch.no_grad():
        torch.testing.assert_close(layer.functional(w, x), torch.cos(w + x))


def test_jax_recomputation_jit_vmap_gradients_hessian_and_flax():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    base = QuantumLayer(circuit(), ["x"], ["w"])
    layer = QuantumLayer.from_spec(base.to_spec(), checkpoint_interval=2, batch_chunk_size=2)
    w, x = jnp.array([[[0.2]], [[0.4]]]), jnp.array([[0.3], [0.7], [0.9]])
    np.testing.assert_allclose(jax.jit(layer)(w, x), base(w, x), atol=3e-7)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(layer, in_axes=(0, None)))(w, x), base(w, x), atol=3e-7
    )

    def loss(candidate):
        return lambda a, b: jnp.square(candidate(a, b)).sum()

    for transform in (lambda f: jax.grad(f, argnums=(0, 1)), jax.hessian):
        actual = jax.jit(transform(loss(layer)))(w, x)
        expected = transform(loss(base))(w, x)
        for a, b in zip(
            jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True
        ):
            np.testing.assert_allclose(a, b, atol=3e-6)
    pytest.importorskip("flax")
    from openqml.interfaces.flax import QuantumLayer as FlaxLayer

    module = FlaxLayer.from_spec(base.to_spec(), checkpoint_interval=2, batch_chunk_size=2)
    params = module.init(jax.random.key(1), x)
    actual = jax.jit(jax.grad(lambda p: module.apply(p, x).sum()))(params)
    expected = jax.grad(lambda p: base(p["params"]["weights"], x).sum())(params)
    for a, b in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True
    ):
        np.testing.assert_allclose(a, b, atol=1e-6)


def test_checkpointed_statevector_keeps_symbolic_global_phase_and_complex_gradients():
    torch = pytest.importorskip("torch")
    c = circuit().with_global_phase(Parameter("w") / 3)
    x, w = (
        torch.tensor(0.3, dtype=torch.float64),
        torch.tensor(0.7, dtype=torch.float64, requires_grad=True),
    )
    left = statevector(c, {"x": x, "w": w}, backend="torch", checkpoint_interval=2)
    right = statevector(c, {"x": x, "w": w}, backend="torch")
    torch.testing.assert_close(left, right)
    torch.testing.assert_close(
        torch.autograd.grad(left.real.sum(), w)[0], torch.autograd.grad(right.real.sum(), w)[0]
    )


@pytest.mark.parametrize("interface", ["torch", "jax"])
@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_recomputation_rejects_invalid_intervals(interface, interval):
    pytest.importorskip(interface)
    from importlib import import_module

    layer = import_module(f"openqml.interfaces.{interface}").QuantumLayer
    with pytest.raises(ValidationError, match="checkpoint_interval"):
        layer.hardware_efficient(1, checkpoint_interval=interval)


@pytest.mark.parametrize("backend", ["numpy", "qiskit", "pennylane", "cirq"])
def test_recomputation_rejects_external_execution_before_loading_sdks(backend):
    with pytest.raises(ValidationError, match="native"):
        statevector(Circuit(1), backend=backend, checkpoint_interval=2)


def test_recomputation_rejects_mixed_state_circuits_in_all_entry_points():
    from openqml import expectation, expectations

    c = Circuit(1).amplitude_damping(0.2, 0)
    for function, args in (
        (statevector, (c,)),
        (expectation, (c, PauliString("Z"))),
        (expectations, (c, [PauliString("Z")])),
    ):
        with pytest.raises(ValidationError, match="unitary"):
            function(*args, backend="torch", checkpoint_interval=2)
