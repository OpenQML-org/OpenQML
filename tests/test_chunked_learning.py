import numpy as np
import pytest

from openqml import (
    Circuit,
    Parameter,
    PauliString,
    ResourceError,
    ValidationError,
    convert,
    data_reuploading,
    expectation,
    parameter_shift,
    statevector,
)


@pytest.mark.parametrize("topology,edges", [("linear", 2), ("ring", 3), ("full", 3), ("none", 0)])
def test_reuploading_parameter_ownership_and_topology(topology, edges):
    spec = data_reuploading(3, 2, entanglement=topology)
    assert spec.input_parameters == ("x_0", "x_1", "x_2")
    assert len(spec.weight_parameters) == 18
    assert sum(op.name == "cx" for op in spec.circuit.operations) == 2 * edges
    assert (
        sum(
            op.name == "ry" and set(op.params[0].parameters) == {"x_0"}
            for op in spec.circuit.operations
            if op.params
        )
        == 2
    )
    assert len(data_reuploading(1, 2, entanglement="ring").circuit.operations) == 8


def test_reused_input_gradient_and_qasm_roundtrip():
    spec = data_reuploading(2, 3)
    values = {name: 0.17 + 0.01 * i for i, name in enumerate(spec.circuit.parameters)}
    observable = PauliString("ZI")
    gradient = parameter_shift(spec.circuit, observable, values, parameters=("x_0",))
    plus, minus = dict(values), dict(values)
    plus["x_0"] += 1e-6
    minus["x_0"] -= 1e-6
    expected = (
        expectation(spec.circuit, observable, plus) - expectation(spec.circuit, observable, minus)
    ) / 2e-6
    assert gradient["x_0"] == pytest.approx(expected, abs=2e-9)
    exported = convert(spec.circuit, target="openqasm3")
    restored = convert(exported.value, source="openqasm3", target="ir").value
    np.testing.assert_allclose(statevector(restored, values), statevector(spec.circuit, values))


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_torch_chunking_preserves_outputs_and_shared_gradients(backend):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    torch.manual_seed(9)
    base = QuantumLayer.data_reuploading(2, 2, dtype=torch.float64, backend=backend)
    chunked = QuantumLayer.from_checkpoint(base.checkpoint(), backend=backend, batch_chunk_size=3)
    inputs = torch.randn(2, 4, 2, dtype=torch.float64, requires_grad=True)
    expected, actual = base(inputs), chunked(inputs)
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected.square().sum(), (inputs, base.weights))
    actual_grad = torch.autograd.grad(actual.square().sum(), (inputs, chunked.weights))
    for left, right in zip(expected_grad, actual_grad, strict=True):
        torch.testing.assert_close(left, right)
    assert chunked.to_spec() == base.to_spec()


def test_torch_chunked_hessian_optimizer_and_memory_budget():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    circuit = Circuit(1).ry(Parameter("x"), 0).ry(Parameter("w"), 0)
    layer = QuantumLayer(circuit, ("x",), ("w",), dtype=torch.float64, batch_chunk_size=2)
    inputs = torch.linspace(0.2, 0.8, 5, dtype=torch.float64).reshape(5, 1)
    inputs.requires_grad_()
    gradient = torch.autograd.grad(layer(inputs).sum(), inputs, create_graph=True)[0]
    hessian = torch.autograd.grad(gradient.sum(), inputs)[0]
    torch.testing.assert_close(hessian, -torch.cos(inputs + layer.weights))
    model = torch.nn.Sequential(torch.nn.Linear(2, 1, dtype=torch.float64), layer)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    before = [p.detach().clone() for p in model.parameters()]
    model(torch.ones(5, 2, dtype=torch.float64)).sum().backward()
    optimizer.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    unchunked = QuantumLayer.hardware_efficient(4, 1, max_memory_mb=0.01)
    small = QuantumLayer.from_checkpoint(
        unchunked.checkpoint(), batch_chunk_size=2, max_memory_mb=0.01
    )
    batch = torch.zeros(64, 4)
    with pytest.raises(ResourceError):
        unchunked(batch)
    assert small(batch).shape == (64, 4)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_chunk_sizes(backend, size):
    pytest.importorskip(backend)
    if backend == "torch":
        from openqml.interfaces.torch import QuantumLayer
    else:
        from openqml.interfaces.jax import QuantumLayer
    with pytest.raises(ValidationError):
        QuantumLayer.hardware_efficient(1, batch_chunk_size=size)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_jax_chunking_jit_vmap_and_gradients(backend):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    base = QuantumLayer.data_reuploading(2, 2, backend=backend)
    chunked = QuantumLayer.from_spec(base.to_spec(), backend=backend, batch_chunk_size=3)
    weights = base.init(jax.random.key(2))
    inputs = jnp.linspace(0.1, 0.7, 16).reshape(2, 4, 2)
    np.testing.assert_allclose(jax.jit(chunked)(weights, inputs), base(weights, inputs), atol=2e-6)
    expected = jax.jit(jax.grad(lambda w, x: jnp.square(base(w, x)).sum(), argnums=(0, 1)))(
        weights, inputs
    )
    actual = jax.jit(jax.grad(lambda w, x: jnp.square(chunked(w, x)).sum(), argnums=(0, 1)))(
        weights, inputs
    )
    for a, b in zip(expected, actual, strict=True):
        np.testing.assert_allclose(a, b, atol=4e-6)
    np.testing.assert_allclose(
        jax.vmap(lambda x: chunked(weights, x))(inputs), base(weights, inputs), atol=2e-6
    )


def test_jax_chunked_higher_derivatives_and_flax_unified_training():
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import flax.linen as nn
    import jax.numpy as jnp
    import optax

    from openqml.interfaces.flax import QuantumLayer as FlaxLayer
    from openqml.interfaces.jax import QuantumLayer

    circuit = Circuit(1).ry(Parameter("x"), 0).ry(Parameter("w"), 0)
    layer = QuantumLayer(circuit, ("x",), ("w",), batch_chunk_size=2)
    weights, inputs = jnp.array([0.3]), jnp.linspace(0.1, 0.7, 5).reshape(5, 1)

    def loss(w):
        return layer(w, inputs).sum()

    actual = jax.jit(jax.hessian(loss))(weights)
    np.testing.assert_allclose(actual[0, 0], -jnp.cos(inputs + weights).sum(), atol=2e-6)
    _, tangent = jax.jit(lambda w: jax.jvp(loss, (w,), (jnp.ones_like(w),)))(weights)
    np.testing.assert_allclose(tangent, -jnp.sin(inputs + weights).sum(), atol=2e-6)

    class Model(nn.Module):
        @nn.compact
        def __call__(self, x):
            encoded = nn.Dense(2)(x)
            quantum = FlaxLayer.data_reuploading(2, 2, batch_chunk_size=2)(encoded)
            return nn.Dense(1)(quantum)

    model = Model()
    batch = jnp.ones((5, 3))
    params = model.init(jax.random.key(1), batch)["params"]

    def loss_fn(p):
        return jnp.square(model.apply({"params": p}, batch)).mean()

    gradients = jax.jit(jax.grad(loss_fn))(params)
    optimizer = optax.adam(0.01)
    updates, _ = optimizer.update(gradients, optimizer.init(params), params)
    updated = optax.apply_updates(params, updates)
    assert float(loss_fn(updated)) < float(loss_fn(params))
    assert all(np.isfinite(g).all() for g in jax.tree_util.tree_leaves(gradients))
