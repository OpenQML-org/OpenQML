import numpy as np
import pytest

import openqml as oq


def test_torch_vectorized_nonlinear_batch_input_weight_gradients_and_autocast():
    torch = pytest.importorskip("torch")
    from openqml import Circuit, Parameter
    from openqml.interfaces.torch import QuantumLayer

    c = Circuit(2).h(0).ry(Parameter("x").sin() * Parameter("w"), 1).crx(Parameter("x") ** 2, 0, 1)
    layer = QuantumLayer(c, ("x",), ("w",), dtype=torch.float64)
    inputs = torch.tensor([[0.13], [0.47], [-0.31]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([0.63], dtype=torch.float64, requires_grad=True)

    def f(x, w):
        return torch.func.functional_call(layer, {"weights": w}, (x,))

    assert torch.autograd.gradcheck(f, (inputs, weights), atol=1e-5)
    assert torch.autograd.gradgradcheck(f, (inputs, weights), atol=1e-5)
    torch.testing.assert_close(layer(inputs), torch.stack([layer(row) for row in inputs]))
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 2), QuantumLayer.hardware_efficient(2, 1), torch.nn.Linear(2, 1)
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(torch.ones(4, 3)).float().square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_native_pennylane_torch_templates_train_with_classical_parameters():
    torch = pytest.importorskip("torch")
    qml = pytest.importorskip("pennylane")
    from openqml.interfaces.torch import NativePennyLaneLayer

    torch.manual_seed(17)
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="torch")
    def node(inputs, weights):
        qml.AngleEmbedding(inputs, wires=[0, 1])
        qml.StronglyEntanglingLayers(weights, wires=[0, 1])
        return [qml.expval(qml.Z(i)) for i in range(2)]

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        NativePennyLaneLayer(node, {"weights": (2, 2, 3)}),
        torch.nn.Linear(2, 1),
    )
    x = torch.rand(12, 2)
    y = x[:, :1] - 0.3 * x[:, 1:]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        loss = (model(x) - y).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.3
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_torch_imports_native_qiskit_and_external_aer_gradcheck():
    torch = pytest.importorskip("torch")
    qiskit = pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from qiskit.circuit import ParameterVector

    from openqml.interfaces.torch import QuantumLayer

    x = ParameterVector("x", 1)
    w = ParameterVector("w", 1)
    qc = qiskit.QuantumCircuit(2)
    qc.h(0)
    qc.ry(x[0].sin() * w[0], 1)
    qc.cx(0, 1)
    layer = QuantumLayer.from_program(qc, x, w, dtype=torch.float64, backend="aer")
    inputs = torch.tensor([[0.23], [0.41]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([0.63], dtype=torch.float64, requires_grad=True)

    def f(x, w):
        return torch.func.functional_call(layer, {"weights": w}, (x,))

    assert torch.autograd.gradcheck(f, (inputs, weights), atol=1e-5)
    pure = QuantumLayer.from_program(qc, x, w, dtype=torch.float64)
    pure.load_state_dict(layer.state_dict())
    torch.testing.assert_close(pure(inputs), layer(inputs), atol=1e-12, rtol=1e-12)


def test_jax_vectorized_nonlinear_and_aer_jit_vjp():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("qiskit_aer")
    from openqml import Circuit, Parameter
    from openqml.interfaces.jax import QuantumLayer

    c = Circuit(2).h(0).ry(Parameter("x").sin() * Parameter("w"), 1).cp(Parameter("x") ** 2, 0, 1)
    native = QuantumLayer(c, ("x",), ("w",))
    external = QuantumLayer(c, ("x",), ("w",), backend="aer")
    x = jnp.array([[0.23], [-0.47]], dtype=jnp.float32)
    w = jnp.array([0.63], dtype=jnp.float32)

    def loss(layer, w, x):
        return jnp.mean(layer(w, x) ** 2)

    vn, gn = jax.jit(jax.value_and_grad(lambda w, x: loss(native, w, x), argnums=(0, 1)))(w, x)
    ve, ge = jax.jit(jax.value_and_grad(lambda w, x: loss(external, w, x), argnums=(0, 1)))(w, x)
    np.testing.assert_allclose(vn, ve, atol=2e-6)
    for a, b in zip(gn, ge, strict=True):
        np.testing.assert_allclose(a, b, atol=2e-6)


def test_flax_optax_hybrid_model_one_optimizer():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    nn = pytest.importorskip("flax.linen")
    optax = pytest.importorskip("optax")
    from openqml import hardware_efficient
    from openqml.interfaces.flax import QuantumLayer

    spec = hardware_efficient(2, 1)

    class Hybrid(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(2)(x)
            x = QuantumLayer(spec.circuit, spec.input_parameters, spec.weight_parameters)(x)
            return nn.Dense(1)(x)

    x = jax.random.uniform(jax.random.key(7), (16, 2))
    target = x[:, :1] - 0.4 * x[:, 1:]
    model = Hybrid()
    params = model.init(jax.random.key(9), x)
    tx = optax.adam(0.04)
    state = tx.init(params)

    @jax.jit
    def step(params, state):
        loss, grads = jax.value_and_grad(lambda p: jnp.mean((model.apply(p, x) - target) ** 2))(
            params
        )
        updates, state = tx.update(grads, state, params)
        return optax.apply_updates(params, updates), state, loss, grads

    losses = []
    for _ in range(45):
        params, state, loss, grads = step(params, state)
        losses.append(float(loss))
    assert losses[-1] < losses[0] * 0.3
    assert all(np.isfinite(leaf).all() for leaf in jax.tree_util.tree_leaves(grads))
    assert np.linalg.norm(np.asarray(grads["params"]["QuantumLayer_0"]["weights"])) > 0


def test_native_pennylane_jax_qnode_graph_and_flax_module():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    qml = pytest.importorskip("pennylane")
    pytest.importorskip("flax")
    from openqml.interfaces.flax import NativePennyLaneLayer as LinenQNode
    from openqml.interfaces.jax import NativePennyLaneLayer

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="jax")
    def node(inputs, weights):
        qml.AngleEmbedding(inputs, wires=[0, 1])
        qml.StronglyEntanglingLayers(weights, wires=[0, 1])
        return [qml.expval(qml.Z(i)) for i in range(2)]

    with oq.jax.enable_x64():
        x = jnp.array([[0.23, -0.41], [0.37, 0.61]], dtype=jnp.float64)
        layer = NativePennyLaneLayer(node, {"weights": (2, 2, 3)})
        weights = layer.init(jax.random.key(3), dtype=jnp.float64)
        value, gradient = jax.jit(jax.value_and_grad(lambda w: jnp.mean(layer(w, x) ** 2)))(weights)
        assert np.isfinite(value) and np.linalg.norm(np.asarray(gradient["weights"])) > 0
        module = LinenQNode(node, (2, 2, 3), dtype=jnp.float64)
        params = module.init(jax.random.key(4), x)
        out = module.apply(params, x)
        assert out.shape == (2, 2) and np.isfinite(out).all()


def test_native_pennylane_noisy_device_and_mixed_readouts_keep_gradients():
    torch = pytest.importorskip("torch")
    qml = pytest.importorskip("pennylane")
    from openqml.interfaces.torch import NativePennyLaneLayer

    device = qml.device("default.mixed", wires=2)

    @qml.qnode(device, interface="torch", diff_method="backprop")
    def node(inputs, weights):
        qml.AngleEmbedding(inputs, wires=[0, 1])
        qml.StronglyEntanglingLayers(weights, wires=[0, 1])
        qml.DepolarizingChannel(0.13, wires=0)
        return [qml.expval(qml.Z(0)), qml.probs(wires=[0, 1])]

    layer = NativePennyLaneLayer(node, {"weights": (1, 2, 3)})
    x = torch.tensor([[0.17, -0.43], [0.61, 0.23]], requires_grad=True)
    result = layer(x)
    assert result.shape == (2, 5)
    result.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


def test_transformer_quantum_residual_adapter_supports_token_dimensions():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumAdapter

    encoder = torch.nn.TransformerEncoderLayer(d_model=8, nhead=2, dropout=0, batch_first=True)
    adapter = QuantumAdapter(8, num_qubits=2, depth=1)
    x = torch.randn(2, 3, 8, requires_grad=True)
    result = adapter(encoder(x))
    assert result.shape == x.shape
    result.square().mean().backward()
    assert adapter.quantum.weights.grad is not None
    assert torch.linalg.vector_norm(adapter.quantum.weights.grad) > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_flax_quantum_adapter_supports_token_dimensions_and_gradients():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    from openqml.interfaces.flax import QuantumAdapter

    module = QuantumAdapter(num_qubits=2, depth=1)
    x = jax.random.normal(jax.random.key(7), (2, 3, 8))
    params = module.init(jax.random.key(9), x)
    result = module.apply(params, x)
    assert result.shape == x.shape
    gradient = jax.jit(jax.grad(lambda p: jnp.mean(module.apply(p, x) ** 2)))(params)
    assert np.linalg.norm(np.asarray(gradient["params"]["quantum"]["weights"])) > 0


def test_torch_fullgraph_aot_compilation_preserves_forward_and_gradients():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    torch.manual_seed(41)
    layer = QuantumLayer.hardware_efficient(2, 1, dtype=torch.float64)
    x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    expected = layer(x)
    expected_grads = torch.autograd.grad(expected.square().mean(), (x, layer.weights))
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
    actual = compiled(x)
    actual_grads = torch.autograd.grad(actual.square().mean(), (x, layer.weights))
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    for a, b in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_quantum_image_convolution_chunking_and_classical_optimizer():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumConv2d

    torch.manual_seed(43)
    layer = QuantumConv2d(1, 3, num_qubits=2, depth=1, patch_chunk_size=3, dtype=torch.float64)
    x = torch.randn(2, 1, 6, 6, dtype=torch.float64, requires_grad=True)
    actual = layer(x)
    assert actual.shape == (2, 3, 3, 3)
    actual.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    assert torch.linalg.vector_norm(layer.quantum.weights.grad) > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    layer.patch_chunk_size = 100
    torch.testing.assert_close(layer(x), actual, atol=1e-12, rtol=1e-12)


def test_external_vjp_skips_frozen_parameters():
    torch = pytest.importorskip("torch")
    from openqml import BackendCapabilities, Circuit, Parameter, expectations_batch
    from openqml.interfaces.torch import QuantumLayer

    class Engine:
        name = "frozen-parameters"
        capabilities = BackendCapabilities(native_expectations=True)

        def __init__(self):
            self.jobs = 0

        def expectations_batch(self, circuits, observables, *, options):
            self.jobs += len(circuits)
            return expectations_batch(circuits, observables)

    engine = Engine()
    c = Circuit(1).ry(Parameter("x"), 0).rz(Parameter("w"), 0)
    layer = QuantumLayer(c, ("x",), ("w",), backend=engine, dtype=torch.float64)
    x = torch.tensor([[0.41], [-0.63]], dtype=torch.float64)
    layer(x).sum().backward()
    assert engine.jobs == 6  # two forwards + two shifts per row for w only
    layer.weights.requires_grad_(False)
    x.requires_grad_(True)
    engine.jobs = 0
    layer(x).sum().backward()
    assert engine.jobs == 6
    torch.testing.assert_close(x.grad[:, 0], -torch.sin(x[:, 0]), atol=1e-12, rtol=1e-12)
