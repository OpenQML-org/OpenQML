"""Independent circuit/geometry references and ordinary ML training contracts."""

import math

import numpy as np
import pytest

import openqml as oq


def dense_readout(graph, coordinates):
    n, m = graph.num_qubits, len(graph.edges)
    bindings = {f"alpha_{q}": coordinates[q] for q in range(n)}
    bindings.update({f"omega_{e}": coordinates[n + e] for e in range(m)})
    observables = tuple(
        oq.PauliString("I" * q + p + "I" * (n - q - 1)) for p in "XYZ" for q in range(n)
    )
    xyz = oq.expectations(graph.circuit(), observables, bindings).reshape(3, n)
    polar, azimuth = coordinates[n + m : 2 * n + m], coordinates[2 * n + m :]
    return (
        np.sin(polar) * np.cos(azimuth) * xyz[0]
        + np.sin(polar) * np.sin(azimuth) * xyz[1]
        + np.cos(polar) * xyz[2]
    )


@pytest.mark.parametrize("n", range(1, 7))
@pytest.mark.parametrize("topology", ("default", "star", "empty"))
def test_closed_form_matches_independent_state_simulation(n, topology):
    edges = (
        None
        if topology == "default"
        else tuple((0, q) for q in range(1, n))
        if topology == "star"
        else ()
    )
    graph = oq.IQPGraph(n, edges)
    rng = np.random.default_rng(10 + n)
    for _ in range(3):
        value = rng.uniform(-math.pi, math.pi, graph.num_coordinates)
        np.testing.assert_allclose(graph.evaluate(value), dense_readout(graph, value), atol=3e-14)
    assert sum(len(nodes) * degree for nodes, _, degree in graph._groups) == 2 * len(graph.edges)


def test_iqp_prepared_state_metric_is_quarter_identity():
    graph = oq.IQPGraph(3)
    circuit = graph.circuit()
    values = {name: 0.13 * (i + 1) for i, name in enumerate(circuit.parameters)}
    metric = oq.quantum_metric(circuit).matrix(values)
    np.testing.assert_allclose(metric, np.eye(len(values)) / 4, atol=1e-14)


@pytest.mark.parametrize("edges", (((0, 0),), ((0, 1), (1, 0)), ((0, 3),), ((True, 1),)))
def test_invalid_generators_are_rejected_before_using_closed_form(edges):
    with pytest.raises(oq.ValidationError):
        oq.IQPGraph(3, edges)


def test_graph_resource_and_reference_variance_contracts():
    with pytest.raises(oq.ResourceError):
        oq.IQPGraph(10**12)
    graph = oq.IQPGraph(16)
    profile = graph.to_dict()
    assert profile["max_degree"] == 4
    np.testing.assert_allclose(profile["reference_gradient_variance"], 1 / 48)
    assert oq.IQPGraph(1, ()).evaluate(np.array([0.2, 0.0, 0.0]))[0] == pytest.approx(math.cos(0.2))


def test_torch_native_products_compile_and_mixed_precision():
    torch = pytest.importorskip("torch")
    layer = oq.torch.IQPQuantumLayer(3, dtype=torch.float64)
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer.functional, (layer.weights, x))
    assert torch.autograd.gradgradcheck(layer.functional, (layer.weights, x))
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), layer(x))
    primal, tangent = torch.func.jvp(layer, (x,), (torch.ones_like(x),))
    jac = torch.func.jacrev(layer)(x)
    torch.testing.assert_close(tangent, torch.einsum("abij,ij->ab", jac, torch.ones_like(x)))
    torch.testing.assert_close(torch.vmap(layer)(x), primal)
    single = oq.torch.IQPQuantumLayer(3)
    half = x.detach().float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = single(half)
        output.square().mean().backward()
    assert torch.isfinite(single.weights.grad).all()


def test_zero_cosine_incidence_products_retain_second_derivatives():
    torch = pytest.importorskip("torch")
    graph = oq.IQPGraph(3, ((0, 1), (0, 2)))
    coordinates = torch.tensor(
        np.random.default_rng(7).normal(size=graph.num_coordinates), requires_grad=True
    )
    with torch.no_grad():
        coordinates[3:5] = math.pi / 2

    def fn(v):
        return graph.evaluate(v, backend="torch")[0]

    hessian = torch.func.hessian(fn)(coordinates)
    assert torch.isfinite(hessian).all() and abs(hessian[3, 4].item()) > 0.01
    assert torch.autograd.gradgradcheck(fn, (coordinates,))


def test_jax_iqp_jit_vmap_higher_derivatives_match_torch():
    jax = pytest.importorskip("jax")
    torch = pytest.importorskip("torch")
    with oq.jax.enable_x64():
        graph = oq.IQPGraph(3)
        value = np.random.default_rng(3).normal(size=graph.num_coordinates)
        tx = torch.tensor(value, dtype=torch.float64)
        jx = oq.jax.numpy.asarray(value)

        def tf(x):
            return graph.evaluate(x, backend="torch").square().sum()

        def jf(x):
            return (graph.evaluate(x, backend="jax") ** 2).sum()

        np.testing.assert_allclose(
            jax.jit(jax.grad(jf))(jx), torch.func.grad(tf)(tx).numpy(), atol=1e-13
        )
        np.testing.assert_allclose(
            jax.jit(jax.hessian(jf))(jx), torch.func.hessian(tf)(tx).numpy(), atol=1e-13
        )
        values = oq.jax.numpy.stack((jx, jx * 0.4))
        np.testing.assert_allclose(
            jax.jit(jax.vmap(jf))(values), np.array([jf(v) for v in values]), atol=1e-13
        )
        layer = oq.jax.IQPQuantumLayer(3)
        weights = layer.init(jax.random.key(3), dtype=oq.jax.numpy.float64)
        assert jax.jit(layer)(weights, oq.jax.numpy.ones((2, 3))).shape == (2, 3)


@pytest.mark.parametrize("qubits", (64, 1024))
def test_wide_iqp_training_never_allocates_a_statevector(qubits, monkeypatch):
    torch = pytest.importorskip("torch")
    import openqml.simulator as simulator

    monkeypatch.setattr(
        simulator, "_initial_state", lambda *a, **k: pytest.fail("Global state allocated")
    )
    layer = oq.torch.IQPQuantumLayer(qubits)
    inputs = torch.randn(2, qubits, requires_grad=True)
    output = layer(inputs)
    output.square().mean().backward()
    assert output.shape == inputs.shape and torch.isfinite(inputs.grad).all()
    assert torch.isfinite(layer.weights.grad).all() and layer.weights.grad.norm() > 0


def test_hypernetwork_joint_training_chunks_state_dict_and_frozen_backbone():
    torch = pytest.importorskip("torch")
    torch.manual_seed(34)
    adapter = oq.torch.HyperQuantumAdapter(
        8, num_qubits=16, rank=3, batch_chunk_size=3, dtype=torch.float64
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 8, dtype=torch.float64),
        adapter,
        torch.nn.Linear(8, 2, dtype=torch.float64),
    )
    inputs = torch.randn(12, 5, dtype=torch.float64)
    target = torch.stack((inputs[:, 0] * 0.5, inputs[:, 1] - inputs[:, 2]), dim=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    first = float((model(inputs) - target).square().mean().detach())
    for _ in range(14):
        optimizer.zero_grad()
        loss = (model(inputs) - target).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert float(loss.detach()) < first * 0.5
    assert all(not torch.equal(before[k], p) for k, p in model.named_parameters())
    clone = oq.torch.HyperQuantumAdapter(
        8, num_qubits=16, rank=3, batch_chunk_size=None, dtype=torch.float64
    )
    clone.load_state_dict(adapter.state_dict())
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(clone(x), adapter(x), atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(
        torch.func.grad(lambda z: clone(z).square().sum())(x),
        torch.func.grad(lambda z: adapter(z).square().sum())(x),
    )
    assert torch.compile(clone, backend="eager", fullgraph=True)(x).shape == x.shape
    model[0].requires_grad_(False)
    frozen = model[0].weight.detach().clone()
    optimizer.zero_grad()
    model(inputs).square().mean().backward()
    optimizer.step()
    torch.testing.assert_close(model[0].weight, frozen)


def test_native_pullback_and_cg_match_dense_encoder_jacobian():
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    model = oq.torch.HyperQuantumAdapter(2, num_qubits=2, rank=1, dtype=torch.float64)
    inputs = torch.tensor([[0.4, -0.2], [0.1, 0.3]], dtype=torch.float64)
    parameters = dict(model.encoder.named_parameters())
    keys = tuple(parameters)

    def fn(p):
        return torch.func.functional_call(model.encoder, p, (inputs,))[1]

    jacobian = torch.func.jacrev(fn)(parameters)
    matrix = torch.cat([jacobian[k].reshape(-1, parameters[k].numel()) for k in keys], dim=1)
    metric = matrix.T @ matrix / (4 * len(inputs))
    vector = {k: torch.randn_like(p) for k, p in parameters.items()}
    flat = torch.cat([vector[k].flatten() for k in keys])
    actual = model.metric_product(inputs, vector)
    with torch.no_grad():
        without_tape = model.metric_product(inputs, vector)
    for key in keys:
        torch.testing.assert_close(actual[key], without_tape[key])
    torch.testing.assert_close(torch.cat([actual[k].flatten() for k in keys]), metric @ flat)
    result = model.natural_gradient(inputs, vector, damping=0.1, tolerance=1e-11, max_iterations=16)
    expected = torch.linalg.solve(metric + 0.1 * torch.eye(len(flat), dtype=metric.dtype), flat)
    torch.testing.assert_close(
        torch.cat([result.direction[k].flatten() for k in keys]), expected, atol=1e-10, rtol=1e-10
    )
    assert bool(result.converged) and result.metric_products == int(result.iterations) + 1
    assert result.metric_products < 17
    zero = {k: torch.zeros_like(p) for k, p in parameters.items()}
    empty = model.natural_gradient(inputs, zero, max_iterations=2)
    assert bool(empty.converged) and float(empty.iterations) == 0
    # Metric derivatives retain a native graph for meta-learning/Hessian products.
    objective = sum(value.square().sum() for value in actual.values())
    assert all(
        torch.isfinite(v).all() for v in torch.autograd.grad(objective, tuple(parameters.values()))
    )


def test_jax_hypernetwork_optax_metric_solve_and_flax_modules():
    jax = pytest.importorskip("jax")
    optax = pytest.importorskip("optax")
    pytest.importorskip("flax")
    with oq.jax.enable_x64():
        jnp = oq.jax.numpy
        layer = oq.jax.HyperQuantumAdapter(3, num_qubits=8, rank=2, batch_chunk_size=2)
        params = layer.init(jax.random.key(41), dtype=jnp.float64)
        inputs = jax.random.normal(jax.random.key(5), (2, 3, 3), dtype=jnp.float64)
        target = 0.8 * inputs
        optimizer = optax.adam(0.02)
        state = optimizer.init(params)

        def loss(p):
            return jnp.square(layer(p, inputs) - target).mean()

        initial = float(loss(params))

        @jax.jit
        def step(p, state):
            gradients = jax.grad(loss)(p)
            updates, state = optimizer.update(gradients, state, p)
            return optax.apply_updates(p, updates), state

        for _ in range(12):
            params, state = step(params, state)
        assert float(loss(params)) < initial
        gradient = jax.grad(loss)(params)["encoder"]
        result = jax.jit(
            lambda p, g: layer.natural_gradient(
                p, inputs, g, damping=0.1, tolerance=1e-10, max_iterations=64
            )
        )(params, gradient)
        product = layer.metric_product(params, inputs, result.direction)
        residual = math.sqrt(
            sum(
                float(jnp.square(product[k] + 0.1 * result.direction[k] - gradient[k]).sum())
                for k in gradient
            )
        )
        assert residual == pytest.approx(float(result.residual_norm), abs=1e-12)
        assert bool(result.converged)
        module = oq.flax.HyperQuantumAdapter(3, num_qubits=8, rank=2, dtype=jnp.float64)
        variables = module.init(jax.random.key(2), inputs)
        gradients = jax.jit(jax.grad(lambda v: jnp.square(module.apply(v, inputs)).mean()))(
            variables
        )
        assert all(bool(jnp.isfinite(v).all()) for v in jax.tree_util.tree_leaves(gradients))
        quantum = oq.flax.IQPQuantumLayer(3, dtype=jnp.float64)
        weights = quantum.init(jax.random.key(1), inputs)
        assert quantum.apply(weights, inputs).shape == inputs.shape


@pytest.mark.parametrize("framework", ("torch", "jax", "flax"))
def test_identity_blocks_start_at_encoding_but_entanglers_can_learn(framework):
    pytest.importorskip("torch" if framework == "torch" else "jax")
    if framework == "flax":
        pytest.importorskip("flax")
    recipe = oq.identity_block_ansatz(2, depth=2)
    if framework == "torch":
        torch = oq.torch
        layer = torch.QuantumLayer.identity_blocks(2, depth=2, dtype=torch.float64)
        inputs = torch.tensor([[0.3, -0.4]], dtype=torch.float64)
        output = layer(inputs)
        weights = layer.weights.detach().numpy()
        np.testing.assert_allclose(output.detach(), np.cos(inputs.numpy()), atol=1e-13)
        output.square().sum().backward()
        assert layer.weights.grad.norm() > 1e-4
    else:
        jax = oq.jax
        with jax.enable_x64():
            inputs = jax.numpy.array([[0.3, -0.4]])
            if framework == "jax":
                layer = jax.QuantumLayer.identity_blocks(2, depth=2)
                weights = layer.init(jax.random.key(2), dtype=jax.numpy.float64)
                output = jax.jit(layer)(weights, inputs)
            else:
                layer = oq.flax.QuantumLayer.identity_blocks(2, depth=2, dtype=jax.numpy.float64)
                variables = layer.init(jax.random.key(2), inputs)
                output = jax.jit(layer.apply)(variables, inputs)
                weights = variables["params"]["weights"]
            np.testing.assert_allclose(output, np.cos(inputs), atol=1e-13)
            weights = np.asarray(weights)
    for a, b in recipe.pairs:
        assert weights[b] == -weights[a]
    values = dict(zip(recipe.spec.weight_parameters, weights.tolist(), strict=True))
    values.update({"x_0": 0.3, "x_1": -0.4})
    encoding = oq.Circuit(2).ry(0.3, 0).ry(-0.4, 1)
    assert oq.prove_equivalence(recipe.spec.circuit.bind(values), encoding).proven
    entangler = next(op for op in recipe.spec.circuit.operations if op.name == "rzz")
    name = next(iter(entangler.params[0].parameters))
    values[name] += 0.4
    state = oq.statevector(recipe.spec.circuit, values).reshape(2, 2)
    assert np.linalg.svd(state, compute_uv=False)[1] > 1e-4


@pytest.mark.parametrize("framework", ("torch", "jax"))
def test_existing_quantum_layers_automatically_factor_noisy_feedback(framework):
    pytest.importorskip(framework)
    x, w = oq.Parameter("x"), oq.Parameter("w")
    n = 64
    circuit = (
        oq.Circuit(n)
        .ry(x * w, 0)
        .measure(0, 3)
        .if_else(oq.ClassicalCondition((3,), 1), oq.Circuit(n).ry(w, n - 1))
        .amplitude_damping(0.2, n - 1)
    )
    observable = oq.PauliString("Z" + "I" * (n - 2) + "Z")
    if framework == "torch":
        torch = oq.torch
        layer = torch.QuantumLayer(circuit, ("x",), ("w",), (observable,), dtype=torch.float64)
        assert isinstance(layer.execution_plan, oq.ComponentPlan)
        inputs = torch.tensor([[0.2], [0.4]], dtype=torch.float64, requires_grad=True)
        output = layer(inputs)
        first = torch.autograd.grad(output.sum(), inputs, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), inputs)[0]
        assert torch.isfinite(second).all()
    else:
        jax = oq.jax
        with jax.enable_x64():
            layer = jax.QuantumLayer(circuit, ("x",), ("w",), (observable,))
            assert isinstance(layer.execution_plan, oq.ComponentPlan)
            weights = layer.init(jax.random.key(3), dtype=jax.numpy.float64)
            inputs = jax.numpy.array([[0.2], [0.4]])
            output = jax.jit(layer)(weights, inputs)
            hessian = jax.jit(jax.hessian(lambda z: layer(weights, z).sum()))(inputs)
            assert bool(jax.numpy.isfinite(hessian).all()) and output.shape == (2, 1)


def test_torch_checkpoint_rejects_changed_graph_even_with_same_parameter_count():
    pytest.importorskip("torch")
    left = oq.torch.IQPQuantumLayer(4, edges=((0, 1), (2, 3)))
    right = oq.torch.IQPQuantumLayer(4, edges=((0, 2), (1, 3)))
    with pytest.raises(oq.ValidationError, match="configuration"):
        right.load_state_dict(left.state_dict())


def test_quantum_adapter_inside_a_real_transformer_block():
    torch = pytest.importorskip("torch")
    torch.manual_seed(12)
    backbone = torch.nn.TransformerEncoderLayer(
        8, 2, dim_feedforward=16, dropout=0.0, batch_first=True
    )
    backbone.requires_grad_(False)
    adapter = oq.torch.HyperQuantumAdapter(8, num_qubits=32, rank=4)
    model = torch.nn.Sequential(backbone, adapter, torch.nn.Linear(8, 3))
    inputs = torch.randn(2, 5, 8, requires_grad=True)
    saved = backbone.linear1.weight.detach().clone()
    before = adapter.encoder.angles.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model(inputs).square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    optimizer.step()
    assert not torch.equal(before, adapter.encoder.angles.weight)
    torch.testing.assert_close(saved, backbone.linear1.weight)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None, reason="Torch unavailable"
)
def test_native_hypernetwork_cuda_when_available():
    torch = oq.torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    layer = torch.HyperQuantumAdapter(32, num_qubits=128, rank=8, device="cuda")
    x = torch.randn(2, 4, 32, device="cuda", requires_grad=True)
    layer(x).square().mean().backward()
    assert x.grad.device.type == "cuda" and torch.isfinite(x.grad).all()
