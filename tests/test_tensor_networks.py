"""Independent dense/circuit references, bounded contractions and native training."""

import math

import numpy as np
import pytest

import openqml as q


@pytest.mark.parametrize("n,heads", ((1, 1), (2, 1), (3, 1), (5, 1), (7, 1), (3, 2)))
def test_quantum_tree_matches_independent_exported_quantum_circuit(n, heads):
    plan = q.QuantumTreePlan(n, heads)
    spec = plan.to_spec()
    rng = np.random.default_rng(n + heads)
    for _ in range(3):
        w, x = rng.uniform(-2, 2, plan.num_parameters), rng.uniform(-3, 3, n)
        bindings = dict(zip(spec.input_parameters, x, strict=True)) | dict(
            zip(spec.weight_parameters, w, strict=True)
        )
        actual = plan.evaluate(w, x)
        expected = q.expectations(spec.circuit, spec.observables, bindings, backend="numpy")
        np.testing.assert_allclose(actual, expected, atol=3e-14)


def test_tree_generates_entanglement_and_a_positive_normalized_root_state():
    plan = q.QuantumTreePlan(2, 1)
    weights = np.zeros(plan.num_parameters)
    weights[6:9] = math.pi / 4
    spec = plan.to_spec()
    inputs = np.array([0.0, math.pi])
    bindings = dict(zip(spec.input_parameters, inputs, strict=True)) | dict(
        zip(spec.weight_parameters, weights, strict=True)
    )
    psi = q.statevector(spec.circuit, bindings, backend="numpy").reshape(2, 2)
    rho = psi @ psi.conj().T
    np.testing.assert_allclose(rho, np.eye(2) / 2, atol=1e-14)
    np.testing.assert_allclose(
        np.linalg.svd(psi, compute_uv=False), np.ones(2) / np.sqrt(2), atol=1e-14
    )
    rng = np.random.default_rng(7)
    plan = q.QuantumTreePlan(13, 1)
    w = rng.normal(size=plan.num_parameters)
    x = rng.normal(size=(7, 13))
    bloch = []
    for polar, azimuth in ((math.pi / 2, 0), (math.pi / 2, math.pi / 2), (0, 0)):
        w[-2:] = polar, azimuth
        bloch.append(plan.evaluate(w, x)[..., 0])
    assert np.max(np.linalg.norm(np.stack(bloch, -1), axis=-1)) <= 1 + 1e-13


def parity_weights(plan):
    w = np.zeros(plan.num_parameters)
    nodes = w[: 12 * plan.num_outputs * (plan.num_qubits - 1)].reshape(
        plan.num_outputs, plan.num_qubits - 1, 12
    )
    nodes[..., 0] = math.pi / 2
    nodes[..., 1] = math.pi / 2
    nodes[..., 2] = -math.pi / 2
    nodes[..., 4] = math.pi / 2
    nodes[..., 6] = -math.pi / 2
    return w


@pytest.mark.parametrize("n", (2, 3, 8, 65, 1024))
def test_tree_represents_global_high_order_feature_interactions(n):
    plan = q.QuantumTreePlan(n, 2)
    x = np.linspace(0.001, 0.03, n)
    np.testing.assert_allclose(
        plan.evaluate(parity_weights(plan), x), np.prod(np.cos(x)), rtol=2e-12, atol=2e-12
    )
    assert len(plan.levels) == (n - 1).bit_length()


def dense_from_core_indices(plan, cores):
    matrix = np.empty((math.prod(plan.output_modes), math.prod(plan.input_modes)))
    for o in range(matrix.shape[0]):
        out_index = np.unravel_index(o, plan.output_modes)
        for i in range(matrix.shape[1]):
            in_index = np.unravel_index(i, plan.input_modes)
            value = np.ones((1, 1))
            for core, a, b in zip(cores, in_index, out_index, strict=True):
                value = value @ core[:, a, b, :]
            matrix[o, i] = value[0, 0]
    return matrix[: plan.out_features, : plan.in_features]


@pytest.mark.parametrize(
    "ins,outs,features",
    (
        ((2, 3, 2), (3, 2, 2), (11, 10)),
        ((2, 2, 2), (4, 1, 1), (8, 4)),
        ((1, 4, 2), (2, 2, 2), (7, 7)),
        ((3,), (2,), (3, 2)),
    ),
)
def test_mpo_contractions_match_independent_virtual_index_sums(ins, outs, features):
    plan = q.TensorTrainPlan(ins, outs, 2, *features)
    rng = np.random.default_rng(sum(ins) + sum(outs))
    cores = tuple(rng.normal(size=s) for s in plan.core_shapes)
    x = rng.normal(size=(2, 3, plan.in_features))
    dense = dense_from_core_indices(plan, cores)
    np.testing.assert_allclose(plan.apply(cores, x), x @ dense.T, atol=3e-14)
    if outs == (4, 1, 1):
        assert plan.direction == "right"


@pytest.mark.parametrize("shape", ((13, 11), (32, 32), (1, 7), (7, 1), (1, 1)))
def test_tt_svd_exact_import_padding_and_dense_export(shape):
    rng = np.random.default_rng(sum(shape))
    weight = rng.normal(size=shape)
    matrix = q.TensorTrainMatrix.from_dense(weight)
    np.testing.assert_allclose(matrix.to_dense(), weight, atol=8e-14)
    x = rng.normal(size=(3, shape[1]))
    np.testing.assert_allclose(matrix(x), x @ weight.T, atol=3e-13)
    assert matrix.error_bound == 0
    with pytest.raises(ValueError):
        matrix.cores[0].flat[0] = 1


def test_mpo_rank_compression_reports_real_operator_error_and_enforces_targets():
    rng = np.random.default_rng(11)
    dense = rng.normal(size=(32, 32))
    compressed = q.TensorTrainMatrix.from_dense(dense, max_bond_dimension=3)
    error = np.linalg.norm(compressed.to_dense() - dense)
    assert error <= compressed.error_bound + 1e-12
    x = rng.normal(size=32)
    assert (
        np.linalg.norm(compressed(x) - dense @ x)
        <= compressed.error_bound * np.linalg.norm(x) + 1e-12
    )
    with pytest.raises(q.ResourceError, match="rank cap"):
        q.TensorTrainMatrix.from_dense(dense, max_bond_dimension=1, relative_error=0.01)
    product = np.array([[1.0]])
    for _ in range(5):
        product = np.kron(product, rng.normal(size=(2, 2)))
    compressed = q.TensorTrainMatrix.from_dense(product, max_bond_dimension=1, relative_error=1e-10)
    np.testing.assert_allclose(compressed.to_dense(), product, atol=1e-12)
    assert compressed.plan.num_parameters < product.size // 10
    zero = q.TensorTrainMatrix.from_dense(np.zeros((9, 7)), max_bond_dimension=1, relative_error=0)
    assert zero.source_norm == zero.error_bound == 0
    np.testing.assert_array_equal(zero.to_dense(), 0)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: q.QuantumTreePlan(10**12),
        lambda: q.TensorTrainPlan.from_features(10**12, 10**12),
        lambda: q.TensorTrainMatrix.from_dense(np.ones((8, 8)), max_memory_mb=0.0001),
        lambda: q.QuantumTreePlan(4, 0),
        lambda: q.TensorTrainPlan((2, 2), (2,), 2),
        lambda: q.TensorTrainPlan((2, 2), (2, 2), 0),
    ),
)
def test_tensor_network_resources_and_invalid_contracts_are_rejected(factory):
    with pytest.raises((q.ResourceError, q.ValidationError)):
        factory()


def random_circuit(n, depth=2):
    rng = np.random.default_rng(21 + n)
    builder = q.CircuitBuilder(n)
    theta = q.Parameter("theta")
    for _layer in range(depth):
        for i in range(n):
            builder.ry(theta * (i + 1) / n, i).rz(float(rng.normal()), i)
        order = rng.permutation(n)
        for a, b in zip(order[::2], order[1::2], strict=False):
            builder.rxx(float(rng.normal()), int(a), int(b)).rzz(theta.sin(), int(a), int(b))
    if n >= 3:
        builder.ccx(0, 1, 2)
    return builder.build().with_global_phase(q.Parameter("phase"))


@pytest.mark.parametrize("n", (1, 2, 3, 5))
def test_general_circuit_tensor_network_matches_independent_dense_states(n):
    circuit = random_circuit(n)
    obs = (
        q.PauliString("Y" + "I" * (n - 1)),
        q.PauliString("Z" * n),
        q.Hamiltonian((q.PauliString("X" * n, 0.3), q.PauliString("I" * n, -0.7))),
    )
    plan = q.TensorNetworkPlan(circuit, obs)
    values = {"theta": np.array([[-0.3], [0.7]]), "phase": np.array([[0.2, 0.4, 0.8]])}
    expected = q.expectations(circuit, obs, values, backend="numpy")
    np.testing.assert_allclose(plan.evaluate(values), expected, atol=5e-14)
    np.testing.assert_allclose(
        plan.evaluate({"theta": values["theta"]}), expected[:, :1], atol=5e-14
    )
    assert "phase" not in plan.active_parameters


def slicing_circuit():
    rng = np.random.default_rng(19)
    builder = q.CircuitBuilder(10)
    for _ in range(5):
        for i in range(10):
            builder.ry(float(rng.normal()), i).rz(float(rng.normal()), i)
        order = rng.permutation(10)
        for a, b in zip(order[::2], order[1::2], strict=True):
            builder.rxx(float(rng.normal()), int(a), int(b)).rzz(
                float(rng.normal()), int(a), int(b)
            )
    return builder.build()


@pytest.mark.parametrize("reuse", [False, True])
def test_exact_slicing_reduces_live_memory_and_never_prunes_amplitudes(reuse):
    circuit, obs = slicing_circuit(), (q.PauliString("Z" * 10),)
    whole = q.TensorNetworkPlan(circuit, obs, max_memory_mb=64, max_slices=1)
    budget = (whole._metadata_bytes + 0.6 * (whole.working_bytes - whole._metadata_bytes)) / 2**20
    sliced = q.TensorNetworkPlan(
        circuit, obs, max_memory_mb=budget, max_slices=64, reuse_intermediates=reuse
    )
    assert sliced.working_bytes <= budget * 2**20 < whole.working_bytes
    assert sliced.to_dict()["contractions"][0]["slices"] > 1
    np.testing.assert_allclose(
        sliced.evaluate(), q.expectations(circuit, obs, backend="numpy"), atol=1e-13
    )
    with pytest.raises(q.ResourceError):
        q.TensorNetworkPlan(circuit, obs, max_memory_mb=budget, max_slices=1)
    with pytest.raises(q.ResourceError, match="max_flops"):
        q.TensorNetworkPlan(circuit, obs, max_flops=1)


def test_optional_cotengra_search_and_unsupported_quantum_effects():
    pytest.importorskip("cotengra")
    circuit = random_circuit(4)
    obs = (q.PauliString("XYZZ"),)
    plan = q.TensorNetworkPlan(circuit, obs, optimizer="cotengra", search_trials=2)
    np.testing.assert_allclose(
        plan.evaluate({"theta": 0.4}),
        q.expectations(circuit, obs, {"theta": 0.4, "phase": 0.8}, backend="numpy"),
        atol=2e-14,
    )
    with pytest.raises(q.ValidationError, match="unitary"):
        q.TensorNetworkPlan(q.Circuit(1).amplitude_damping(0.2, 0), (q.PauliString("Z"),))


def test_sliced_execution_reuses_pure_gate_tensors_without_caching_readouts(monkeypatch):
    import openqml.simulator as simulator

    circuit, obs = slicing_circuit(), (q.PauliString("Z" * 10),)
    whole = q.TensorNetworkPlan(circuit, obs, max_memory_mb=64, max_slices=1)
    budget = (whole._metadata_bytes + 0.6 * (whole.working_bytes - whole._metadata_bytes)) / 2**20
    sliced = q.TensorNetworkPlan(circuit, obs, max_memory_mb=budget, max_slices=64)
    calls, original = [], simulator._gate_matrix

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(simulator, "_gate_matrix", counted)
    for _ in range(2):
        sliced.evaluate()
    assert len(calls) == 2 * len(circuit.operations)


def test_wide_entangled_circuit_contracts_and_differentiates_without_a_full_state(monkeypatch):
    torch = pytest.importorskip("torch")
    import openqml.simulator as simulator

    n = 48
    x, w = q.Parameter("x"), q.Parameter("w")
    builder = q.CircuitBuilder(n)
    for i in range(n):
        builder.ry(x, i).rz(0.2, i)
    for _ in range(2):
        for start in (0, 1):
            for i in range(start, n - 1, 2):
                builder.rxx(w, i, i + 1)
    circuit = builder.build()
    obs = (q.PauliString("X" + "Z" * (n - 1)),)

    def reference(a, b):
        return q.expectations(circuit, obs, {"x": a, "w": b}, backend=q.MPSBackend())[0]

    expected = reference(0.21, 0.37)
    epsilon = 1e-5
    derivatives = (
        (reference(0.21 + epsilon, 0.37) - reference(0.21 - epsilon, 0.37)) / (2 * epsilon),
        (reference(0.21, 0.37 + epsilon) - reference(0.21, 0.37 - epsilon)) / (2 * epsilon),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A global wavefunction was allocated.")

    monkeypatch.setattr(simulator, "_initial_state", forbidden)
    plan = q.TensorNetworkPlan(circuit, obs, max_memory_mb=16)
    a, b = (
        torch.tensor(0.21, dtype=torch.float64, requires_grad=True),
        torch.tensor(0.37, dtype=torch.float64, requires_grad=True),
    )
    result = plan.evaluate({"x": a, "w": b}, backend="torch", like=a)[0]
    grads = torch.autograd.grad(result, (a, b))
    assert float(result.detach()) == pytest.approx(expected, abs=1e-13)
    np.testing.assert_allclose([float(g) for g in grads], derivatives, atol=2e-7, rtol=2e-5)


def test_torch_tree_and_mpo_native_higher_derivatives_transforms_and_compilation():
    torch = pytest.importorskip("torch")
    tree = q.torch.TreeQuantumLayer(3, num_outputs=2, dtype=torch.float64, device="cpu")
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(tree.functional, (tree.weights, x))
    assert torch.autograd.gradgradcheck(tree.functional, (tree.weights, x))
    compiled = torch.compile(tree, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), tree(x))
    primal, tangent = torch.func.jvp(tree, (x,), (torch.ones_like(x),))
    jac = torch.func.jacrev(tree)(x)
    torch.testing.assert_close(tangent, torch.einsum("abij,ij->ab", jac, torch.ones_like(x)))
    torch.testing.assert_close(torch.vmap(tree)(x), primal)
    layer = q.torch.TensorTrainLinear(7, 9, bond_dimension=2, dtype=torch.float64, device="cpu")
    x = torch.randn(2, 7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,))
    assert torch.autograd.gradgradcheck(layer, (x,))
    torch.testing.assert_close(torch.compile(layer, backend="eager", fullgraph=True)(x), layer(x))
    torch.testing.assert_close(torch.vmap(layer)(x), layer(x))


def test_torch_native_tensor_circuit_jacobians_and_checkpoint_portability():
    torch = pytest.importorskip("torch")
    circuit = (
        q.Circuit(3).ry(q.Parameter("x"), 0).cx(0, 2).rxx(q.Parameter("w").sin(), 1, 2).rx(0.3, 1)
    )
    obs = (q.PauliString("ZIZ"), q.PauliString("XYY"))
    layer = q.torch.QuantumLayer(
        circuit,
        ("x",),
        ("w",),
        obs,
        execution_method="tensor-network",
        dtype=torch.float64,
        device="cpu",
    )
    x = torch.tensor([[0.3], [0.7]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer.functional, (layer.weights, x))
    assert torch.autograd.gradgradcheck(layer.functional, (layer.weights, x))
    tree = q.torch.TreeQuantumLayer(3, num_outputs=2, dtype=torch.float64, device="cpu")
    restored = q.torch.QuantumLayer.from_checkpoint(
        tree.checkpoint(), backend="torch", execution_method="tensor-network", device="cpu"
    )
    values = torch.randn(2, 3, dtype=torch.float64)
    torch.testing.assert_close(restored(values), tree(values), atol=2e-14, rtol=1e-12)
    with pytest.raises(q.ValidationError):
        q.torch.TreeQuantumLayer(4, num_outputs=2, device="cpu").load_state_dict(tree.state_dict())


def test_torch_dense_import_and_joint_tensor_adapter_training():
    torch = pytest.importorskip("torch")
    torch.manual_seed(41)
    weight, bias = torch.randn(13, 11, dtype=torch.float64), torch.randn(13, dtype=torch.float64)
    layer = q.torch.TensorTrainLinear.from_dense(weight, bias, dtype=torch.float64, device="cpu")
    x = torch.randn(5, 11, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(layer(x), x @ weight.T + bias, atol=2e-13, rtol=1e-12)
    model = torch.nn.Sequential(
        q.torch.TensorNetworkAdapter(
            12, num_qubits=7, num_outputs=3, bond_dimension=2, device="cpu"
        ),
        torch.nn.Linear(12, 2),
    )
    inputs, target = torch.randn(6, 12), torch.randn(6, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    losses = []
    for step in range(25):
        optimizer.zero_grad()
        loss = (model(inputs) - target).square().mean()
        loss.backward()
        if step == 0:
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
            )
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.3


def test_tree_quantum_actor_supports_a_different_readout_width():
    torch = pytest.importorskip("torch")
    model = q.torch.QuantumActorCritic(
        3,
        2,
        num_qubits=7,
        quantum_layer=q.torch.TreeQuantumLayer(7, num_outputs=3, device="cpu"),
        device="cpu",
    )
    x = torch.randn(5, 3)
    result = model(x)
    result.policy.square().mean().backward()
    assert model.actor.in_features == 3 and model.quantum.weights.grad is not None


def test_jax_tensor_network_hessians_jit_vmap_and_optax_updates():
    jax, jnp, optax = (
        pytest.importorskip("jax"),
        pytest.importorskip("jax.numpy"),
        pytest.importorskip("optax"),
    )
    tree = q.jax.TreeQuantumLayer(3, num_outputs=2)
    w = tree.init(jax.random.key(3))
    x = jnp.array([[0.2, 0.3, 0.4], [-0.2, 0.5, 0.1]])
    value = jax.jit(tree)(w, x)
    np.testing.assert_allclose(value, tree.plan.evaluate(np.asarray(w), np.asarray(x)), atol=2e-6)
    np.testing.assert_allclose(jax.vmap(lambda row: tree(w, row))(x), value, atol=1e-6)
    hessian = jax.jit(jax.hessian(lambda w: tree(w, x).sum()))(w)
    assert np.isfinite(hessian).all()
    native = q.jax.QuantumLayer.from_spec(
        tree.to_spec(), backend="jax", execution_method="tensor-network"
    )
    np.testing.assert_allclose(jax.jit(native)(w, x), value, atol=2e-6)
    mpo, params = q.jax.TensorTrainLinear.from_dense(np.arange(63.0).reshape(9, 7) * 0.01)
    np.testing.assert_allclose(
        jax.jit(mpo)(params, jnp.ones((2, 7))),
        np.ones((2, 7)) @ np.arange(63.0).reshape(9, 7).T * 0.01,
        atol=2e-6,
    )
    adapter = q.jax.TensorNetworkAdapter(12, num_qubits=7, num_outputs=3, bond_dimension=2)
    p = adapter.init(jax.random.key(8))
    inputs = jnp.arange(48.0, dtype=jnp.float32).reshape(4, 12) / 48
    optimizer = optax.adam(0.03)
    state = optimizer.init(p)

    def loss(p):
        return jnp.mean((adapter(p, inputs) - inputs * 0.7) ** 2)

    @jax.jit
    def update(p, state):
        value, grad = jax.value_and_grad(loss)(p)
        delta, state = optimizer.update(grad, state, p)
        return optax.apply_updates(p, delta), state, value

    before = float(loss(p))
    for _ in range(25):
        p, state, _ = update(p, state)
    assert float(loss(p)) < before * 0.5


def test_flax_linen_tensor_modules_and_quantum_policy_gradients():
    pytest.importorskip("flax")
    jax, jnp = pytest.importorskip("jax"), pytest.importorskip("jax.numpy")
    x = jnp.ones((2, 3, 12)) * 0.3
    module = q.flax.TensorNetworkAdapter(12, num_qubits=7, num_outputs=3, bond_dimension=2)
    variables = module.init(jax.random.key(8), x)
    value, grad = jax.jit(jax.value_and_grad(lambda p: jnp.mean(module.apply(p, x) ** 2)))(
        variables
    )
    assert np.isfinite(value) and all(np.isfinite(a).all() for a in jax.tree.leaves(grad))
    actor = q.flax.QuantumActorCritic(
        12, 2, num_qubits=7, quantum_layer=q.flax.TreeQuantumLayer(7, num_outputs=3)
    )
    variables = actor.init(jax.random.key(9), x)
    assert jax.jit(actor.apply)(variables, x).policy.shape == (2, 3, 2)
    dense = np.arange(63.0).reshape(9, 7) * 0.01
    imported, variables = q.flax.TensorTrainLinear.from_dense(dense, np.arange(9.0) * 0.1)
    np.testing.assert_allclose(
        jax.jit(imported.apply)(variables, jnp.ones((2, 7))),
        np.ones((2, 7)) @ dense.T + np.arange(9.0) * 0.1,
        atol=2e-6,
    )
    assert imported.initial_error_bound == 0 and imported.initial_source_norm > 0


def test_classical_mpo_rejects_complex_values_instead_of_discarding_phase():
    plan = q.TensorTrainPlan((2,), (2,), 1)
    with pytest.raises(q.ValidationError, match="real"):
        q.TensorTrainMatrix(plan, (np.ones(plan.core_shapes[0], dtype=complex),))
    with pytest.raises(q.ValidationError, match="real"):
        plan.apply((np.ones(plan.core_shapes[0]),), np.ones(2, dtype=complex))


def test_large_tensor_layers_never_allocate_an_exponential_quantum_state(monkeypatch):
    torch = pytest.importorskip("torch")
    import openqml.simulator as simulator

    def forbidden(*args, **kwargs):
        raise AssertionError("An exponential quantum state was allocated.")

    monkeypatch.setattr(simulator, "_initial_state", forbidden)
    tree = q.torch.TreeQuantumLayer(1024, num_outputs=2, device="cpu")
    x = torch.full((3, 1024), 0.02, requires_grad=True)
    tree(x).sum().backward()
    assert x.grad is not None and tree.weights.grad is not None
    mpo = q.torch.TensorTrainLinear(1024, 1024, bond_dimension=4, device="cpu")
    assert mpo.plan.num_parameters < 1024 * 1024 // 100
    mpo(x).sum().backward()


def test_cuda_tensor_adapter_and_complex_contractions_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")
    layer = q.torch.TensorNetworkAdapter(16, num_qubits=8, num_outputs=4, device="cuda")
    x = torch.randn(3, 16, device="cuda", requires_grad=True)
    result = layer(x)
    result.sum().backward()
    assert result.device.type == "cuda" and x.grad is not None


def test_readme_quickstart_is_executable_and_completes_an_optimizer_step():
    pytest.importorskip("torch")
    import re
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", readme, re.S)
    assert blocks
    namespace = {}
    for block in blocks:
        exec(compile(block, "README.md", "exec"), namespace)
    model, trainer, metrics = (namespace[name] for name in ("model", "trainer", "metrics"))
    assert tuple(model(namespace["x"]).shape) == (32, 2)
    assert metrics["updates"] == trainer.updates == 1 and not metrics["skipped"]
    assert np.isfinite(metrics["loss"])
    assert trainer.optimizer.state
    assert any(state["exp_avg"].count_nonzero() for state in trainer.optimizer.state.values())
