"""Independent exactness, expressivity and execution-count regressions."""

from itertools import product

import numpy as np
import pytest

import openqml as q
from openqml._tensor_contraction import TensorNetworkPlan
from openqml.ir import GATES, UNITARY_GATES


@pytest.mark.parametrize("gate", sorted(UNITARY_GATES - {"unitary"}))
def test_sparse_transfer_all_gates_all_paulis_against_dense(gate):
    arity, angles = GATES[gate]
    c = q.Circuit(3).ry(0.23, 0).rx(-0.51, 1).cx(0, 2)
    c = c.gate(
        gate,
        *(2, 0, 1)[:arity],
        params=tuple((i + 1) * q.Parameter("w").sin() for i in range(angles)),
    )
    obs = tuple(q.PauliString("".join(p)) for p in product("IXYZ", repeat=3))
    obs += (q.Hamiltonian((obs[1], q.PauliString("ZZZ", -0.3))), q.PauliString("III", 0))
    plan = q.PauliTransferPlan(c, obs)
    bindings = {"w": np.asarray([-0.4, 0, 0.7])} if angles else {}
    np.testing.assert_allclose(
        plan.evaluate(bindings), q.expectations(c, obs, bindings), atol=2e-12
    )


def test_sparse_transfer_nonlinear_broadcast_hessian_and_layer_integration():
    torch, jax = pytest.importorskip("torch"), pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    a, b = q.Parameter("a"), q.Parameter("b")
    c = q.Circuit(2).ry(a * b, 0).cx(0, 1).rx(b.sin(), 1).rzz(a, 0, 1)
    obs = (q.PauliString("XY"), q.PauliString("ZZ"))
    plan = q.PauliTransferPlan(c, obs)
    data = np.asarray([0.17, -0.31])
    plan.prepare_torch(torch.tensor(data, dtype=torch.float64))

    def function(w, compiled):
        bindings = {"a": w[0], "b": w[1]}
        result = (
            plan.evaluate(bindings, backend="torch")
            if compiled
            else q.expectations(c, obs, bindings, backend="torch")
        )
        return (result**2).sum()

    w = torch.tensor(data, requires_grad=True)
    np.testing.assert_allclose(
        torch.autograd.functional.hessian(lambda w: function(w, True), w),
        torch.autograd.functional.hessian(lambda w: function(w, False), w),
        atol=1e-12,
    )

    def jfunction(w):
        return jnp.sum(plan.evaluate(dict(a=w[0], b=w[1]), backend="jax") ** 2)

    np.testing.assert_allclose(
        jax.jit(jax.hessian(jfunction))(jnp.asarray(data, dtype="float32")),
        torch.autograd.functional.hessian(lambda w: function(w, True), w),
        atol=2e-6,
    )
    spec = q.LayerSpec(c, ("a",), ("b",), obs)
    for framework in ("torch", "jax"):
        layer_type = __import__(
            f"openqml.interfaces.{framework}", fromlist=["QuantumLayer"]
        ).QuantumLayer
        layer = layer_type.from_spec(spec, backend=framework, execution_method="pauli-transfer")
        assert isinstance(layer.execution_plan, q.PauliTransferPlan)
    broadcast = plan.evaluate(dict(a=np.asarray([[0.1], [0.2]]), b=np.asarray([0.3, 0.4, 0.5])))
    assert broadcast.shape == (2, 3, 2)
    jax.clear_caches()


def test_transfer_limits_zero_paths_and_wide_sparse_circuit():
    c = q.Circuit(1024).ry(q.Parameter("a"), 1023)
    p = q.PauliTransferPlan(c, (q.PauliString("I" * 1023 + "Z"),))
    np.testing.assert_allclose(p.evaluate(dict(a=0.3)), [np.cos(0.3)], atol=1e-14)
    for setting in (dict(max_work=1), dict(max_terms=1), dict(max_memory_mb=0.0001)):
        with pytest.raises(q.ResourceError):
            q.PauliTransferPlan(
                q.Circuit(1).ry(q.Parameter("a"), 0).rx(0.2, 0), (q.PauliString("Z"),), **setting
            )
    dead = q.PauliTransferPlan(
        q.Circuit(1).rx(q.Parameter("a"), 0).with_global_phase(q.Parameter("phase")),
        (q.PauliString("I"), q.PauliString("Z", 0)),
    )
    assert dead.active_parameters == ()
    np.testing.assert_array_equal(dead.evaluate(batch_shape=(3,)), [[1, 0]] * 3)
    with pytest.raises(q.ResourceError):
        q.PauliTransferPlan(c, p.observables, max_memory_mb=1).evaluate(
            dict(a=np.zeros(100_000)), batch_shape=(100_000,)
        )


def test_clifford_frame_fusion_preserves_full_unitaries_and_global_phase():
    from openqml.simulator import _evolve_unitary, _TensorOps

    a, b = q.Parameter("a"), q.Parameter("b")
    circuits = [
        q.Circuit(2).rzz(a, 0, 1).cx(0, 1).rz(b, 1).cx(0, 1),
        q.Circuit(2).rx(a, 0).z(0).rx(b, 0),
        q.Circuit(2).ry(a, 0).s(0).rx(b, 0),
        q.Circuit(2).rz(a, 1).rx(0.4, 0).rz(-a, 1),
        q.Circuit(2).rx(a, 0).ry(b, 0).rx(-a, 0),
        q.Circuit(2).rx(a, 0).barrier().rx(-a, 0),
    ]
    for c in circuits:
        c = c.with_global_phase(a * b)
        report = q.fuse_pauli_rotations(c)
        for bindings in (dict(a=0.7, b=-0.4), dict(a=2 * np.pi, b=0.2)):
            matrices = []
            for program in (c, report.circuit):
                program = program.bind(bindings)
                matrices.append(
                    _evolve_unitary(
                        np.eye(4, dtype=complex), program.operations, {}, 2, _TensorOps(), None
                    )
                    * np.exp(1j * program.global_phase.constant)
                )
            np.testing.assert_allclose(*matrices, atol=2e-14)
    report = q.fuse_pauli_rotations(circuits[0])
    assert report.before.gates == 4 and report.after.gates == 1
    assert report.after.multi_qubit_gates == 1
    assert q.fuse_pauli_rotations(circuits[-1]).fused_rotations == 0
    with pytest.raises(q.ResourceError):
        q.fuse_pauli_rotations(circuits[0], max_work=1)


def test_readout_contraction_reuse_reduces_work_and_does_not_cache_new_bindings():
    c = q.Circuit(3).ry(q.Parameter("a"), 0).cx(0, 1).rx(0.4, 1).cx(1, 2)
    obs = tuple(q.PauliString("".join(p)) for p in product("IZ", repeat=3))
    plain, shared = (
        TensorNetworkPlan(c, obs),
        TensorNetworkPlan(c, obs, reuse_intermediates=True),
    )
    expected, baseline = plain.evaluate(dict(a=0.3), return_report=True)
    actual, report = shared.evaluate(dict(a=0.3), return_report=True)
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    assert 0 < report["contractions_executed"] < baseline["contractions_executed"]
    changed = shared.evaluate(dict(a=-0.6))
    np.testing.assert_allclose(changed, q.expectations(c, obs, dict(a=-0.6)), atol=2e-14)
    assert np.linalg.norm(changed - expected) > 0.01


def test_fusion_and_sparse_transfer_keep_source_parameter_ownership_in_training():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    c = q.Circuit(2).h(0).rzz(q.Parameter("a"), 0, 1).cx(0, 1).rz(q.Parameter("b"), 1).cx(0, 1)
    layer = QuantumLayer(
        c,
        ("a",),
        ("b",),
        (q.PauliString("XI"),),
        backend="torch",
        dtype=torch.float64,
        optimize=True,
        optimization_options={"pauli_fusion": True},
        execution_method="pauli-transfer",
    )
    assert layer.to_spec().circuit == c and len(layer.circuit.operations) == 2
    assert layer.execution_plan._torch_prepared is not None
    with torch.no_grad():
        layer.weights.fill_(-0.2)
    x = torch.tensor([[0.3], [0.4]], dtype=torch.float64, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    np.testing.assert_allclose(y.detach().flatten(), np.cos([0.1, 0.2]), atol=1e-14)
    np.testing.assert_allclose(x.grad.flatten(), -np.sin([0.1, 0.2]), atol=1e-14)
    np.testing.assert_allclose(layer.weights.grad, [-np.sin(0.1) - np.sin(0.2)], atol=1e-14)


def test_shared_contraction_gradients_and_hessian_match_unshared():
    torch, jax = pytest.importorskip("torch"), pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    c = q.Circuit(2).ry(q.Parameter("a"), 0).cx(0, 1).rx(q.Parameter("a").sin(), 1)
    obs = (q.PauliString("ZZ"), q.PauliString("IZ"), q.PauliString("XX"))
    plans = [TensorNetworkPlan(c, obs, reuse_intermediates=b) for b in (False, True)]
    t = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    results = [
        torch.autograd.functional.hessian(
            lambda a, p=p: p.evaluate(dict(a=a), backend="torch").square().sum(), t
        )
        for p in plans
    ]
    np.testing.assert_allclose(*results, atol=1e-12)
    results = [
        jax.jit(jax.hessian(lambda a, p=p: jnp.sum(p.evaluate(dict(a=a), backend="jax") ** 2)))(
            jnp.asarray(0.31)
        )
        for p in plans
    ]
    np.testing.assert_allclose(*results, atol=1e-6)
    jax.clear_caches()


def regular_graphs():
    ring = [(i, (i + 1) % 6) for i in range(6)]
    triangles = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    edges = [[pair for a, b in graph for pair in ((a, b), (b, a))] for graph in (ring, triangles)]
    return dict(nodes=np.ones((2, 6, 1)), edge_index=np.asarray(edges))


def test_pair_model_distinguishes_a_concrete_one_wl_counterexample_and_is_invariant():
    data = regular_graphs()
    for order in (1, 2):
        p = q.HybridArchitecture.graph_network(
            1,
            2,
            order=order,
            hidden_features=8,
            depth=2,
            quantum=q.QuantumCellSpec(kind="hardware-efficient", num_qubits=2),
        )
        w = p.initial_parameters(19)
        for name in w:
            if "gain" in name:
                w[name] = np.asarray(0.5)
        y = p.evaluate(w, data)
        if order == 1:
            np.testing.assert_allclose(y[0], y[1], atol=2e-14)
        else:
            assert np.linalg.norm(y[0] - y[1]) > 1e-3
            permutation = np.asarray([3, 1, 5, 0, 4, 2])
            relabeled = dict(
                nodes=data["nodes"][:, permutation],
                edge_index=np.argsort(permutation)[data["edge_index"]],
            )
            np.testing.assert_allclose(p.evaluate(w, relabeled), y, atol=2e-14)
            assert q.HybridArchitecture.from_dict(p.to_dict()).fingerprint == p.fingerprint
            assert sum(p.execution_batches(p.input_metadata(data), 1, 0)) == 2 * 36


def test_pair_admission_precedes_quadratic_allocation_and_state_cell_validation():
    p = q.HybridArchitecture.graph_network(
        1, 1, order=2, depth=1, hidden_features=4, max_memory_mb=1
    )
    data = dict(nodes=np.ones((1, 1000, 1)), edge_index=np.zeros((1, 0, 2), dtype=int))
    with pytest.raises(q.ResourceError):
        p.evaluate(p.initial_parameters(0), data)
    with pytest.raises(q.ValidationError, match="complete local"):
        q.HybridArchitecture.transformer(
            1,
            1,
            depth=1,
            attention="state",
            quantum=q.QuantumCellSpec(kind="hardware-efficient", num_qubits=2),
        )


def test_jitted_pair_graph_rejects_bad_active_edges_even_when_every_node_is_masked():
    jax, jnp = pytest.importorskip("jax"), pytest.importorskip("jax.numpy")
    p = q.HybridArchitecture.graph_network(
        1,
        1,
        order=2,
        depth=1,
        hidden_features=4,
        quantum=q.QuantumCellSpec(kind="hardware-efficient", num_qubits=2),
        native_execution="pauli-transfer",
    )
    m = p.build("jax", backend="jax")
    w = m.init(jax.random.key(5))
    data = dict(
        nodes=jnp.ones((2, 3, 1)),
        node_mask=jnp.asarray([[1, 1, 1], [0, 0, 0]], dtype=bool),
        edge_index=jnp.asarray([[[0, 1]], [[99, -1]]]),
    )
    y = np.asarray(jax.jit(m)(w, data))
    assert np.isfinite(y[0]).all() and not np.isfinite(y[1]).any()
    data["edge_mask"] = jnp.asarray([[1], [0]], dtype=bool)
    assert np.isfinite(np.asarray(jax.jit(m)(w, data))).all()
    jax.clear_caches()


@pytest.mark.parametrize("causal", [False, True])
def test_state_attention_matches_density_matrix_channel_kernel_and_uses_one_feature_bank(causal):
    from openqml._hybrid_operators import OperatorContext

    p = q.HybridArchitecture(
        2,
        2,
        hidden_features=4,
        depth=1,
        quantum=q.QuantumCellSpec(kind="hardware-efficient", num_qubits=2, num_outputs=6),
        operators=q.OperatorSpec.state_attention(
            heads=2, causal=causal, positional=False, chunk_size=2
        ),
    )
    w = p.initial_parameters(9)
    x = np.random.default_rng(4).normal(size=(1, 5, 4))
    mask = np.asarray([[1, 1, 0, 1, 1]], dtype=bool)
    calls = []
    spec = p.cells[0].to_spec()

    def evaluator(weights, controls):
        calls.append(controls.shape[:-1])
        bindings = {n: controls[..., i] for i, n in enumerate(spec.input_parameters)}
        bindings.update({n: weights[i] for i, n in enumerate(spec.weight_parameters)})
        return q.expectations(spec.circuit, spec.observables, bindings)

    ctx = OperatorContext(p, 0, w, x, dict(mask=mask), "numpy", evaluator)
    actual = p.operator_specs[0].implementation.apply(ctx)
    assert calls == [(1, 2), (1, 2), (1, 1)]
    features = ctx.kernel(x)
    # Independent 2 x 2 density matrices and explicit Kraus conjugations.
    paulis = np.asarray(
        [np.eye(2), [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], np.diag([1, -1])], dtype=complex
    )
    r = np.stack([features[..., [0, 2, 4]], features[..., [1, 3, 5]]], axis=-2)
    rho = (paulis[0] + np.einsum("bngc,cij->bngij", r, paulis[1:])) / 2
    logits = w["block_0_state_channel"] + [2, 0, 0, 0]
    probabilities = np.exp(logits - logits.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    channels = np.stack(
        [
            sum(probabilities[h, g, a] * paulis[a] @ rho[0, :, g] @ paulis[a] for a in range(4))
            for h in range(2)
            for g in range(2)
        ]
    )
    channels = 0.99 * channels + 0.01 * paulis[0] / 2
    channels = channels.reshape((2, 2, 5, 2, 2))
    kernels = np.einsum("hgnij,hgmji->hnm", channels, channels).real / 2
    assert np.linalg.eigvalsh(kernels).min() > -1e-12 and kernels.min() > 0
    v = ctx.project("decode", features).reshape((1, 5, 2, 2))[0].transpose((1, 0, 2))
    reference = np.zeros((2, 5, 2))
    for h in range(2):
        for i in range(5):
            enabled = mask[0] & (np.arange(5) <= i if causal else True)
            if mask[0, i] and enabled.any():
                weights = kernels[h, i] * enabled
                reference[h, i] = weights @ v[h] / weights.sum()
    reference = reference.transpose((1, 0, 2)).reshape((1, 5, 4))
    np.testing.assert_allclose(actual, reference, atol=3e-14)
    meta = p.input_metadata(np.ones((1, 5, 2)))
    assert p.execution_batches(meta, 1, 0) == (2, 2, 1)
    legacy = q.HybridArchitecture.transformer(2, 2, hidden_features=4, depth=1)
    assert sum(legacy.execution_batches(meta, 1, 0)) == 3 * sum(p.execution_batches(meta, 1, 0))


@pytest.mark.parametrize("deterministic", [True, False])
def test_numeric_forward_reuses_only_deterministic_rows_with_distinct_input_cotangents(
    deterministic,
):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    class Counter:
        name = "row-reuse-test"
        capabilities = q.BackendCapabilities(
            native_expectations=True, deterministic_expectations=deterministic
        )

        def __init__(self):
            self.sizes = []

        def expectations_batch(self, circuits, observables, *, options):
            self.sizes.append(len(circuits))
            return q.expectations_batch(circuits, observables)

    backend = Counter()
    c = q.Circuit(1).ry(q.Parameter("x"), 0).rx(q.Parameter("w"), 0)
    ob = (q.PauliString("Z"),)
    layer = QuantumLayer(c, ("x",), ("w",), ob, backend=backend, dtype=torch.float64)
    with torch.no_grad():
        layer.weights.fill_(0.27)
    x = torch.full((5, 1), 0.4, dtype=torch.float64, requires_grad=True)
    y = layer(x)
    assert sum(backend.sizes) == (1 if deterministic else 5)
    cotangents = torch.arange(1, 6, dtype=torch.float64)[:, None]
    (y * cotangents).sum().backward()
    np.testing.assert_allclose(x.grad, -np.sin(0.4) * np.cos(0.27) * cotangents, atol=1e-14)
    np.testing.assert_allclose(layer.weights.grad, [-np.cos(0.4) * np.sin(0.27) * 15], atol=1e-14)
    assert sum(backend.sizes) == (5 if deterministic else 25)
    # No numerical result survives the call or a weight update.
    with torch.no_grad():
        layer.weights.fill_(0.61)
    np.testing.assert_allclose(layer(x).detach(), np.cos(0.4) * np.cos(0.61), atol=1e-14)
