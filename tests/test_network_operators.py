"""Structured models share semantics, quantum gradients and training ownership."""

import copy

import numpy as np
import pytest

import openqml as q


def plan(operator, **options):
    config = dict(
        input_features=2,
        output_features=2,
        hidden_features=4,
        depth=1,
        quantum=q.QuantumCellSpec(kind="hardware-efficient", num_qubits=2),
        operators=operator,
    )
    config.update(options)
    return q.HybridArchitecture(**config)


def opened(p, dtype="float32"):
    values = p.initial_parameters(19, dtype=dtype)
    for i in range(p.depth):
        values[f"block_{i}_gain_q"] = np.asarray(0.3, dtype=dtype)
        values[f"block_{i}_gain_ff"] = np.asarray(-0.1, dtype=dtype)
    return values


def graph_inputs():
    return dict(
        nodes=np.random.default_rng(9).normal(size=(2, 4, 2)).astype("float32"),
        edge_index=np.asarray([[[0, 1], [1, 2], [2, 0], [3, 1], [-1, -1]]] * 2, dtype="int32"),
        node_mask=np.asarray([[1, 1, 1, 0]] * 2, dtype=bool),
        edge_mask=np.asarray([[1, 1, 1, 0, 0]] * 2, dtype=bool),
        edge_features=np.random.default_rng(10).normal(size=(2, 5, 1)).astype("float32"),
        positions=np.random.default_rng(11).normal(size=(2, 4, 3)).astype("float32"),
    )


@pytest.mark.parametrize(
    "kind", ["conv1d", "conv2d", "gaussian", "linear", "graph", "state", "pair"]
)
def test_structured_model_outputs_and_all_parameter_gradients_match_across_frameworks(kind):
    torch, jax = pytest.importorskip("torch"), pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    if kind.startswith("conv"):
        dimensions = 1 if kind == "conv1d" else 2
        op = q.OperatorSpec.convolution(spatial_dims=dimensions, chunk_size=2)
        shape = (2, 4, 2) if dimensions == 1 else (2, 2, 3, 2)
        data = np.random.default_rng(12).normal(size=shape).astype("float32")
    elif kind in {"graph", "pair"}:
        op, data = (
            q.OperatorSpec.graph(edge_features=1, geometric=True, chunk_size=2),
            graph_inputs(),
        )
    elif kind == "state":
        op = q.OperatorSpec.state_attention(heads=2, causal=True, chunk_size=2)
        data = dict(
            x=np.random.default_rng(12).normal(size=(2, 4, 2)).astype("float32"),
            mask=np.asarray([[1, 1, 1, 0], [0, 0, 0, 0]], dtype=bool),
        )
    else:
        op = q.OperatorSpec.attention(heads=2, key_features=3, mode=kind, causal=True, chunk_size=2)
        data = dict(
            x=np.random.default_rng(12).normal(size=(2, 4, 2)).astype("float32"),
            mask=np.asarray([[1, 1, 1, 0], [0, 0, 0, 0]], dtype=bool),
        )
    p, values = (None if kind == "state" else plan(op)), None
    if kind == "state":
        p = plan(
            op,
            quantum=q.QuantumCellSpec(kind="qcnn", num_qubits=2),
            native_execution="pauli-transfer",
        )
    elif kind == "pair":
        p = q.HybridArchitecture.graph_network(
            2,
            2,
            order=2,
            edge_features=1,
            geometric=True,
            hidden_features=4,
            depth=1,
            quantum=q.QuantumCellSpec(kind="hardware-efficient", num_qubits=2),
            native_execution="pauli-transfer",
        )
    values = opened(p)
    expected = p.evaluate(values, data)
    model = p.build("torch", backend="torch", device="cpu")
    with torch.no_grad():
        for name, value in values.items():
            model.values[name].copy_(torch.from_numpy(value))
    td = (
        {k: torch.from_numpy(v) for k, v in data.items()}
        if isinstance(data, dict)
        else torch.from_numpy(data)
    )
    got = model(td)
    got.square().sum().backward()
    np.testing.assert_allclose(got.detach(), expected, atol=3e-6, rtol=3e-5)
    functional = p.build("jax", backend="jax")
    jd = jax.tree_util.tree_map(jnp.asarray, data)
    jvalues = jax.tree_util.tree_map(jnp.asarray, values)
    jy = jax.jit(functional)(jvalues, jd)
    grads = jax.jit(jax.grad(lambda w: jnp.sum(functional(w, jd) ** 2)))(jvalues)
    np.testing.assert_allclose(jy, expected, atol=3e-6, rtol=3e-5)
    for name in values:
        np.testing.assert_allclose(
            model.values[name].grad, grads[name], atol=4e-5, rtol=6e-4, err_msg=name
        )
    quantum = [name for name in values if "_quantum" in name]
    assert all(np.linalg.norm(grads[name]) > 0 for name in quantum)
    pytest.importorskip("flax")
    linen = p.build("flax", backend="jax")
    initialized = linen.init(jax.random.key(3), jd)
    assert set(initialized["params"]) == set(values)
    np.testing.assert_allclose(
        jax.jit(linen.apply)({"params": jvalues}, jd), expected, atol=3e-6, rtol=3e-5
    )
    jax.clear_caches()


@pytest.mark.parametrize("mode", ["gaussian", "linear", "convolution", "state"])
def test_causal_prefix_and_padding_semantics(mode):
    op = (
        q.OperatorSpec.convolution(padding="causal", chunk_size=2)
        if mode == "convolution"
        else q.OperatorSpec.state_attention(causal=True, chunk_size=2)
        if mode == "state"
        else q.OperatorSpec.attention(mode=mode, causal=True, chunk_size=2)
    )
    p = (
        plan(op, quantum=q.QuantumCellSpec(kind="qcnn", num_qubits=2))
        if mode == "state"
        else plan(op)
    )
    values = opened(p, dtype="float64")
    data = dict(
        x=np.random.default_rng(4).normal(size=(2, 6, 2)),
        mask=np.asarray([[1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]], dtype=bool),
    )
    before = p.evaluate(values, data)
    changed = copy.deepcopy(data)
    changed["x"][:, 3:] += 100
    np.testing.assert_allclose(p.evaluate(values, changed)[:, :3], before[:, :3], atol=1e-12)
    assert np.isfinite(before).all() and not before[1].any() and not before[0, -1].any()
    changed = copy.deepcopy(data)
    changed["x"][~changed["mask"]] = np.nan
    np.testing.assert_allclose(p.evaluate(values, changed), before, atol=1e-12)


@pytest.mark.parametrize("order", [1, 2])
def test_graph_permutation_and_euclidean_invariance_and_padding(order):
    p = (
        plan(q.OperatorSpec.graph(edge_features=1, geometric=True, chunk_size=2))
        if order == 1
        else (
            q.HybridArchitecture.graph_network(
                2,
                2,
                hidden_features=4,
                depth=1,
                order=2,
                edge_features=1,
                geometric=True,
                readout_pooling="none",
                quantum=q.QuantumCellSpec(kind="hardware-efficient", num_qubits=2),
            )
        )
    )
    values, data = opened(p), graph_inputs()
    expected = p.evaluate(values, data)
    perm = np.asarray([2, 0, 3, 1])
    inverse = np.argsort(perm)
    transformed = {k: v.copy() for k, v in data.items()}
    for key in ("nodes", "positions", "node_mask"):
        transformed[key] = transformed[key][:, perm]
    transformed["edge_index"] = inverse[np.maximum(data["edge_index"], 0)]
    np.testing.assert_allclose(
        p.evaluate(values, transformed),
        expected[:, perm] if order == 1 else expected[:, perm][:, :, perm],
        atol=3e-6,
        rtol=3e-5,
    )
    transformed = copy.deepcopy(data)
    rotation = np.linalg.qr(np.random.default_rng(15).normal(size=(3, 3)))[0].astype("float32")
    transformed["positions"] = transformed["positions"] @ rotation + 3
    np.testing.assert_allclose(p.evaluate(values, transformed), expected, atol=4e-6, rtol=4e-5)
    transformed["nodes"][~data["node_mask"]] = np.nan
    transformed["positions"][~data["node_mask"]] = np.nan
    np.testing.assert_allclose(p.evaluate(values, transformed), expected, atol=4e-6, rtol=4e-5)
    bad = copy.deepcopy(data)
    bad["edge_index"][0, 0, 0] = 100
    with pytest.raises(q.ValidationError, match="out-of-range"):
        p.evaluate(values, bad)


def test_shared_qcnn_pooling_odd_width_and_resource_admission():
    cell = q.FeatureCircuitPlan(5, 1, 100, pooling=True, retained_qubits=4)
    assert (
        len(cell.retained_wires) == 4 and len(set(o.paulis for o in cell.spec.observables)) == 100
    )
    for n in (1, 3, 5, 8):
        c = q.FeatureCircuitPlan(n, pooling=True)
        assert len(c.retained_wires) == 1
        values = {name: 0.12 for name in c.spec.circuit.parameters}
        output = q.expectations(c.spec.circuit, c.spec.observables, values, backend="numpy")
        assert np.isfinite(output).all() and np.max(np.abs(output)) <= 1 + 1e-10
    with pytest.raises(q.ResourceError):
        q.FeatureCircuitPlan(100, max_operations=5)


def test_operator_registry_roundtrip_and_mixed_composition():
    class Scaled(q.NetworkOperator):
        def validate(self, c):
            assert set(c) == {"scale"}

        def apply(self, ctx):
            return super().apply(ctx) * ctx.spec.config["scale"]

    q.register_network_operator("tests.scaled", Scaled())
    custom = q.OperatorSpec("tests.scaled", {"scale": 2})
    p = plan(
        (q.OperatorSpec.convolution(), q.OperatorSpec.attention(mode="linear"), custom), depth=3
    )
    values = opened(p)
    x = np.random.default_rng(18).normal(size=(2, 4, 2)).astype("float32")
    restored = q.HybridArchitecture.from_dict(p.to_dict())
    assert restored.fingerprint == p.fingerprint
    np.testing.assert_allclose(restored.evaluate(values, x), p.evaluate(values, x), atol=0, rtol=0)
    assert "tests.scaled" in q.list_network_operators()
    with pytest.raises(q.ValidationError, match="already"):
        q.register_network_operator("tests.scaled", Scaled())


def test_planner_counts_windows_tokens_edges_and_sequential_call_batches():
    metadata = {"samples": 3, "sites": 5, "edges": 7, "shape": (3, 5, 2)}
    conv = plan(q.OperatorSpec.convolution(chunk_size=3))
    graph = plan(q.OperatorSpec.graph(chunk_size=3))
    attention = plan(q.OperatorSpec.attention())
    assert conv.execution_batches(metadata, 2, 0) == (6, 4, 3, 2)
    assert graph.execution_batches(metadata, 2, 0) == (6, 6, 2, 3, 3, 1)
    assert attention.execution_batches(metadata, 2, 0) == (10, 10, 10, 5, 5, 5)
    for p in (conv, graph, attention):
        forecast = p.schedule(
            scheduler=q.CostAwareBackend(
                profiles={
                    "cpu": q.ExecutionProfile(
                        0.1, cost_per_second=0, cost_per_circuit=1, relative_uncertainty=0
                    )
                },
                policy=q.SchedulingPolicy(require_profiles=True),
            ),
            metadata=metadata,
            microbatch_samples=2,
        )
        calls = sum(p.execution_batches(metadata, 2, 0))
        assert forecast.forecast.circuit_evaluations == calls * 3  # forward, replay, backward
        assert forecast.forecast.upper_cost == calls * 3


@pytest.mark.parametrize("family", ["convolution", "attention", "graph", "state", "pair"])
def test_complete_models_train_and_resume_without_replacing_parameters(family, tmp_path):
    torch = pytest.importorskip("torch")
    if family == "convolution":
        p = q.HybridArchitecture.convolutional(
            2, 2, hidden_features=4, depth=2, quantum=q.QuantumCellSpec(kind="qcnn", num_qubits=2)
        )
        data = torch.tensor(
            np.random.default_rng(23).normal(size=(3, 2, 2, 2)), dtype=torch.float32
        )
        target = data.mean(dim=(1, 2))
    elif family in {"attention", "state"}:
        p = q.HybridArchitecture.transformer(
            2,
            2,
            hidden_features=4,
            depth=2,
            heads=2,
            attention="state" if family == "state" else "linear",
            readout_pooling="mean",
            quantum=q.QuantumCellSpec(
                kind="qcnn" if family == "state" else "hardware-efficient", num_qubits=2
            ),
            native_execution="pauli-transfer" if family == "state" else "auto",
        )
        data = torch.tensor(np.random.default_rng(23).normal(size=(3, 4, 2)), dtype=torch.float32)
        target = data.mean(dim=1)
    else:
        p = q.HybridArchitecture.graph_network(
            2,
            2,
            hidden_features=4,
            depth=2,
            edge_features=1,
            geometric=True,
            order=2 if family == "pair" else 1,
            quantum=q.QuantumCellSpec(kind="hardware-efficient", num_qubits=2),
            native_execution="pauli-transfer" if family == "pair" else "auto",
        )
        data = {k: torch.from_numpy(v) for k, v in graph_inputs().items()}
        target = data["nodes"][:, :3].mean(dim=1)
    model = p.build("torch", backend="torch", device="cpu", active_depth=1)
    with torch.no_grad():
        for name, value in opened(p).items():
            model.values[name].copy_(torch.from_numpy(value))
        # The future block is frozen at identity until its curriculum boundary.
        model.values["block_1_gain_q"].zero_()
        model.values["block_1_gain_ff"].zero_()
    identities = {name: id(value) for name, value in model.values.items()}
    before = {
        name: value.detach().clone()
        for name, value in model.values.items()
        if "_quantum" in name and name.startswith("block_0_")
    }
    trainer = model.trainer(
        lr=0.02,
        quantum_lr=0.01,
        microbatch_size=1,
        depth_schedule=q.DepthSchedule(((0, 1), (3, 2))),
    )

    def loss(m, b):
        return (m(b[0]) - b[1]).square().mean()

    initial = float(loss(model, (data, target)).detach())
    for _ in range(8):
        metrics = trainer.step((data, target), loss)
    assert metrics["updates"] == 8 and metrics["active_depth"] == 2
    assert float(loss(model, (data, target)).detach()) < initial
    assert identities == {name: id(value) for name, value in model.values.items()}
    assert all(not torch.equal(model.values[name], value) for name, value in before.items())
    trainer.save(tmp_path / f"{family}.pt")
    restored = p.build("torch", backend="torch", device="cpu").trainer(
        lr=0.02, quantum_lr=0.01, microbatch_size=1
    )
    restored.load(tmp_path / f"{family}.pt")
    trainer.step((data, target), loss)
    restored.step((data, target), loss)
    for name in model.values:
        torch.testing.assert_close(
            model.values[name], restored.network.values[name], atol=0, rtol=0
        )


@pytest.mark.parametrize("family", ["convolution", "attention"])
def test_qpu_forecast_matches_actual_provider_chunks_and_reserved_cost(monkeypatch, family):
    torch = pytest.importorskip("torch")
    from openqml import backends

    class Provider:
        name = "structured-test-qpu"
        capabilities = q.BackendCapabilities(native_expectations=True, device_type="QPU")

        def __init__(self):
            self.calls = []

        def is_available(self):
            return True

        def expectations_batch(self, circuits, observables, *, options):
            self.calls.append(len(circuits))
            return q.expectations_batch(circuits, observables, backend="numpy", **options.__dict__)

    provider = Provider()
    monkeypatch.setitem(backends._REGISTRY, provider.name, provider)
    ledger = q.BudgetLedger(1000)
    runtime = q.CostAwareBackend(
        profiles={
            provider.name: q.ExecutionProfile(
                0.001,
                queue_seconds=0,
                cost_per_second=0,
                cost_per_job=1,
                noise_bias=0,
                variance=0,
                max_batch_size=3,
                relative_uncertainty=0,
                source="synthetic_correctness_only",
            )
        },
        ledger=ledger,
    )
    op = (
        q.OperatorSpec.convolution(kernel_size=3, chunk_size=5)
        if family == "convolution"
        else q.OperatorSpec.attention(mode="gaussian", chunk_size=3)
    )
    p = plan(op, quantum=q.QuantumCellSpec(kind="hardware-efficient", num_qubits=1))
    model = p.build("torch", backend=runtime, device="cpu")
    with torch.no_grad():
        for name, value in opened(p).items():
            model.values[name].copy_(torch.from_numpy(value))
    trainer = model.trainer(lr=0.001, microbatch_size=2)
    x = torch.tensor(np.random.default_rng(17).normal(size=(3, 5, 2)), dtype=torch.float32)
    metrics = trainer.step((x, x * 0.7), lambda m, b: (m(b[0]) - b[1]).square().mean())
    assert not metrics["skipped"] and provider.calls
    forecast = model.execution_plan.forecast
    assert forecast.provider_jobs == len(provider.calls)
    assert ledger.reserved == len(provider.calls) == forecast.upper_cost
    assert max(provider.calls) <= 3
    count = len(provider.calls)
    refreshed = model.reschedule(
        workload=q.TrainingWorkload(batch_size=15, gradient_chunk_size=64),
        metadata=p.input_metadata(x),
        microbatch_samples=2,
    )
    assert refreshed.forecast.observations > 0
    assert len(provider.calls) == count


def test_default_architecture_serialization_preserves_legacy_fingerprint():
    import hashlib
    import json

    p = q.HybridArchitecture(2, 1, hidden_features=4, depth=2)
    value = p.to_dict()
    assert "operators" not in value and "readout_pooling" not in value
    expected = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    assert q.HybridArchitecture.from_dict(value).fingerprint == expected
    op = q.OperatorSpec("pointwise")
    with pytest.raises(TypeError):
        op.config["bad"] = True


def test_portable_custom_cells_include_sdk_composition_and_frozen_quantum_features():
    x, w = q.Parameter("x"), q.Parameter("w")
    circuit = q.Circuit(2).ry(x, 0).rxx(w, 0, 1).rz(x, 1)
    spec = q.LayerSpec(circuit, ("x",), ("w",), (q.PauliString("ZI"), q.PauliString("XX")))
    cell = q.QuantumCellSpec.from_spec(spec)
    p = plan(q.OperatorSpec.attention(key_features=2), quantum=cell)
    restored = q.HybridArchitecture.from_dict(p.to_dict())
    assert restored.fingerprint == p.fingerprint
    assert restored.cells[0].to_spec().to_dict() == spec.to_dict()
    data = np.ones((2, 3, 2), dtype="float32") * 0.2
    np.testing.assert_allclose(
        p.evaluate(opened(p), data), restored.evaluate(opened(p), data), atol=0, rtol=0
    )
    fixed = q.LayerSpec(q.Circuit(1).ry(x, 0), ("x",), (), (q.PauliString("Z"),))
    p = plan(q.OperatorSpec.convolution(), quantum=q.QuantumCellSpec.from_spec(fixed))
    assert p.initial_parameters(1)["block_0_quantum"].shape == (0,)
    assert np.isfinite(p.evaluate(p.initial_parameters(1), data)).all()
    torch = pytest.importorskip("torch")
    m = p.build("torch", backend="torch", device="cpu")
    m(torch.from_numpy(data)).square().sum().backward()
    assert m.values["stem_weight"].grad is not None


def test_flax_graph_attention_model_progressive_training_and_checkpoint(tmp_path):
    jax, jnp = pytest.importorskip("jax"), pytest.importorskip("jax.numpy")
    optax = pytest.importorskip("optax")
    pytest.importorskip("flax")
    p = plan(
        (
            q.OperatorSpec.graph(edge_features=1, geometric=True),
            q.OperatorSpec.attention(mode="linear", positional=False),
        ),
        depth=2,
        readout_pooling="mean",
    )
    module = p.build("flax", backend="jax", active_depth=1)
    data = jax.tree_util.tree_map(jnp.asarray, graph_inputs())
    target = jnp.mean(data["nodes"][:, :3], axis=1)
    variables = module.init(jax.random.key(19), data)

    def loss(model, params, b):
        return jnp.mean((model(params, b[0]) - b[1]) ** 2, axis=-1)

    trainer = module.trainer(
        optax.adam(0.02),
        loss,
        variables=variables,
        microbatch_size=1,
        depth_schedule=q.DepthSchedule(((0, 1), (2, 2))),
    )
    for _ in range(4):
        metrics = trainer.step((data, target))
    assert trainer.updates == 4 and metrics["active_depth"] == 2
    full = p.build("flax", backend="jax", active_depth=2)
    np.testing.assert_allclose(
        full.apply(trainer.variables, data),
        trainer.model(trainer.parameters, data),
        atol=2e-6,
        rtol=2e-5,
    )
    trainer.save(tmp_path / "mixed.npz")
    restored = module.trainer(optax.adam(0.02), loss, variables=variables, microbatch_size=1)
    restored.load(tmp_path / "mixed.npz")
    trainer.step((data, target))
    restored.step((data, target))
    for name in trainer.parameters:
        np.testing.assert_array_equal(trainer.parameters[name], restored.parameters[name])
    jax.clear_caches()


def test_jax_native_gpu_admission_control_without_a_torch_gpu(monkeypatch):
    """Control-path unit test only; this does not claim physical GPU execution."""
    jax, jnp = pytest.importorskip("jax"), pytest.importorskip("jax.numpy")
    optax = pytest.importorskip("optax")
    from openqml import backends
    from openqml._cuda_backend import CUDABackend

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    def forbid_torch_probe(self):
        raise AssertionError("JAX native GPU admission must not probe Torch CUDA")

    monkeypatch.setattr(CUDABackend, "is_available", forbid_torch_probe)
    monkeypatch.setitem(backends._REGISTRY, "gpu", CUDABackend())
    ledger = q.BudgetLedger(1)
    runtime = q.CostAwareBackend(
        ledger=ledger,
        profiles={
            "jax-gpu": q.ExecutionProfile(
                0.01,
                queue_seconds=0,
                cost_per_second=0,
                cost_per_circuit=0.001,
                relative_uncertainty=0,
            )
        },
    )
    p = plan(q.OperatorSpec.attention(mode="linear"))
    model = p.build("jax", backend=runtime)
    assert all(stage.backend == "jax-gpu" for stage in model.execution_plan.stages)
    assert all(layer.execution_backend == "jax" for layer in model._layers)
    trainer = model.trainer(
        optax.adam(0.01),
        lambda m, w, b: jnp.mean((m(w, b[0]) - b[1]) ** 2, axis=(1, 2)),
        key=jax.random.key(3),
        microbatch_size=1,
    )
    x = jnp.asarray(np.random.default_rng(7).normal(size=(2, 3, 2)), dtype=jnp.float32)
    for _ in range(2):
        metrics = trainer.step((x, x * 0.7))
    assert trainer.updates == 2 and not bool(metrics["skipped"])
    assert ledger.reserved == pytest.approx(0.108)
    candidate = model._native_candidate
    circuit = q.Circuit(1).ry(0.2, 0)
    numeric = candidate.expectations_batch(
        [circuit], (q.PauliString("Z"),), options=q.ExecutionOptions()
    )
    np.testing.assert_allclose(numeric, [[np.cos(0.2)]], atol=1e-6)
    with pytest.raises(q.ValidationError, match="unique"):
        q.CostAwareBackend().plan(
            circuit, observables=(q.PauliString("Z"),), native_candidates=(candidate, candidate)
        )
    jax.clear_caches()


def test_physical_jax_gpu_keeps_native_gradients_without_torch_dispatch(monkeypatch):
    jax, jnp = pytest.importorskip("jax"), pytest.importorskip("jax.numpy")
    if jax.default_backend() != "gpu":
        pytest.skip("A physical JAX GPU backend is required")
    from openqml import backends
    from openqml._cuda_backend import CUDABackend

    monkeypatch.setattr(CUDABackend, "is_available", lambda self: False)
    monkeypatch.setitem(backends._REGISTRY, "gpu", CUDABackend())
    p = plan(q.OperatorSpec.attention(mode="linear"))
    model = p.build("jax")
    params = model.init(jax.random.key(8))
    x = jnp.ones((2, 3, 2), dtype=jnp.float32)
    grads = jax.jit(jax.grad(lambda w: jnp.sum(model(w, x) ** 2)))(params)
    assert all(stage.backend == "jax-gpu" for stage in model.execution_plan.stages)
    assert all(device.platform == "gpu" for value in grads.values() for device in value.devices())
    assert all(np.isfinite(value).all() for value in grads.values())
