"""A shared model program, real training, depth growth and network-level budgets."""

import copy

import numpy as np
import pytest

import openqml as q


def architecture(**options):
    defaults = dict(
        input_features=2,
        output_features=1,
        hidden_features=4,
        depth=3,
        quantum=q.QuantumCellSpec(num_qubits=2),
        rematerialize=True,
    )
    defaults.update(options)
    return q.HybridArchitecture(**defaults)


def test_shared_layout_identity_growth_and_architecture_roundtrip():
    plan = architecture()
    parameters = plan.initial_parameters(19)
    x = np.random.default_rng(20).normal(size=(2, 3, 2))
    baseline = (x @ parameters["stem_weight"] + parameters["stem_bias"]) @ parameters[
        "readout_weight"
    ]
    baseline += parameters["readout_bias"]
    for depth in range(4):
        np.testing.assert_allclose(
            plan.evaluate(parameters, x, active_depth=depth), baseline, atol=1e-14
        )
    restored = q.HybridArchitecture.from_dict(plan.to_dict())
    assert restored.fingerprint == plan.fingerprint
    assert restored.parameters == plan.parameters
    profile = plan.resource_profile(active_depth=1)
    assert profile["active_depth"] == 1 and profile["depth"] == 3
    assert profile["projection_parameters"] <= profile["dense_projection_parameters"]
    with pytest.raises(q.ResourceError):
        architecture(depth=10000, max_parameters=100)
    with pytest.raises(q.ValidationError):
        plan.evaluate({"wrong": np.ones(2)}, x)


def test_one_model_program_has_matching_torch_jax_flax_outputs_and_gradients():
    torch = pytest.importorskip("torch")
    jax, jnp = pytest.importorskip("jax"), pytest.importorskip("jax.numpy")
    plan = architecture(
        depth=2,
        quantum=(
            q.QuantumCellSpec(num_qubits=2),
            q.QuantumCellSpec(kind="residual-channel", num_qubits=2, depth=1),
        ),
    )
    parameters = plan.initial_parameters(7, dtype="float32")
    for i in range(plan.depth):
        parameters[f"block_{i}_gain_q"] = np.asarray(0.2, dtype="float32")
        parameters[f"block_{i}_gain_ff"] = np.asarray(-0.1, dtype="float32")
    x = np.random.default_rng(5).normal(size=(3, 2)).astype("float32")
    expected = plan.evaluate(parameters, x)
    model = plan.build("torch", backend="torch")
    with torch.no_grad():
        for name, value in parameters.items():
            model.values[name].copy_(torch.from_numpy(value))
    y = model(torch.from_numpy(x))
    y.square().sum().backward()
    np.testing.assert_allclose(y.detach(), expected, atol=2e-6, rtol=2e-5)
    functional = plan.build("jax", backend="jax")
    values = {name: jnp.asarray(value) for name, value in parameters.items()}
    got = jax.jit(functional)(values, jnp.asarray(x))
    gradient = jax.jit(jax.grad(lambda p: jnp.sum(functional(p, jnp.asarray(x)) ** 2)))(values)
    np.testing.assert_allclose(got, expected, atol=2e-6, rtol=2e-5)
    for name in ("stem_weight", "block_0_quantum", "block_1_quantum", "readout_weight"):
        np.testing.assert_allclose(model.values[name].grad, gradient[name], atol=3e-5, rtol=3e-4)
    assert np.linalg.norm(gradient["block_0_quantum"]) > 0
    assert np.linalg.norm(gradient["block_1_quantum"]) > 0
    pytest.importorskip("flax")
    linen = plan.build("flax", backend="jax")
    variables = linen.init(jax.random.key(2), jnp.asarray(x))
    assert set(variables["params"]) == set(values)
    np.testing.assert_allclose(
        jax.jit(linen.apply)({"params": values}, jnp.asarray(x)), got, atol=2e-6, rtol=2e-5
    )


def test_torch_progressive_training_preserves_parameter_identity_optimizer_age_and_resume(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(23)
    plan = architecture()
    model = plan.build("torch", backend="torch", active_depth=1, dtype=torch.float64)
    trainer = model.trainer(
        lr=0.02, microbatch_size=2, depth_schedule=q.DepthSchedule(((0, 1), (3, 2), (7, 3)))
    )
    x = torch.tensor([[-1.0, -0.5], [-0.4, 0.2], [0.3, 0.6], [1.0, -0.2]], dtype=torch.float64)
    batch = (x, (x[:, :1] - 0.4 * x[:, 1:]))

    def loss(network, b):
        return (network(b[0]) - b[1]).square().mean()

    ids = {name: id(p) for name, p in model.values.items()}
    inactive = {
        name: value.detach().clone()
        for name, value in model.values.items()
        if name.startswith("block_1_") or name.startswith("block_2_")
    }
    initial = float(loss(model, batch).detach())
    for _ in range(3):
        trainer.step(batch, loss)
    for name, value in inactive.items():
        torch.testing.assert_close(model.values[name], value, atol=0, rtol=0)
        assert model.values[name] not in trainer.optimizer.state
    before = model(x).detach()
    model.activate_depth(2)
    torch.testing.assert_close(model(x), before, atol=0, rtol=0)
    assert ids == {name: id(p) for name, p in model.values.items()}
    trainer.step(batch, loss)
    assert int(trainer.optimizer.state[model.values["stem_weight"]]["step"]) == 4
    assert int(trainer.optimizer.state[model.values["block_1_gain_q"]]["step"]) == 1
    assert model.values["block_2_gain_q"] not in trainer.optimizer.state
    for _ in range(5):
        trainer.step(batch, loss)
    assert float(loss(model, batch).detach()) < initial
    assert any(
        not torch.equal(model.values[name], inactive[name])
        for name in inactive
        if name.startswith("block_1_")
    )
    trainer.save(tmp_path / "network.pt")
    saved = copy.deepcopy(model.state_dict())
    trainer.step(batch, loss)
    expected = {name: p.detach().clone() for name, p in model.values.items()}
    other = plan.build("torch", backend="torch", dtype=torch.float64).trainer(
        lr=0.02, microbatch_size=2
    )
    other.load(tmp_path / "network.pt")
    assert other.network.active_depth == saved["_extra_state"]["active_depth"]
    other.step(batch, loss)
    for name, value in expected.items():
        torch.testing.assert_close(other.network.values[name], value, atol=0, rtol=0)
    bad = architecture(hidden_features=5).build("torch", backend="torch").trainer()
    before = bad.network.values["stem_weight"].detach().clone()
    with pytest.raises(q.ValidationError, match="architecture"):
        bad.load(tmp_path / "network.pt")
    torch.testing.assert_close(bad.network.values["stem_weight"], before, atol=0, rtol=0)


def test_jax_stagewise_adamw_freezes_inactive_state_and_resumes_curriculum(tmp_path):
    jax, jnp = pytest.importorskip("jax"), pytest.importorskip("jax.numpy")
    optax = pytest.importorskip("optax")
    plan = architecture()
    model = plan.build("jax", backend="jax", active_depth=1)

    def loss(m, parameters, batch):
        return jnp.mean((m(parameters, batch[0]) - batch[1]) ** 2, axis=-1)

    trainer = model.trainer(
        optax.adamw(0.02, weight_decay=0.1),
        loss,
        key=jax.random.key(23),
        microbatch_size=2,
        depth_schedule=q.DepthSchedule(((0, 1), (3, 2), (7, 3))),
    )
    x = jnp.asarray([[-1.0, -0.5], [-0.4, 0.2], [0.3, 0.6], [1.0, -0.2]])
    batch = (x, x[:, :1] - 0.4 * x[:, 1:])
    before = {name: np.asarray(value).copy() for name, value in trainer.parameters.items()}
    state = jax.tree_util.tree_map(
        lambda value: np.asarray(value).copy(), trainer.optimizer_state["block_1"]
    )
    for _ in range(3):
        trainer.step(batch)
    for name, value in before.items():
        if name.startswith("block_1_") or name.startswith("block_2_"):
            np.testing.assert_array_equal(trainer.parameters[name], value)
    for old, current in zip(
        jax.tree_util.tree_leaves(state),
        jax.tree_util.tree_leaves(trainer.optimizer_state["block_1"]),
        strict=True,
    ):
        np.testing.assert_array_equal(old, current)
    np.testing.assert_array_equal(
        trainer.model(trainer.parameters, x), trainer.model.with_depth(2)(trainer.parameters, x)
    )
    metrics = trainer.step(batch)
    assert metrics["active_depth"] == 2
    assert int(trainer.optimizer_state["shared"][0].count) == 4
    assert int(trainer.optimizer_state["block_1"][0].count) == 1
    trainer.save(tmp_path / "network.npz")
    trainer.step(batch)
    expected = trainer.parameters
    other = model.trainer(
        optax.adamw(0.02, weight_decay=0.1), loss, key=jax.random.key(4), microbatch_size=2
    )
    other.load(tmp_path / "network.npz")
    other.step(batch)
    for name in expected:
        np.testing.assert_array_equal(other.parameters[name], expected[name])
    assert other.depth_schedule == trainer.depth_schedule


def test_network_scheduler_finds_mixed_devices_under_a_shared_budget(monkeypatch):
    from openqml import backends

    class Device:
        def __init__(self, name, kind, available=True):
            self.name, self.available, self.calls = name, available, []
            self.capabilities = q.BackendCapabilities(
                native_expectations=True, device_type=kind, exact_expectations=kind != "QPU"
            )

        def is_available(self):
            return self.available

        def expectations_batch(self, circuits, observables, *, options):
            self.calls.append(len(circuits))
            return q.expectations_batch(circuits, observables, backend="numpy", **options.__dict__)

    qpu, cpu = Device("architecture-qpu", "QPU"), Device("cpu", "CPU")
    for device in (qpu, cpu, Device("gpu", "GPU", False)):
        monkeypatch.setitem(backends._REGISTRY, device.name, device)
    profiles = {
        name: q.ExecutionProfile(
            seconds,
            queue_seconds=0,
            cost_per_second=0,
            noise_bias=0,
            relative_uncertainty=0,
            cost_per_job=cost,
            max_batch_size=256,
            source="synthetic_test",
        )
        for name, seconds, cost in ((qpu.name, 0.001, 4), (cpu.name, 1, 0))
    }
    ledger = q.BudgetLedger(16)
    runtime = q.CostAwareBackend(
        profiles=profiles, policy=q.SchedulingPolicy(max_cost=8), ledger=ledger
    )
    plan = architecture(depth=2, quantum=q.QuantumCellSpec(num_qubits=1))
    scheduled = plan.schedule(scheduler=runtime, workload=q.TrainingWorkload(batch_size=3))
    assert {stage.backend for stage in scheduled.stages} == {qpu.name, cpu.name}
    assert scheduled.forecast.upper_cost == 8 and scheduled.forecast.upper_seconds < 12
    assert not qpu.calls and not cpu.calls
    torch = pytest.importorskip("torch")
    model = plan.build("torch", backend=runtime, dtype=torch.float64)
    with torch.no_grad():
        for i in range(2):
            model.values[f"block_{i}_gain_q"].fill_(0.2)
    ids = {name: id(value) for name, value in model.values.items()}
    trainer = model.trainer(lr=0.001, microbatch_size=3)
    batch = (
        torch.tensor([[0.2, 0.4], [-0.1, 0.5], [0.3, -0.2]], dtype=torch.float64),
        torch.ones((3, 1), dtype=torch.float64),
    )

    def loss(network, b):
        return (network(b[0]) - b[1]).square().mean()

    trainer.step(batch, loss)
    assert qpu.calls
    assert ledger.reserved == 8
    submitted = list(qpu.calls)
    qpu.available = False
    trainer.step(batch, loss)
    assert qpu.calls == submitted
    assert ledger.reserved == 8
    assert ids == {name: id(value) for name, value in model.values.items()}
    assert int(trainer.optimizer.state[model.values["stem_weight"]]["step"]) == 2
    qpu.available = True
    impossible = q.CostAwareBackend(
        profiles=profiles, policy=q.SchedulingPolicy(max_cost=8, max_seconds=0.01)
    )
    with pytest.raises(q.ResourceError):
        plan.schedule(scheduler=impossible, workload=q.TrainingWorkload(batch_size=3))


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_integrated_native_training_reserves_shared_budget_before_updates(framework):
    package = pytest.importorskip(framework)
    ledger = q.BudgetLedger(2)
    runtime = q.CostAwareBackend(
        profiles={
            "cpu": q.ExecutionProfile(
                0.01,
                queue_seconds=0,
                cost_per_second=0,
                cost_per_circuit=0.1,
                relative_uncertainty=0,
            )
        },
        workload=q.TrainingWorkload(classical_cost_per_step=0.1),
        ledger=ledger,
    )
    plan = architecture(depth=2, quantum=q.QuantumCellSpec(num_qubits=1))
    model = plan.build(framework, backend=runtime)
    if framework == "torch":
        with pytest.raises(q.ValidationError, match="tensor meshes"):
            plan.build("torch", backend=runtime, tensor_mesh=object(), interconnect=object())
        x = package.tensor([[0.1, 0.3], [0.3, -0.1], [0.4, 0.2]])
        trainer = model.trainer(microbatch_size=3)

        def update():
            return trainer.step(
                {"target": x[:, 0], "x": x},
                lambda m, b: (m(b["x"])[:, 0] - b["target"]).square().mean(),
            )
    else:
        optax = pytest.importorskip("optax")
        jnp = pytest.importorskip("jax.numpy")
        x = jnp.asarray([[0.1, 0.3], [0.3, -0.1], [0.4, 0.2]])
        trainer = model.trainer(
            optax.adam(0.01),
            lambda m, p, b: (m(p, b["x"])[:, 0] - b["target"]) ** 2,
            key=package.random.key(7),
            microbatch_size=3,
        )

        def update():
            return trainer.step({"target": x[:, 0], "x": x})

    assert ledger.reserved == 0
    update()
    assert ledger.reserved == pytest.approx(1.9)
    assert trainer.updates == 1
    with pytest.raises(q.ResourceError, match="budget|schedule"):
        update()
    assert trainer.updates == 1 and ledger.reserved == pytest.approx(1.9)
