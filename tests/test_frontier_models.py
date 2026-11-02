"""Physical, differentiation, streaming and integrated training contracts."""

import gc
from dataclasses import replace

import numpy as np
import pytest

import openqml as q


@pytest.fixture
def native():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield torch, jax, jnp
    finally:
        jax.config.update("jax_enable_x64", previous)
    jax.clear_caches()
    gc.collect()


def architecture(kind, **options):
    options = dict(hidden_features=3, rematerialize=True, **options)
    if kind == "kan":
        return q.HybridArchitecture.kan(2, 1, depth=1, **options)
    if kind == "memory":
        return q.HybridArchitecture.recurrent(2, 1, depth=1, **options)
    if kind == "born":
        return q.HybridArchitecture.generative(2, 2, depth=1, **options)
    return q.HybridArchitecture.language_model(3, memory_qubits=1, depth=1, **options)


def loaded_torch(a, parameters, torch):
    model = a.build("torch", backend="torch", dtype=torch.float64, device="cpu")
    with torch.no_grad():
        for name, value in model.values.items():
            value.copy_(torch.as_tensor(parameters[name]))
    return model


def opened_parameters(a):
    values = a.initial_parameters(7)
    for name, value in values.items():
        if "gain_" in name:
            value[...] = 0.3
    return values


@pytest.mark.parametrize("memory,environment", [(1, 1), (2, 1), (1, 2)])
def test_channel_is_independent_stinespring_kraus_map(memory, environment):
    from openqml.simulator import _evolve_unitary, _TensorOps

    plan = q.QuantumChannelPlan(memory, environment)
    rng = np.random.default_rng(31)
    weights = rng.normal(size=len(plan.spec.weight_parameters))
    controls = rng.normal(size=(environment,))
    n, d, e = memory + environment, 1 << memory, 1 << environment
    a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    bound = plan.spec.circuit.bind(plan._bindings(weights, controls))
    operations = tuple(op for op in bound.operations if op.name != "reset")
    unitary = _evolve_unitary(
        np.eye(1 << n, dtype=complex), operations, {}, n, _TensorOps(), None
    ).T
    kraus = [unitary.reshape((d, e, d, e))[:, i, :, 0] for i in range(e)]
    np.testing.assert_allclose(sum(k.conj().T @ k for k in kraus), np.eye(d), atol=2e-14)
    expected = sum(k @ rho @ k.conj().T for k in kraus)
    actual = plan.apply(weights, controls, rho)
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    np.testing.assert_allclose(np.trace(actual), 1, atol=2e-14)
    assert np.linalg.eigvalsh(actual)[0] >= -1e-14
    other = np.eye(d) / d
    np.testing.assert_allclose(
        plan.apply(weights, controls, 0.3 * rho + 0.7 * other),
        0.3 * actual + 0.7 * plan.apply(weights, controls, other),
        atol=2e-14,
    )


def test_channel_scan_matches_physical_reset_unroll():
    p = q.QuantumChannelPlan()
    rng = np.random.default_rng(8)
    w, x = rng.normal(size=len(p.spec.weight_parameters)), rng.normal(size=(1, 4, 1))
    y, state = p.scan(w, x)
    for t in range(1, 5):
        circuit = p.unroll(w, x[0, :t])
        np.testing.assert_allclose(
            q.expectations(circuit, p.spec.observables), y[0, t - 1], atol=2e-14
        )
    final = q.density_matrix(circuit).reshape((2, 2, 2, 2))
    np.testing.assert_allclose(np.einsum("aibi->ab", final), state[0], atol=2e-14)


@pytest.mark.parametrize("kind", ["kan", "memory", "born", "language"])
def test_all_framework_outputs_carries_and_every_parameter_gradient(kind, native):
    torch, jax, jnp = native
    pytest.importorskip("flax")
    a = architecture(kind)
    p = opened_parameters(a)
    x = np.random.default_rng(4).normal(size=(2, 3, a.input_features))
    reference, states = a.evaluate(p, x, return_state=True)
    model = loaded_torch(a, p, torch)
    y, carry = model(torch.tensor(x), return_state=True)
    (y.square().sum()).backward()
    np.testing.assert_allclose(y.detach(), reference, atol=3e-12)
    jm = a.build("jax", backend="jax")
    jp = {n: jnp.asarray(v) for n, v in p.items()}
    jy, js = jax.jit(lambda params: jm(params, jnp.asarray(x), return_state=True))(jp)
    gradients = jax.jit(jax.grad(lambda params: jnp.sum(jm(params, jnp.asarray(x)) ** 2)))(jp)
    fm = a.build("flax", backend="jax", dtype=jnp.float64)
    fy = fm.apply({"params": jp}, jnp.asarray(x))
    fg = jax.grad(lambda params: jnp.sum(fm.apply({"params": params}, jnp.asarray(x)) ** 2))(jp)
    np.testing.assert_allclose(jy, reference, atol=3e-12)
    np.testing.assert_allclose(fy, reference, atol=3e-12)
    for name, state in states.items():
        np.testing.assert_allclose(carry[name].detach(), state, atol=3e-12)
        np.testing.assert_allclose(js[name], state, atol=3e-12)
    for name, parameter in model.values.items():
        assert parameter.grad is not None, name
        assert np.isfinite(gradients[name]).all(), name
        np.testing.assert_allclose(gradients[name], parameter.grad, atol=3e-10, rtol=2e-7)
        np.testing.assert_allclose(fg[name], gradients[name], atol=3e-10, rtol=2e-7)
    assert sum(float(jnp.linalg.norm(g)) for n, g in gradients.items() if "quantum" in n) > 1e-7
    assert q.HybridArchitecture.from_dict(a.to_dict()).fingerprint == a.fingerprint


@pytest.mark.parametrize("kind", ["memory", "language"])
@pytest.mark.parametrize("method", ["sequential", "parallel"])
def test_streaming_chunks_preserve_full_bptt_masks_and_reset(kind, method):
    torch = pytest.importorskip("torch")
    a = architecture(kind, scan=method)
    p = opened_parameters(a)
    model = loaded_torch(a, p, torch)
    x = torch.randn((2, 5, a.input_features), dtype=torch.float64)
    mask = torch.tensor([[True, True, False, True, True], [True] * 5])
    reset = torch.zeros_like(mask)
    reset[0, 3] = True
    y, state = model(dict(x=x, mask=mask, reset_mask=reset), return_state=True)
    y.square().sum().backward()
    gradients = {n: v.grad.clone() for n, v in model.values.items()}
    model.zero_grad()
    left, carry = model(
        dict(x=x[:, :2], mask=mask[:, :2], reset_mask=reset[:, :2]), return_state=True
    )
    original = {n: v.clone() for n, v in carry.items()}
    right, final = model(
        dict(x=x[:, 2:], mask=mask[:, 2:], reset_mask=reset[:, 2:]), state=carry, return_state=True
    )
    joined = torch.cat((left, right), dim=1)
    np.testing.assert_allclose(joined.detach(), y.detach(), atol=2e-12)
    joined.square().sum().backward()
    for n, v in model.values.items():
        np.testing.assert_allclose(v.grad, gradients[n], atol=3e-11)
    for n in state:
        np.testing.assert_allclose(final[n].detach(), state[n].detach(), atol=2e-12)
        assert torch.equal(carry[n], original[n])
        assert not q.detach_quantum_state(carry)[n].requires_grad


@pytest.mark.parametrize("memory,outcomes", [(1, 16), (2, 2)])
def test_povm_completeness_positive_effects_and_born_reference(memory, outcomes):
    plan = q.QuantumPOVMPlan(memory, outcomes, depolarization=0)
    rng = np.random.default_rng(61)
    frame = rng.normal(size=plan.parameter_shape)
    measurement = plan.prepare(frame)
    effects = np.einsum("kac,kbc->kab", measurement.weights, measurement.weights.conj())
    for j in range(1 << memory):
        v = measurement.inverse[:, j]
        effects[j % outcomes] += measurement.alpha * np.outer(v, v.conj())
    np.testing.assert_allclose(effects.sum(0), np.eye(1 << memory), atol=2e-14)
    assert np.linalg.eigvalsh(effects).min() >= -1e-14
    rho = np.diag(np.arange(1, (1 << memory) + 1, dtype=float))
    rho /= np.trace(rho)
    actual = plan.probabilities(frame, rho)
    np.testing.assert_allclose(actual, np.einsum("ab,kba->k", rho, effects).real, atol=1e-14)
    np.testing.assert_allclose(actual.sum(), 1, atol=2e-14)


def test_povm_second_derivatives_at_degenerate_frame(native):
    torch, jax, jnp = native
    plan = q.QuantumPOVMPlan(outcomes=3)
    frame = np.zeros(plan.parameter_shape)
    rho = np.diag([0.8, 0.2])

    def tf(v):
        return plan.probabilities(v, torch.tensor(rho), backend="torch")[0]

    actual = torch.autograd.functional.hessian(tf, torch.tensor(frame))

    def jf(v):
        return plan.probabilities(v, jnp.asarray(rho), backend="jax")[0]

    expected = jax.jit(jax.hessian(jf))(jnp.asarray(frame))
    assert np.isfinite(expected).all() and np.linalg.norm(expected) > 0
    np.testing.assert_allclose(actual, expected, atol=2e-11)


@pytest.mark.parametrize("readout", ["pauli", "basis", "povm"])
def test_associative_scan_values_gradients_and_hessians(readout, native):
    torch, jax, jnp = native
    plan = q.QuantumChannelPlan(memory_qubits=2, num_outputs=3)
    rng = np.random.default_rng(12)
    data = rng.normal(size=len(plan.spec.weight_parameters))
    x = rng.normal(size=(2, 5, 1))
    mask = np.ones((2, 5), bool)
    mask[0, 2] = False
    reset = np.zeros_like(mask)
    reset[1, 3] = True
    kwargs = dict(mask=mask, reset_mask=reset, mixing=rng.uniform(size=(2, 5)))
    frame = rng.normal(size=q.QuantumPOVMPlan(2, 8).parameter_shape)

    def settings(backend, like):
        opts = dict(kwargs)
        if readout == "basis":
            opts["visible_qubits"] = 2
        if readout == "povm":
            opts["povm"] = q.QuantumPOVMPlan(2, 8).prepare(frame, backend=backend, like=like)
        return opts

    reference = plan.scan(data, x, **settings("numpy", data))
    for method in ("sequential", "parallel"):
        tw = torch.tensor(data, requires_grad=True)
        ty, ts = plan.scan(
            tw, torch.tensor(x), backend="torch", method=method, **settings("torch", tw)
        )
        gradients = torch.autograd.grad(
            ty.square().sum() + ts.real.square().sum(), tw, create_graph=True
        )[0]
        second = torch.autograd.grad(gradients.sum(), tw)[0]

        def function(w, method=method):
            y, state = plan.scan(
                w, jnp.asarray(x), backend="jax", method=method, **settings("jax", w)
            )
            return jnp.sum(y**2) + jnp.sum(state.real**2)

        jw = jnp.asarray(data)
        jg = jax.jit(jax.grad(function))(jw)
        jh = jax.jit(jax.grad(lambda w: jnp.sum(jax.grad(function)(w))))(jw)
        np.testing.assert_allclose(ty.detach(), reference[0], atol=3e-12)
        np.testing.assert_allclose(ts.detach(), reference[1], atol=3e-12)
        np.testing.assert_allclose(gradients.detach(), jg, atol=2e-10)
        np.testing.assert_allclose(second, jh, atol=2e-9)
        if method == "sequential":
            baseline = gradients.detach().numpy()
        else:
            np.testing.assert_allclose(gradients.detach(), baseline, atol=2e-10)


def test_born_fwht_matches_independent_statevector_marginal():
    from openqml.simulator import _evolve_unitary, _TensorOps

    p = q.BornCircuitPlan(3, 2)
    rng = np.random.default_rng(2)
    bindings = {name: rng.normal() for name in p.spec.circuit.parameters}
    initial = np.zeros(8, complex)
    initial[0] = 1
    state = _evolve_unitary(initial, p.spec.circuit.operations, bindings, 3, _TensorOps(), None)
    expected = (abs(state) ** 2).reshape((4, 2)).sum(-1)
    moments = q.expectations(p.spec.circuit, p.spec.observables, bindings)
    np.testing.assert_allclose(
        q.born_probabilities(moments, depolarization=0), expected, atol=2e-14
    )


@pytest.mark.parametrize("kind", ["kan", "memory", "born", "language"])
def test_real_training_depth_growth_and_exact_checkpoint_resume(kind, tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(5)
    a = architecture(kind)
    model = a.build("torch", dtype=torch.float64, backend="torch", device="cpu")
    x = torch.randn((4, 3, a.input_features), dtype=torch.float64)
    y = torch.zeros((4, 3), dtype=torch.long)
    target = torch.full((4, 3, 1), 0.6, dtype=torch.float64)

    def loss(m, batch):
        inputs, targets = batch
        p = m(inputs)
        return (
            q.born_nll(p, targets, backend="torch")
            if a.has_terminal
            else (p - targets).square().mean()
        )

    batch = (x, y if a.has_terminal else target)
    trainer = model.trainer(
        lr=0.03, microbatch_size=2, depth_schedule=q.DepthSchedule(((0, 0), (2, 1)))
    )
    before = float(loss(model, batch).detach())
    identities = {n: id(p) for n, p in model.values.items()}
    for _ in range(10):
        result = trainer.step(batch, loss)
        assert not result["skipped"]
    assert float(loss(model, batch).detach()) < before
    assert model.active_depth == 1 and identities == {n: id(p) for n, p in model.values.items()}
    path = tmp_path / "model.pt"
    trainer.save(path)
    restored = a.build("torch", dtype=torch.float64, backend="torch", device="cpu")
    second = restored.trainer(
        lr=0.03, microbatch_size=2, depth_schedule=q.DepthSchedule(((0, 0), (2, 1)))
    )
    second.load(path)
    trainer.step(batch, loss)
    second.step(batch, loss)
    for name in model.values:
        assert torch.equal(model.values[name], restored.values[name]), name


def test_channel_layers_and_quantum_diffusion_training(native):
    torch, jax, jnp = native
    from openqml.interfaces.flax import ChannelLayer as FlaxChannel
    from openqml.interfaces.jax import ChannelLayer as JAXChannel
    from openqml.interfaces.torch import ChannelLayer

    plan = q.QuantumChannelPlan()
    layer = ChannelLayer(plan, dtype=torch.float64, device="cpu")
    weights = layer.weights.detach().numpy()
    rho = np.broadcast_to(np.eye(2) / 2, (3, 2, 2)).copy()
    controls = np.ones((3, 1)) * 0.7
    expected = layer(torch.tensor(rho), torch.tensor(controls))
    jl = JAXChannel(plan)
    np.testing.assert_allclose(
        jl(jnp.asarray(weights), jnp.asarray(rho), jnp.asarray(controls)),
        expected.detach(),
        atol=2e-13,
    )
    fl = FlaxChannel(plan=plan, dtype=jnp.float64)
    np.testing.assert_allclose(
        fl.apply(
            {"params": {"weights": jnp.asarray(weights)}}, jnp.asarray(rho), jnp.asarray(controls)
        ),
        expected.detach(),
        atol=2e-13,
    )
    process = q.QuantumDiffusionPlan(plan, steps=3)
    target = torch.tensor(np.broadcast_to(np.diag([0.8, 0.2]), (3, 2, 2)).copy())
    times = torch.tensor([1, 2, 3])

    def denoise(rho, t):
        return layer(rho, layer.time_controls(t))

    optimizer = torch.optim.Adam(layer.parameters(), lr=0.06)
    losses = []
    for _ in range(12):
        optimizer.zero_grad()
        loss = process.loss(denoise, target, times, backend="torch", like=layer.weights).mean()
        losses.append(float(loss.detach()))
        loss.backward()
        assert torch.isfinite(layer.weights.grad).all()
        optimizer.step()
    assert losses[-1] < losses[0]
    sampled = process.sample(denoise, 2, backend="torch", like=layer.weights).detach().numpy()
    np.testing.assert_allclose(np.trace(sampled, axis1=-1, axis2=-2), 1, atol=2e-13)
    assert np.linalg.eigvalsh(sampled).min() > -1e-13


def test_streaming_generation_has_no_prefix_replay_and_uses_local_rng(monkeypatch, native):
    torch, jax, jnp = native
    a = architecture("language")
    p = opened_parameters(a)
    model = loaded_torch(a, p, torch)
    prompt = torch.tensor([[1, 2, 3]])
    calls, original = [], q.QuantumChannelPlan._step

    def step(self, *args):
        calls.append(1)
        return original(self, *args)

    monkeypatch.setattr(q.QuantumChannelPlan, "_step", step)
    rng = torch.get_rng_state().clone()
    tokens, state = model.sample_tokens(5, prompt=prompt, seed=8)
    assert tokens.shape == (1, 5) and len(state) == 2
    assert len(calls) == (3 + 5) * 2
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(tokens, model.sample_tokens(5, prompt=prompt, seed=8)[0])
    jm = a.build("jax", backend="jax")
    jp = jax.tree_util.tree_map(jnp.asarray, p)
    result = jm.sample_tokens(jp, jax.random.key(8), 5, prompt=jnp.asarray(prompt.numpy()))[0]
    assert result.shape == (1, 5) and np.all((np.asarray(result) >= 0) & (np.asarray(result) < 8))
    np.testing.assert_array_equal(
        result, jm.sample_tokens(jp, jax.random.key(8), 5, prompt=jnp.asarray(prompt.numpy()))[0]
    )


def test_native_carry_admission_precedes_qpu_probe_and_pareto_pruning(monkeypatch):
    from openqml import backends

    class QPU:
        name = "unusable-memory-qpu"
        capabilities = q.BackendCapabilities(device_type="QPU", native_expectations=True)

        def is_available(self):
            raise AssertionError("A QPU cannot receive the native density carry.")

    monkeypatch.setitem(backends._REGISTRY, QPU.name, QPU())
    a = architecture("language")
    scheduler = q.CostAwareBackend()
    plan = a.schedule(
        scheduler=scheduler,
        active_depth=0,
        native_devices=("CPU",),
        metadata=a.input_metadata(np.zeros((2, 4, a.input_features))),
    )
    assert plan.stage_indices == (1,) and len(plan.stages) == 1
    assert plan.stages[0].device == "CPU"
    excluded = next(c for c in plan.stages[0].candidates if c.backend == QPU.name)
    assert not excluded.eligible
    assert a.execution_batches(a.input_metadata(np.zeros((2, 4, a.input_features))), 2, 1) == (
        2,
        2,
        2,
        2,
    )


def test_masks_teacher_forcing_domains_and_budget_guards(native):
    torch, jax, jnp = native
    ids = np.array([[2, 1, -1, 3]])
    mask = np.array([[True, True, False, True]])
    reset = np.array([[False, False, False, True]])
    features = q.shifted_tokens(ids, 4, mask=mask, reset_mask=reset)
    assert features[0, 0].sum() == 0 and features[0, 1, 2] == 1
    assert not features[0, 2:].any()
    p = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]], requires_grad=True)
    loss = q.born_nll(
        p, torch.tensor([[1, -1]]), backend="torch", mask=torch.tensor([[True, False]])
    )
    loss.backward()
    assert torch.isfinite(p.grad).all()
    plan = q.QuantumChannelPlan()
    weights = np.zeros(len(plan.spec.weight_parameters))
    with pytest.raises(q.ValidationError):
        plan.apply(weights, np.ones(1), np.diag([1.2, -0.2]))
    with pytest.raises(q.ValidationError):
        plan.scan(weights, np.ones((1, 2, 1)), mixing=np.full((1, 2), 1.1))
    with pytest.raises(q.ResourceError):
        plan.scan(weights, np.ones((1, 5, 1)), method="parallel", max_work=1)
    with pytest.raises(q.ResourceError):
        replace(plan, max_memory_mb=0.0001).scan(weights, np.ones((1, 2, 1)))
    with pytest.raises(q.ResourceError):
        q.QuantumPOVMPlan(memory_qubits=1_000_000)
    invalid = jax.jit(
        lambda rho: plan.apply(jnp.asarray(weights), jnp.ones(1), rho, backend="jax")
    )(jnp.diag(jnp.array([1.2, -0.2])))
    assert np.isnan(invalid).all()


def test_flax_mixed_program_training_growth_and_checkpoint(native, tmp_path):
    _, jax, jnp = native
    pytest.importorskip("flax")
    optax = pytest.importorskip("optax")
    a = q.HybridArchitecture(
        8,
        8,
        hidden_features=3,
        depth=3,
        quantum=(
            q.QuantumCellSpec(kind="activation", num_qubits=1, depth=2),
            q.QuantumCellSpec(kind="memory", num_qubits=2, memory_qubits=1),
            q.QuantumCellSpec(kind="memory", num_qubits=2, memory_qubits=1),
        ),
        operators=(
            q.OperatorSpec.quantum_kan(),
            q.OperatorSpec.memory(),
            q.OperatorSpec.born_readout(3, memory=True, measurement="povm"),
        ),
    )
    model = a.build("flax", backend="jax", dtype=jnp.float64)
    ids = jnp.asarray([[0, 1, 2], [3, 4, 5], [1, 2, 3], [2, 3, 4]])
    x = q.shifted_tokens(ids, 8, backend="jax", like=jnp.zeros((), dtype=jnp.float64))
    variables = model.init(jax.random.key(41), x)

    def loss(network, params, batch):
        return q.born_nll(
            network(params, batch[0]), batch[1], backend="jax", reduction="none"
        ).mean(axis=-1)

    options = dict(depth_schedule=q.DepthSchedule(((0, 0), (1, 2))), microbatch_size=2)
    trainer = model.trainer(optax.adam(0.02), loss, variables=variables, **options)
    for _ in range(4):
        metrics = trainer.step((x, ids))
        assert not bool(metrics["skipped"])
    assert trainer.model.active_depth == 2
    checkpoint = tmp_path / "mixed.npz"
    trainer.save(checkpoint)
    restored = model.trainer(optax.adam(0.02), loss, variables=variables, **options)
    restored.load(checkpoint)
    trainer.step((x, ids))
    restored.step((x, ids))
    for name, value in trainer.parameters.items():
        np.testing.assert_array_equal(value, restored.parameters[name])
    generated, carry = model.sample_tokens(trainer.variables, jax.random.key(2), 3)
    assert generated.shape == (1, 3) and set(carry) == {"block_1", "block_2"}
