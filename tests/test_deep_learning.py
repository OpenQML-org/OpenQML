"""Independent quantum-channel references and actual end-to-end learning."""

import math

import numpy as np
import pytest

import openqml as q


def ry(angle):
    c, s = np.cos(angle / 2), np.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(angle):
    return np.diag(np.exp(np.array([-1j, 1j]) * angle / 2))


def channel_reference(plan, weights, inputs):
    assert plan.num_qubits == 2
    psi = np.kron(ry(inputs[0])[:, 0], ry(inputs[1])[:, 0])
    rho = np.outer(psi, psi.conj())
    values = dict(zip(plan.spec.weight_parameters, weights, strict=True))
    for block in range(plan.depth):
        gate = np.eye(4, dtype=complex)
        for direction, positions in (("f", range(5)), ("r", reversed(range(5)))):
            for index in positions:
                angle = values[f"deep_{block}_{direction}_{index}"]
                local = (
                    np.kron(ry(angle), np.eye(2))
                    if index == 0
                    else np.kron(rz(angle), np.eye(2))
                    if index == 1
                    else np.kron(np.eye(2), ry(angle))
                    if index == 2
                    else np.kron(np.eye(2), rz(angle))
                    if index == 3
                    else np.diag(np.exp(-0.5j * angle * np.array([1, -1, -1, 1])))
                )
                gate = local @ gate
        if plan.reupload:
            embedding = np.kron(rz(inputs[0]), rz(inputs[1]))
            gate = embedding.conj().T @ gate @ embedding
        p = math.sin(values[f"deep_{block}_mix"] / 2) ** 2
        rho = (1 - p) * rho + p * gate @ rho @ gate.conj().T
    return rho


@pytest.mark.parametrize("reupload", (False, True))
def test_persistent_deep_quantum_state_matches_independent_cptp_channel(reupload):
    plan = q.DeepQuantumPlan(2, 3, reupload=reupload)
    weights = np.random.default_rng(31).normal(0, 0.8, plan.num_parameters)
    inputs = [0.3, -0.6]
    bindings = dict(zip(plan.spec.weight_parameters, weights, strict=True))
    bindings.update(zip(plan.spec.input_parameters, inputs, strict=True))
    density = q.density_matrix(plan.spec.circuit, bindings, backend="numpy").reshape(2, 4, 2, 4)
    data = density[0, :, 0, :] + density[1, :, 1, :]
    np.testing.assert_allclose(data, channel_reference(plan, weights, inputs), atol=3e-14)
    assert np.trace(data) == pytest.approx(1)
    assert np.linalg.eigvalsh(data).min() >= -1e-14
    np.testing.assert_allclose(density[1, :, 1, :], 0, atol=1e-15)
    assert plan.to_dict()["postselection"] is False


def test_identity_growth_preserves_trained_quantum_state_and_named_parameters():
    plan = q.DeepQuantumPlan(2, 2)
    weights = np.random.default_rng(7).normal(0, 0.4, plan.num_parameters)
    checkpoint = q.LayerCheckpoint(plan.spec, tuple(weights))
    grown, saved = plan.grow(checkpoint, add_blocks=3, seed=13)
    assert saved.weights[: plan.num_parameters] == checkpoint.weights
    for inputs in ([0.1, -0.8], [0.5, 0.7]):
        np.testing.assert_allclose(
            channel_reference(grown, saved.weights, inputs),
            channel_reference(plan, weights, inputs),
            atol=2e-14,
        )
    assert grown.depth == 5 and grown.to_dict()["physical_qubits"] == 3
    initial = plan.initial_weights(seed=4)
    expected = np.kron(ry(0.4)[:, 0], ry(0.7)[:, 0])
    np.testing.assert_allclose(
        channel_reference(plan, initial, [0.4, 0.7]),
        np.outer(expected, expected.conj()),
        atol=2e-14,
    )
    with pytest.raises(q.ValidationError, match="exact deep"):
        grown.grow(checkpoint)


def test_deep_construction_and_simulation_admit_resources_separately():
    with pytest.raises(q.ResourceError, match="max_operations"):
        q.DeepQuantumPlan(20, 10, max_operations=10)
    with pytest.raises(q.ValidationError):
        q.DeepQuantumPlan(2, initial_mixing=0)
    plan = q.DeepQuantumPlan(20, 1)
    assert plan.to_dict()["physical_qubits"] == 21
    bindings = dict.fromkeys(plan.spec.input_parameters + plan.spec.weight_parameters, 0.1)
    with pytest.raises(q.ResourceError):
        q.expectations(
            plan.spec.circuit, plan.spec.observables, bindings, backend="numpy", max_memory_mb=1
        )


def test_torch_deep_channels_train_and_support_native_higher_derivatives():
    torch = pytest.importorskip("torch")
    torch.manual_seed(18)
    model = q.torch.QuantumDeepNetwork(
        2, 1, num_qubits=2, depth=2, backend="torch", dtype=torch.float64, device="cpu"
    )
    x = torch.tensor([[-1.0, -0.5], [-0.4, 0.2], [0.3, 0.6], [1.0, -0.2]], dtype=torch.float64)
    target = x[:, :1] - 0.4 * x[:, 1:]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    before = model.quantum.weights.detach().clone()
    initial = float((model(x) - target).square().mean().detach())
    for _ in range(18):
        optimizer.zero_grad()
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < 0.6 * initial
    assert not torch.equal(before, model.quantum.weights)
    result = model(x.reshape(2, 2, 2))
    assert result.shape == (2, 2, 1)
    layer = q.torch.QuantumLayer.deep(1, 2, backend="torch", dtype=torch.float64)
    inputs = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(layer(inputs).sum(), layer.weights, create_graph=True)[0]
    second = torch.autograd.grad(gradient.square().sum(), layer.weights)[0]
    assert torch.isfinite(second).all() and gradient.norm() > 0


def test_jax_deep_model_jit_learning_and_flax_parameter_trees():
    jax, jnp = pytest.importorskip("jax"), pytest.importorskip("jax.numpy")
    optax = pytest.importorskip("optax")
    model = q.jax.QuantumDeepNetwork(2, 1, num_qubits=2, depth=2, backend="jax")
    parameters = model.init(jax.random.key(18))
    x = jnp.asarray([[-1.0, -0.5], [-0.4, 0.2], [0.3, 0.6], [1.0, -0.2]])
    y = x[:, :1] - 0.4 * x[:, 1:]

    def objective(p, batch):
        return jnp.mean((model(p, batch[0]) - batch[1]) ** 2, axis=-1)

    optimizer = optax.adam(0.04)
    step = q.jax.make_train_step(optimizer, objective, microbatch_size=2, max_grad_norm=None)
    state = optimizer.init(parameters)
    initial = float(objective(parameters, (x, y)).mean())
    before = np.asarray(parameters["quantum"]).copy()
    for _ in range(18):
        parameters, state, _ = step(parameters, state, (x, y))
    assert float(objective(parameters, (x, y)).mean()) < 0.6 * initial
    assert not np.array_equal(before, np.asarray(parameters["quantum"]))
    assert jax.jit(model)(parameters, x.reshape(2, 2, 2)).shape == (2, 2, 1)
    pytest.importorskip("flax")
    module = q.flax.QuantumDeepNetwork(2, 1, num_qubits=2, depth=2, backend="jax")
    variables = module.init(jax.random.key(7), x)
    gradient = jax.jit(jax.grad(lambda p: jnp.sum(module.apply({"params": p}, x) ** 2)))(
        variables["params"]
    )
    assert np.linalg.norm(gradient["quantum"]["weights"]) > 0
    assert module.apply(variables, x).shape == (4, 1)
