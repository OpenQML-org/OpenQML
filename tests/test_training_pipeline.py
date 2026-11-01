import copy
from types import SimpleNamespace

import numpy as np
import pytest

import openqml as oq
from openqml.rl_training import EnvironmentBatch, PPOTrainingLoop, prepare_ppo_batch


class Bandit:
    def __init__(self, continuous=False, truncated=False):
        self.continuous, self.truncated = continuous, truncated
        self.state = np.zeros(2, dtype=np.float32)
        self.action_space = SimpleNamespace(low=np.array([-1.8]), high=np.array([2.1]))

    def reset(self, *, seed=None):
        self.state[:] = [1, 0]
        return self.state, {}

    def step(self, action):
        self.state[:] = [3, 1]
        reward = -float((action[0] - 0.2) ** 2) if self.continuous else float(action == 1)
        return self.state, reward, not self.truncated, self.truncated, {}


def test_final_observation_survives_inplace_reset_and_bootstraps_truncation():
    class Learner(PPOTrainingLoop):
        def _sample(self, observations, masks):
            size = len(observations)
            return np.zeros(size), np.zeros(size), np.zeros(size), observations[:, 0]

        def _value(self, observations):
            return observations[:, 0]

    learner = Learner()
    learner._initialize_loop([Bandit(truncated=True)], 2, 2, False, oq.PPOConfig(discount=0.9), 0)
    rollout, _ = learner.collect(2)
    np.testing.assert_array_equal(rollout["next_values"], [[3], [3]])
    np.testing.assert_array_equal(learner.environments.observations, [[1, 0]])
    # reward zero, final value 3, time truncation bootstraps but breaks traces.
    np.testing.assert_allclose(prepare_ppo_batch(rollout)["returns"], [2.7, 2.7])


def test_rollout_memory_budget_and_masks():
    class Masked(Bandit):
        def reset(self, *, seed=None):
            x, _ = super().reset(seed=seed)
            return x, {"action_mask": [False, True]}

    env = EnvironmentBatch([Masked], 2, 2, False)
    np.testing.assert_array_equal(env.masks, [[False, True]])
    with pytest.raises(oq.ValidationError):
        oq.PPOConfig(rollout_steps=0)
    with pytest.raises(oq.ValidationError):
        oq.PPOConfig(target_kl=-1)

    class EmptyMask(Masked):
        def reset(self, *, seed=None):
            x, _ = super().reset(seed=seed)
            return x, {"action_mask": [False, False]}

    with pytest.raises(oq.ValidationError, match="mask"):
        _ = EnvironmentBatch([EmptyMask], 2, 2, False).masks


def test_torch_microbatch_update_matches_full_hybrid_batch():
    torch = pytest.importorskip("torch")
    torch.manual_seed(4)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 2, dtype=torch.float64),
        oq.torch.TreeQuantumLayer(2, num_outputs=2, dtype=torch.float64),
        torch.nn.Linear(2, 1, dtype=torch.float64),
    )
    reference = copy.deepcopy(model)
    x, y = torch.randn(7, 3, dtype=torch.float64), torch.randn(7, 1, dtype=torch.float64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    direct = torch.optim.SGD(reference.parameters(), lr=0.1)
    loss = (reference(x) - y).square().mean()
    loss.backward()
    direct.step()
    trainer = oq.torch.HybridTrainer(model, optimizer, microbatch_size=3, max_grad_norm=None)
    metrics = trainer.step({"x": x, "y": y}, lambda m, b: (m(b["x"]) - b["y"]).square().mean(-1))
    assert metrics["samples"] == 7 and metrics["updates"] == 1 and not metrics["skipped"]
    assert metrics["loss"] == pytest.approx(loss.item(), abs=1e-12)
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)


def test_torch_nonfinite_rejection_and_optimizer_rng_resume(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(9)
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 1))
    trainer = oq.torch.HybridTrainer(
        model, torch.optim.Adam(model.parameters(), lr=0.01), microbatch_size=2
    )
    batch = torch.randn(5, 2)

    def objective(m, b):
        return m(b).square().mean()

    trainer.step(batch, objective)
    trainer.save(tmp_path / "training.pt")
    expected = trainer.step(batch, objective)
    weights = [p.detach().clone() for p in model.parameters()]
    trainer.load(tmp_path / "training.pt")
    resumed = trainer.step(batch, objective)
    assert resumed["loss"] == expected["loss"]
    for p, value in zip(model.parameters(), weights, strict=True):
        torch.testing.assert_close(p, value, atol=0, rtol=0)
    before = [p.detach().clone() for p in model.parameters()]
    result = trainer.step(batch, lambda m, b: m(b).mean() * float("nan"))
    assert result["skipped"] and result["updates"] == 2
    for p, value in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(p, value, atol=0, rtol=0)


@pytest.mark.parametrize("architecture", ["tree", "multiscale", "iqp"])
def test_amp_keeps_quantum_arithmetic_in_float32(architecture):
    torch = pytest.importorskip("torch")
    if architecture == "tree":
        layer = oq.torch.TreeQuantumLayer(2, num_outputs=2)
    elif architecture == "multiscale":
        layer = oq.torch.QuantumLayer.multiscale(
            2, depth=1, bond_qubits=2, backend="torch", execution_method="tensor-network"
        )
    else:
        layer = oq.torch.IQPQuantumLayer(2)
    x = torch.randn(3, 2)
    expected = layer(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = layer(x)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_jax_scan_weighted_gradients_and_auxiliary_metrics():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    optax = pytest.importorskip("optax")
    params = {"w": jnp.asarray([0.2, -0.3])}
    batch = {"x": jnp.arange(14, dtype=jnp.float32).reshape(7, 2) / 10, "y": jnp.arange(7) / 7}

    def loss(p, b):
        return (b["x"] @ p["w"] - b["y"]) ** 2

    value, gradient = jax.value_and_grad(lambda p: loss(p, batch).mean())(params)
    for size in (1, 3, 20):
        actual, grad, count = jax.jit(
            lambda p, b, size=size: oq.jax.accumulate_gradients(loss, p, b, microbatch_size=size)
        )(params, batch)
        assert int(count) == 7
        np.testing.assert_allclose(actual, value, rtol=2e-6)
        np.testing.assert_allclose(grad["w"], gradient["w"], rtol=2e-6)
    optimizer = optax.sgd(0.1)

    def auxiliary(p, b):
        return loss(p, b), {"target": b["y"]}

    step = oq.jax.make_train_step(
        optimizer, auxiliary, microbatch_size=3, max_grad_norm=None, has_aux=True
    )
    updated, _, metrics = step(params, optimizer.init(params), batch)
    np.testing.assert_allclose(updated["w"], params["w"] - 0.1 * gradient["w"], rtol=1e-6)
    np.testing.assert_allclose(metrics["target"], batch["y"].mean(), rtol=1e-6)
    bad = oq.jax.make_train_step(optimizer, lambda p, b: loss(p, b) * jnp.nan)
    updated, _, metrics = bad(params, optimizer.init(params), batch)
    assert bool(metrics["skipped"])
    np.testing.assert_array_equal(updated["w"], params["w"])


def _config(**kwargs):
    return oq.PPOConfig(
        rollout_steps=16,
        epochs=4,
        minibatch_size=64,
        microbatch_size=16,
        entropy_coefficient=0,
        target_kl=None,
        **kwargs,
    )


def test_torch_complete_ppo_learns_and_checkpoints_quantum_parameters(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(3)
    quantum = oq.torch.TreeQuantumLayer(2, num_outputs=2)
    model = oq.torch.QuantumActorCritic(
        2, 2, num_qubits=2, hidden_features=8, quantum_layer=quantum
    )
    before = quantum.weights.detach().clone()
    trainer = oq.torch.PPOTrainer(
        model, torch.optim.Adam(model.parameters(), lr=0.03), [Bandit] * 4, config=_config(), seed=7
    )
    history = trainer.train(512)
    with torch.no_grad():
        probability = model(torch.tensor([[1.0, 0.0]])).policy.softmax(-1)[0, 1].item()
    assert probability > 0.75 and trainer.timesteps == 512 and trainer.updates > 0
    assert not torch.equal(before, quantum.weights)
    assert np.isfinite(history[-1]["loss"])
    assert trainer.evaluate(Bandit, episodes=3)["mean_return"] == 1
    trainer.save(tmp_path / "policy.pt")
    expected = quantum.weights.detach().clone()
    trainer.train(64)
    trainer.load(tmp_path / "policy.pt")
    torch.testing.assert_close(quantum.weights, expected, atol=0, rtol=0)
    assert trainer.timesteps == 512
    with pytest.raises(oq.ResourceError):
        trainer.config = oq.PPOConfig(max_rollout_mb=1e-8)
        trainer.collect()


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_continuous_ppo_density_bounds_and_update(framework, tmp_path):
    if framework == "torch":
        torch = pytest.importorskip("torch")
        model = oq.torch.QuantumActorCritic(
            2,
            1,
            num_qubits=2,
            continuous=True,
            hidden_features=4,
            quantum_layer=oq.torch.TreeQuantumLayer(2, num_outputs=2),
        )
        trainer = oq.torch.PPOTrainer(
            model,
            torch.optim.Adam(model.parameters(), lr=0.001),
            [lambda: Bandit(True)],
            config=oq.PPOConfig(rollout_steps=4, epochs=1, minibatch_size=4, microbatch_size=2),
        )
    else:
        jax = pytest.importorskip("jax")
        optax = pytest.importorskip("optax")
        model = oq.jax.QuantumActorCritic(
            2,
            1,
            num_qubits=2,
            continuous=True,
            hidden_features=4,
            quantum_layer=oq.jax.TreeQuantumLayer(2, num_outputs=2),
        )
        params = model.init(jax.random.PRNGKey(2))
        trainer = oq.jax.PPOTrainer(
            model,
            params,
            optax.adam(0.001),
            [lambda: Bandit(True)],
            config=oq.PPOConfig(rollout_steps=4, epochs=1, minibatch_size=4, microbatch_size=2),
        )
    rollout, _ = trainer.collect()
    # The recorded pre-tanh values remain usable even when tanh saturates.
    raw = rollout["actions"].reshape(-1, 1)
    observations = rollout["observations"].reshape(-1, 2)
    if framework == "torch":
        with torch.no_grad():
            output = model(torch.as_tensor(observations))
            log = oq.squashed_gaussian_log_probability(
                output.policy, output.log_std, torch.as_tensor(raw, dtype=output.policy.dtype)
            )
            expected = (log - trainer.scale.log().sum()).numpy()
    else:
        output = model(trainer.parameters, observations)
        expected = np.asarray(
            oq.squashed_gaussian_log_probability(output.policy, output.log_std, raw)
            - np.log(np.asarray(trainer.scale)).sum()
        )
    np.testing.assert_allclose(expected, rollout["log_probs"].reshape(-1), rtol=2e-5, atol=2e-6)
    history = trainer.train(4)
    assert trainer.updates > 0 and np.isfinite(history[-1]["loss"])
    path = tmp_path / "continuous-checkpoint"
    trainer.save(path)
    trainer.load(path)


def test_jax_complete_ppo_learns_quantum_weights_and_restores_state(tmp_path):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    optax = pytest.importorskip("optax")
    model = oq.jax.QuantumActorCritic(
        2,
        2,
        num_qubits=2,
        hidden_features=8,
        quantum_layer=oq.jax.TreeQuantumLayer(2, num_outputs=2),
    )
    params = model.init(jax.random.PRNGKey(3))
    original = np.asarray(params["quantum"]).copy()
    trainer = oq.jax.PPOTrainer(
        model, params, optax.adam(0.03), [Bandit] * 4, config=_config(), seed=7
    )
    trainer.train(512)
    probability = jax.nn.softmax(model(trainer.parameters, jnp.asarray([[1.0, 0.0]])).policy)[0, 1]
    assert float(probability) > 0.75 and trainer.updates > 0
    assert not np.array_equal(original, np.asarray(trainer.parameters["quantum"]))
    assert trainer.evaluate(Bandit, episodes=3)["mean_return"] == 1
    trainer.save(tmp_path / "policy.npz")
    expected = jax.tree_util.tree_map(np.asarray, trainer.parameters)
    expected_optimizer = jax.tree_util.tree_map(np.asarray, trainer.optimizer_state)
    trainer.train(64)
    trainer.load(tmp_path / "policy.npz")
    for a, b in zip(
        jax.tree_util.tree_leaves(trainer.parameters),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(
        jax.tree_util.tree_leaves(trainer.optimizer_state),
        jax.tree_util.tree_leaves(expected_optimizer),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


def test_flax_ppo_uses_linen_apply():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    optax = pytest.importorskip("optax")
    model = oq.flax.QuantumActorCritic(
        2,
        2,
        num_qubits=2,
        hidden_features=4,
        quantum_layer=oq.flax.TreeQuantumLayer(2, num_outputs=2),
    )
    variables = model.init(jax.random.PRNGKey(3), jnp.ones((1, 2)))
    trainer = oq.flax.PPOTrainer(
        model,
        variables,
        optax.adam(0.01),
        [Bandit],
        config=oq.PPOConfig(rollout_steps=4, epochs=1, minibatch_size=4, microbatch_size=2),
    )
    assert trainer.train(4)[0]["updates"] == 1
