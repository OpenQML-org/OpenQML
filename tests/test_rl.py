"""Numerical targets, gradient ownership and actual quantum policy updates."""

import math

import numpy as np
import pytest

import openqml as oq


def trace_reference(r, v, nv, term, trunc, gamma, decay):
    result, carry = np.empty_like(r), np.zeros_like(r[0])
    for t in reversed(range(len(r))):
        carry = (
            r[t]
            + gamma * (~term[t]) * nv[t]
            - v[t]
            + gamma * decay * (~(term[t] | trunc[t])) * carry
        )
        result[t] = carry
    return result


@pytest.mark.parametrize("length", (1, 2, 3, 7, 64, 129))
def test_parallel_gae_matches_independent_reverse_recursion(length):
    rng = np.random.default_rng(length)
    r, v, nv = rng.normal(size=(3, length, 3))
    term, trunc = rng.random((length, 3)) < 0.1, rng.random((length, 3)) < 0.2
    result = oq.generalized_advantages(
        r, v, nv, term, truncated=trunc, discount=0.93, trace_decay=0.87
    )
    reference = trace_reference(r, v, nv, term, trunc, 0.93, 0.87)
    np.testing.assert_allclose(result.advantages, reference, atol=2e-14)
    np.testing.assert_allclose(result.returns, reference + v, atol=2e-14)


def test_episode_time_limit_bootstraps_final_observation_without_leaking_reset():
    args = [np.array([1.0, 100.0]), np.zeros(2), np.array([8.0, 0.0]), np.array([False, True])]
    result = oq.generalized_advantages(*args, truncated=np.array([True, False]), discount=0.5)
    np.testing.assert_allclose(result.returns, [5, 100])
    args[3][0] = True
    np.testing.assert_allclose(oq.generalized_advantages(*args).returns, [1, 100])


def test_vtrace_matches_importance_weighted_reverse_recursion_and_boundaries():
    rng = np.random.default_rng(19)
    r, v, nv = rng.normal(size=(3, 17, 2))
    term, trunc = rng.random((17, 2)) < 0.2, rng.random((17, 2)) < 0.2
    target, behavior = rng.normal(size=(2, 17, 2))
    ratio = np.exp(target - behavior)
    corrected, carry = np.empty_like(v), np.zeros(2)
    for t in reversed(range(len(v))):
        delta = min_rho = np.minimum(ratio[t], 1.5)
        delta = min_rho * (r[t] + 0.9 * (~term[t]) * nv[t] - v[t])
        carry = delta + 0.9 * np.minimum(ratio[t], 0.7) * (~(term[t] | trunc[t])) * carry
        corrected[t] = v[t] + carry
    next_error = np.concatenate((corrected[1:] - v[1:], np.zeros((1, 2))))
    advantage = np.minimum(ratio, 1.2) * (
        r + 0.9 * (~term) * (nv + (~(term | trunc)) * next_error) - v
    )
    result = oq.vtrace_targets(
        r,
        v,
        nv,
        term,
        target,
        behavior,
        truncated=trunc,
        discount=0.9,
        rho_clip=1.5,
        c_clip=0.7,
        policy_rho_clip=1.2,
    )
    np.testing.assert_allclose(result.values, corrected, atol=2e-14)
    np.testing.assert_allclose(result.advantages, advantage, atol=2e-14)
    # In the on-policy case, V-trace equals lambda=1 GAE.
    on_policy = oq.vtrace_targets(r, v, nv, term, target, target, truncated=trunc, discount=0.9)
    gae = oq.generalized_advantages(r, v, nv, term, truncated=trunc, discount=0.9, trace_decay=1)
    np.testing.assert_allclose(on_policy.values, gae.returns)


def test_masked_categorical_extremes_and_invalid_rows():
    logits = np.array([[10000.0, 10001.0, -10000.0], [5.0, 6.0, 7.0], [0.0, 0.0, 0.0]])
    mask = np.array([[True, True, False], [False, True, False], [False, False, False]])
    stats = oq.categorical_statistics(logits, np.array([1, 1, 0]), action_mask=mask)
    np.testing.assert_allclose(stats.log_probability[:2], [-np.log1p(np.exp(-1)), 0], atol=1e-14)
    assert stats.entropy[1] == 0 and stats.entropy[2] == 0
    assert stats.valid.tolist() == [True, True, False]
    assert np.isnan(stats.log_probability[2])
    for actions in (np.array([-1, 3, 0]), np.array([2, 0, 0])):
        assert not oq.categorical_statistics(logits, actions, action_mask=mask).valid.any()


def test_ppo_clipping_values_and_metrics_match_hand_calculation():
    ratio = np.array([1.5, 0.5, 1.1, 0.9])
    advantage = np.array([2.0, -2.0, 1.0, -1.0])
    values, returns = np.array([1.0, 3.0, 2.0, 0.0]), np.ones(4)
    result = oq.ppo_loss(
        np.log(ratio),
        np.zeros(4),
        advantage,
        values,
        returns,
        entropy=np.ones(4) * 0.7,
        old_values=np.zeros(4),
        value_clip=0.2,
        normalize_advantages=False,
        entropy_coefficient=0.03,
    )
    policy = -np.mean([2.4, -1.6, 1.1, -0.9])
    value = 0.5 * np.mean(
        np.maximum((values - returns) ** 2, (np.clip(values, -0.2, 0.2) - returns) ** 2)
    )
    assert result.policy_loss == pytest.approx(policy)
    assert result.value_loss == pytest.approx(value)
    assert result.loss == pytest.approx(policy + 0.5 * value - 0.03 * 0.7)
    assert result.clip_fraction == 0.5
    assert result.approximate_kl == pytest.approx(np.mean(ratio - 1 - np.log(ratio)))


def test_squashed_gaussian_log_density_and_extreme_stability():
    mean = np.array([[0.1, -0.5], [50.0, -50.0]])
    std = np.full_like(mean, -0.4)
    noise = np.array([[0.3, -0.2], [0.0, 0.0]])
    actions, logp = oq.squashed_gaussian(mean, std, noise)
    latent = mean + np.exp(std) * noise
    reference = np.sum(
        -0.5 * (noise[0] ** 2 + math.log(2 * math.pi))
        - std[0]
        - np.log1p(-(np.tanh(latent[0]) ** 2))
    )
    assert logp[0] == pytest.approx(reference)
    assert np.isfinite(logp).all() and np.isfinite(actions).all()


def test_sac_targets_twin_minimum_and_temperature():
    shape = np.ones(3)
    result = oq.sac_loss(
        -shape,
        shape * 3,
        shape * 2,
        shape,
        shape * 1.5,
        shape * 0.2,
        shape * 5,
        shape * 4,
        -shape * 2,
        np.array([False, True, False]),
        np.array(math.log(0.1)),
        discount=0.9,
        target_entropy=-2,
    )
    target = np.array([0.2 + 0.9 * 4.2, 0.2, 0.2 + 0.9 * 4.2])
    assert result.actor_loss == pytest.approx(-2.1)
    np.testing.assert_allclose(result.td_error, np.stack((shape - target, shape * 1.5 - target)))
    assert result.temperature_loss == pytest.approx(3 * math.log(0.1))


def test_torch_targets_detach_rollouts_and_ppo_has_correct_gradient_ownership():
    torch = pytest.importorskip("torch")
    r = torch.randn(7, 2, requires_grad=True)
    v, nv = torch.randn_like(r, requires_grad=True), torch.randn_like(r, requires_grad=True)
    term = torch.zeros_like(r, dtype=torch.bool)
    targets = oq.generalized_advantages(r, v, nv, term)
    assert not targets.returns.requires_grad
    trace = oq.vtrace_targets(r, v, nv, term, r, r)
    assert not trace.values.requires_grad
    old = torch.randn(7, 2, requires_grad=True)
    current = old.detach().clone().requires_grad_()
    result = oq.ppo_loss(current, old, r, v, nv)
    result.loss.backward()
    assert old.grad is None and r.grad is None and nv.grad is None
    assert current.grad is not None and v.grad is not None
    compiled = torch.compile(
        lambda a, b: oq.generalized_advantages(a, b, b, term).returns,
        backend="eager",
        fullgraph=True,
    )
    torch.testing.assert_close(
        compiled(r, v),
        targets.returns if torch.equal(v, nv) else oq.generalized_advantages(r, v, v, term).returns,
    )


def test_torch_sac_gradient_boundaries_and_tanh_reparameterization():
    torch = pytest.importorskip("torch")
    mean = torch.zeros(4, 2, dtype=torch.float64, requires_grad=True)
    std = torch.full_like(mean, -0.5, requires_grad=True)
    noise = torch.randn_like(mean)
    assert torch.autograd.gradcheck(lambda m, s: oq.squashed_gaussian(m, s, noise), (mean, std))
    actions, lp = oq.squashed_gaussian(mean, std, noise)
    critic = torch.ones(4, requires_grad=True)
    target = torch.randn(4, requires_grad=True)
    alpha = torch.tensor(-2.0, requires_grad=True)
    result = oq.sac_loss(
        lp,
        actions.sum(-1),
        actions.sum(-1) + 1,
        critic,
        critic,
        target,
        target,
        target,
        target,
        torch.zeros(4, dtype=torch.bool),
        alpha,
        target_entropy=-2,
    )
    result.actor_loss.backward(retain_graph=True)
    assert (
        mean.grad is not None and alpha.grad is None and critic.grad is None and target.grad is None
    )
    result.critic_loss.backward(retain_graph=True)
    assert critic.grad is not None and target.grad is None
    result.temperature_loss.backward()
    assert alpha.grad is not None


@pytest.mark.parametrize("structured", (False, True))
def test_torch_quantum_actor_learns_contextual_bandit_with_ordinary_adam(structured):
    torch = pytest.importorskip("torch")
    torch.manual_seed(51)
    quantum = oq.torch.IQPQuantumLayer(3, device="cpu") if structured else None
    model = oq.torch.QuantumActorCritic(
        2,
        2,
        num_qubits=3,
        depth=1,
        hidden_features=8,
        quantum_layer=quantum,
        backend="torch",
        device="cpu",
    )
    x = torch.tensor([[-1.0, -0.3], [-0.8, 0.2], [1.0, -0.2], [0.9, 0.1]])
    action = torch.tensor([0, 0, 1, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.035)
    before = None
    for step in range(35):
        optimizer.zero_grad()
        output = model(x)
        stats = oq.categorical_statistics(output.policy, action)
        if before is None:
            before = stats.log_probability.detach().mean()
        loss = oq.ppo_loss(
            stats.log_probability,
            stats.log_probability.detach(),
            torch.ones(4),
            output.value,
            torch.ones(4),
            normalize_advantages=False,
        ).loss
        loss.backward()
        if step == 0:
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
            )
        optimizer.step()
    assert oq.categorical_statistics(model(x).policy, action).log_probability.mean() > before + 0.4
    assert torch.equal(model(x).policy.argmax(-1), action)


def test_torch_surrogate_fits_without_quantum_calls_and_freezes_response_parameters():
    torch = pytest.importorskip("torch")

    class Measured(torch.nn.Module):
        calls = 0

        def forward(self, x):
            self.calls += 1
            return x.cos()

    measured, network = (
        Measured(),
        torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)),
    )
    layer = oq.torch.QuantumSurrogate(measured, network)
    controls = torch.randn(5, 2, requires_grad=True)
    targets = layer(controls).detach().requires_grad_()
    assert measured.calls == 1
    layer.fit_loss(controls, targets).backward()
    assert controls.grad is None and targets.grad is None
    assert all(p.grad is not None for p in network.parameters())
    network.zero_grad(set_to_none=True)
    layer(controls, mode="surrogate").sum().backward()
    assert measured.calls == 1 and controls.grad is not None
    assert all(p.grad is None for p in network.parameters())
    actual = layer(controls, mode="straight_through")
    torch.testing.assert_close(actual, controls.cos())
    assert measured.calls == 2


def test_jax_compiled_targets_losses_and_optax_policy_training():
    jax = pytest.importorskip("jax")
    jnp, optax = pytest.importorskip("jax.numpy"), pytest.importorskip("optax")
    r = jnp.arange(14.0, dtype=jnp.float32).reshape(7, 2)
    term = jnp.zeros_like(r, dtype=bool)
    result = jax.jit(lambda r: oq.generalized_advantages(r, r * 0.1, r * 0.2, term).returns)(r)
    expected = oq.generalized_advantages(
        np.asarray(r), np.asarray(r) * 0.1, np.asarray(r) * 0.2, np.asarray(term)
    ).returns
    np.testing.assert_allclose(result, expected, atol=1e-5)
    assert not np.any(jax.grad(lambda r: oq.generalized_advantages(r, r, r, term).returns.sum())(r))
    vt = jax.jit(lambda x: oq.vtrace_targets(x, x * 0.1, x * 0.2, term, x, x).values)(r)
    np.testing.assert_allclose(
        vt,
        oq.vtrace_targets(
            np.asarray(r),
            np.asarray(r) * 0.1,
            np.asarray(r) * 0.2,
            np.asarray(term),
            np.asarray(r),
            np.asarray(r),
        ).values,
        rtol=1e-5,
    )
    model = oq.jax.QuantumActorCritic(
        2, 2, num_qubits=3, quantum_layer=oq.jax.IQPQuantumLayer(3), hidden_features=8
    )
    parameters = model.init(jax.random.key(6))
    x, actions = (
        jnp.array([[-1.0, -0.3], [-0.8, 0.2], [1.0, -0.2], [0.9, 0.1]]),
        jnp.array([0, 0, 1, 1]),
    )

    def loss(p):
        out = model(p, x)
        stats = oq.categorical_statistics(out.policy, actions)
        return oq.ppo_loss(
            stats.log_probability,
            jax.lax.stop_gradient(stats.log_probability),
            jnp.ones(4),
            out.value,
            jnp.ones(4),
            normalize_advantages=False,
        ).loss

    optimizer = optax.adam(0.035)
    state = optimizer.init(parameters)

    @jax.jit
    def update(p, s):
        value, g = jax.value_and_grad(loss)(p)
        changes, s = optimizer.update(g, s, p)
        return optax.apply_updates(p, changes), s, value

    start = float(loss(parameters))
    initial_logp = float(
        oq.categorical_statistics(model(parameters, x).policy, actions).log_probability.mean()
    )
    for _ in range(40):
        parameters, state, _ = update(parameters, state)
    assert float(loss(parameters)) < start
    assert (
        float(
            oq.categorical_statistics(model(parameters, x).policy, actions).log_probability.mean()
        )
        > initial_logp + 0.4
    )
    np.testing.assert_array_equal(jnp.argmax(model(parameters, x).policy, -1), actions)
    leaves = jax.tree.leaves(jax.grad(loss)(parameters))
    assert all(np.isfinite(np.asarray(v)).all() for v in leaves)


def test_jax_continuous_policy_sac_surrogate_and_flax_parameter_ownership():
    jax, jnp = pytest.importorskip("jax"), pytest.importorskip("jax.numpy")
    x = jnp.ones((4, 2)) * 0.2
    actor = oq.jax.QuantumActorCritic(2, 2, num_qubits=2, continuous=True, backend="jax", depth=1)
    p = actor.init(jax.random.key(7))

    def loss(p):
        out = actor(p, x)
        a, lp = oq.squashed_gaussian(out.policy, out.log_std, jnp.ones_like(out.policy))
        return oq.sac_loss(
            lp,
            a.sum(-1),
            a.sum(-1),
            out.value,
            out.value,
            jnp.ones(4),
            out.value,
            out.value,
            lp,
            jnp.zeros(4, dtype=bool),
            jnp.array(-2.0),
            target_entropy=-2,
        ).loss

    value, g = jax.jit(jax.value_and_grad(loss))(p)
    assert np.isfinite(value) and all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(g))
    surrogate = oq.jax.QuantumSurrogate(jnp.cos, lambda p, x: x @ p)
    weights = jnp.eye(2) * 0.3
    assert not np.any(jax.grad(lambda w: surrogate(w, x, mode="surrogate").sum())(weights))
    np.testing.assert_allclose(surrogate(weights, x, mode="straight_through"), np.cos(x))
    pytest.importorskip("flax")
    module = oq.flax.QuantumActorCritic(2, 2, num_qubits=2, depth=1, continuous=True, backend="jax")
    variables = module.init(jax.random.key(9), x)
    out = jax.jit(module.apply)(variables, x)
    assert out.policy.shape == (4, 2) and out.log_std.shape == (4, 2)
    grads = jax.grad(lambda p: sum(v.sum() for v in module.apply(p, x)))(variables)
    assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grads))


@pytest.mark.parametrize("control", ({"discount": 1.1}, {"trace_decay": -0.1}))
def test_invalid_trace_controls(control):
    with pytest.raises(oq.ValidationError):
        oq.generalized_advantages(
            np.ones(2), np.zeros(2), np.zeros(2), np.zeros(2, dtype=bool), **control
        )
