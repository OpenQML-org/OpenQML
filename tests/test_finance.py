"""Independent financial optimization, accounting and native-training contracts."""

import gc
import itertools
import subprocess
import sys

import numpy as np
import pytest

import openqml as q
from openqml import finance as f


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


def test_finance_namespace_keeps_core_install_light():
    source = (
        "import sys,openqml as q; assert q.finance.AllocationPlan; "
        "assert not {'torch','jax','flax','scipy','pandas'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", source], check=True)


def test_weighted_covariance_fractional_cvar_and_zero_mass_reference():
    samples = np.array([[-0.1, 0.2], [0.03, -0.04], [0.07, 0.01], [-0.02, -0.01]])
    p = np.array([0, 0.1, 0.2, 0.7])
    risk = f.factor_risk(samples, p, shrinkage=0.2, variance_floor=1e-6)
    mean = p @ samples
    delta = samples - mean
    covariance = (
        0.8 * delta.T @ np.diag(p) @ delta + 0.2 * np.diag(p @ (delta**2)) + np.eye(2) * 1e-6
    )
    np.testing.assert_allclose(risk.mean, mean, atol=1e-15)
    np.testing.assert_allclose(f.dense_covariance(risk), covariance, atol=1e-15)
    w = np.array([0.3, 0.7])
    np.testing.assert_allclose(f.covariance_matvec(risk, w), covariance @ w, atol=1e-15)
    np.testing.assert_allclose(f.portfolio_variance(w, risk), w @ covariance @ w, atol=1e-15)
    losses = np.array([10.0, 4.0, 2.0, -1.0])
    assert f.expected_shortfall(losses, probabilities=p, confidence=0.75) == pytest.approx(2.8)
    assert f.expected_shortfall(losses, probabilities=p, confidence=0) == pytest.approx(p @ losses)
    empirical = f.expected_shortfall(-samples @ w, probabilities=p, confidence=0.8)
    assert f.wasserstein_cvar(
        w, samples, probabilities=p, confidence=0.8, radius=0.01
    ) == pytest.approx(empirical + 0.05 * np.linalg.norm(w))


@pytest.mark.parametrize("n", [1, 2, 7, 20])
def test_sorted_cost_prox_matches_independent_convex_solver(n):
    scipy = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(41 + n)
    for _ in range(3):
        v, previous = rng.normal(size=n), rng.dirichlet(np.ones(n))
        caps = np.full(n, min(1.0, 2 / n))
        fees = rng.uniform(0, 0.1, n)
        w = f.cost_aware_projection(v, previous=previous, caps=caps, penalty=fees)

        def objective(x, v=v, fees=fees, previous=previous):
            return 0.5 * np.sum((x - v) ** 2) + fees @ abs(x - previous)

        reference = scipy.minimize(
            objective,
            np.ones(n) / n,
            bounds=[(0, c) for c in caps],
            constraints=[dict(type="eq", fun=lambda x: x.sum() - 1)],
            method="SLSQP",
            options=dict(ftol=1e-12, maxiter=1000),
        )
        assert abs(w.sum() - 1) < 2e-11
        assert np.all(w >= 0) and np.all(w <= caps)
        assert objective(w) <= reference.fun + 2e-8
    # A fully saturated face creates a flat multiplier interval.
    np.testing.assert_allclose(
        f.cost_aware_projection(np.arange(n, dtype=float), caps=1 / n), np.ones(n) / n
    )
    np.testing.assert_array_equal(f.cost_aware_projection(np.ones(n), budget=0), np.zeros(n))


@pytest.mark.parametrize("fees", [0, 0.0003, 0.1])
def test_factor_allocation_matches_independent_objective_and_kkt(fees):
    scipy = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(71)
    risk = f.factor_risk(rng.normal(size=(18, 6)) * 0.04)
    mean, previous = rng.normal(size=6) * 0.003, rng.dirichlet(np.ones(6))
    plan = f.AllocationPlan()
    result = plan.solve(mean, risk, previous=previous, caps=0.5, fees=fees)
    covariance = f.dense_covariance(risk)

    def objective(w):
        return (
            0.5 * w @ covariance @ w
            - mean @ w
            + 0.5 * plan.ridge * (w @ w)
            + 0.5 * plan.turnover * np.sum((w - previous) ** 2)
            + fees * sum(abs(w - previous))
        )

    reference = scipy.minimize(
        objective,
        np.ones(6) / 6,
        bounds=[(0, 0.5)] * 6,
        constraints=[dict(type="eq", fun=lambda w: w.sum() - 1)],
        method="SLSQP",
        options=dict(ftol=1e-13, maxiter=2000),
    )
    assert result.converged and result.residual < plan.tolerance
    assert float(result.objective) <= reference.fun + 5e-8
    assert float(result.objective) == pytest.approx(objective(result.weights), abs=1e-13)
    if fees == 0.1:
        np.testing.assert_allclose(result.weights, previous, atol=1e-10)


def test_matrix_free_factor_path_scales_without_asset_kkt(monkeypatch):
    rng = np.random.default_rng(8)
    risk = f.factor_risk(rng.normal(size=(6, 512)) * 0.01)
    real_eigen = np.linalg.eigvalsh
    real_solve = np.linalg.solve

    def eigen(matrix, *args, **kwargs):
        assert matrix.shape[-1] <= 6
        return real_eigen(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigvalsh", eigen)

    def factor_solve(matrix, *args, **kwargs):
        assert matrix.shape[-1] <= 6, "Dense asset solve used"
        return real_solve(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "solve", factor_solve)
    plan = f.AllocationPlan(turnover=0.2)
    output = plan.solve(rng.normal(size=512) * 0.0001, risk, previous=np.ones(512) / 512, caps=0.01)
    assert output.converged and abs(sum(output.weights) - 1) < 1e-9
    assert plan.resource_profile(512, 6)["solver"] == "factor_Woodbury_refinement"


def test_implicit_allocation_derivatives_and_higher_derivatives(native):
    torch, jax, jnp = native
    rng = np.random.default_rng(31)
    samples = rng.normal(size=(15, 5)) * 0.05
    risk = f.factor_risk(samples)
    mean, previous = rng.normal(size=5) * 0.001, np.ones(5) / 5
    plan = f.AllocationPlan()
    mean_t = torch.tensor(mean, requires_grad=True)
    tw = plan.solve(mean_t, risk, previous=previous, caps=0.5, fees=0.0001).weights
    grad = torch.autograd.grad(tw.square().sum(), mean_t, create_graph=True)[0]
    hessian_direction = torch.autograd.grad(grad.sum(), mean_t)[0]

    def function(mu):
        return jnp.sum(plan.solve(mu, risk, previous=previous, caps=0.5, fees=0.0001).weights ** 2)

    jgrad = jax.jit(jax.grad(function))(jnp.asarray(mean))
    jh = jax.jit(jax.grad(lambda mu: jnp.sum(jax.grad(function)(mu))))(jnp.asarray(mean))
    np.testing.assert_allclose(grad.detach(), jgrad, atol=2e-8)
    np.testing.assert_allclose(hessian_direction, jh, atol=2e-7)
    step = 1e-7
    finite = np.array(
        [
            (
                function(jnp.asarray(mean + np.eye(5)[i] * step))
                - function(jnp.asarray(mean - np.eye(5)[i] * step))
            )
            / (2 * step)
            for i in range(5)
        ]
    )
    np.testing.assert_allclose(finite, jgrad, atol=1e-5)
    direction = rng.normal(size=5)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(torch.tensor(mean), torch.tensor(direction))
        dual_weights = plan.solve(dual, risk, previous=previous, caps=0.5, fees=0.0001).weights
        tangent = torch.autograd.forward_ad.unpack_dual(dual_weights).tangent
    _, expected_tangent = jax.jvp(
        lambda mu: plan.solve(mu, risk, previous=previous, caps=0.5, fees=0.0001).weights,
        (jnp.asarray(mean),),
        (jnp.asarray(direction),),
    )
    np.testing.assert_allclose(tangent, expected_tangent, atol=2e-8)
    rr = f.factor_risk(np.broadcast_to(samples, (2, 15, 5)).copy())
    batched = jax.jit(lambda mu: plan.solve(mu, rr, previous=previous, caps=0.5, fees=0.0001))(
        jnp.asarray(np.broadcast_to(mean, (2, 5)))
    )
    np.testing.assert_allclose(batched.weights, np.broadcast_to(tw.detach(), (2, 5)), atol=2e-10)


def test_zero_probability_factor_gradients_are_finite_and_correct(native):
    torch, jax, jnp = native
    atoms = np.random.default_rng(4).normal(size=(4, 3)) * 0.01
    probabilities = np.array([0.0, 0.2, 0.3, 0.5])
    plan = f.AllocationPlan(turnover=0.5)

    def tf(p):
        risk = f.factor_risk(torch.tensor(atoms), p)
        return plan.solve(risk.mean, risk, previous=np.ones(3) / 3).weights.square().sum()

    tp = torch.tensor(probabilities, requires_grad=True)
    tg = torch.autograd.grad(tf(tp), tp)[0]

    def jf(p):
        risk = f.factor_risk(jnp.asarray(atoms), p)
        return jnp.sum(plan.solve(risk.mean, risk, previous=np.ones(3) / 3).weights ** 2)

    jg = jax.grad(jf)(jnp.asarray(probabilities))
    assert np.isfinite(jg).all()
    np.testing.assert_allclose(tg, jg, atol=1e-10)


@pytest.mark.parametrize("lag", [1, 3])
def test_self_financing_accounting_independent_reference_and_streaming(lag):
    brentq = pytest.importorskip("scipy.optimize").brentq

    rng = np.random.default_rng(82)
    returns = rng.normal(size=(9, 3)) * 0.03
    decisions = rng.dirichlet(np.ones(4), size=9)[:, :3]
    initial = np.array([0.2, 0.3, 0.1])
    book = f.SelfFinancingPlan(fees=(0.001, 0.002, 0.003), impact=0.02, execution_lag=lag)
    actual = book.rollout(returns, decisions, initial_weights=initial)
    positions, cash, wealth, costs = initial.copy(), 0.4, [], []
    for t in range(len(returns)):
        capital = sum(positions) + cash
        if t >= lag:
            target = decisions[t - lag]

            def equation(value, target=target, positions=positions, capital=capital):
                trade = value * target - positions
                return (
                    value
                    + np.sum(np.asarray(book.fees) * abs(trade) + 0.02 * trade**2 / capital)
                    - capital
                )

            value = brentq(equation, 0, capital, xtol=1e-14)
            trade = value * target - positions
            cost = np.sum(np.asarray(book.fees) * abs(trade) + 0.02 * trade**2 / capital)
            np.testing.assert_allclose(capital, value + cost, atol=2e-13)
            positions, cash = value * target, value * (1 - sum(target))
        else:
            cost = 0
        positions *= 1 + returns[t]
        wealth.append(sum(positions) + cash)
        costs.append(cost)
    np.testing.assert_allclose(actual.wealth, wealth, atol=3e-13)
    np.testing.assert_allclose(actual.costs, costs, atol=3e-13)
    left = book.rollout(returns[:4], decisions[:4], initial_weights=initial)
    right = book.rollout(returns[4:], decisions[4:], state=left.state)
    np.testing.assert_array_equal(actual.wealth, np.r_[left.wealth, right.wealth])
    changed = decisions.copy()
    changed[-1] = [0.1, 0.1, 0.1]
    np.testing.assert_array_equal(
        book.rollout(returns, changed, initial_weights=initial).wealth, actual.wealth
    )


def test_self_financing_native_gradients_and_masked_chunk_boundaries(native):
    torch, jax, jnp = native
    rng = np.random.default_rng(18)
    r = rng.normal(size=(2, 6, 3)) * 0.02
    w = rng.dirichlet(np.ones(4), size=(2, 6))[:, :, :3]
    mask = np.ones((2, 6), bool)
    mask[0, 2] = False
    book = f.SelfFinancingPlan(impact=0.03, execution_lag=2)
    tw = torch.tensor(w, requires_grad=True)
    result = book.rollout(torch.tensor(r), tw, mask=mask)
    gradient = torch.autograd.grad(result.wealth.sum(), tw)[0]

    def function(weights):
        return jnp.sum(book.rollout(jnp.asarray(r), weights, mask=mask).wealth)

    np.testing.assert_allclose(gradient, jax.jit(jax.grad(function))(jnp.asarray(w)), atol=2e-11)
    left = book.rollout(torch.tensor(r[:, :3]), tw[:, :3], mask=mask[:, :3])
    right = book.rollout(torch.tensor(r[:, 3:]), tw[:, 3:], state=left.state, mask=mask[:, 3:])
    full = torch.cat((left.wealth, right.wealth), dim=-1)
    np.testing.assert_allclose(full.detach(), result.wealth.detach(), atol=1e-14)
    chunk_gradient = torch.autograd.grad(full.sum(), tw)[0]
    np.testing.assert_allclose(chunk_gradient, gradient, atol=2e-13)


def test_bankruptcy_is_absorbing_and_trading_masks_do_not_advance_lag():
    book = f.SelfFinancingPlan(fees=0)
    r = np.array([[0.0], [-1.0], [2.0], [3.0]])
    result = book.rollout(r, np.ones_like(r))
    np.testing.assert_array_equal(result.wealth, [1, 0, 0, 0])
    np.testing.assert_array_equal(result.returns, [0, -1, 0, 0])
    r = np.array([[np.nan], [0.1], [0.2]])
    w = np.array([[np.nan], [1.0], [1.0]])
    result = book.rollout(r, w, mask=np.array([False, True, True]))
    np.testing.assert_allclose(result.wealth, [1, 1, 1.2], atol=1e-14)


def test_purging_uses_label_endpoints_and_never_future_training():
    times = np.arange(24)
    ends = times + 3
    folds = list(f.PurgedWalkForward(8, 4, gap=1, expanding=False).split(times, ends))
    assert len(folds) == 3
    for fold in folds:
        assert len(fold.train) == 6 and len(fold.purged) == 2
        assert np.all(ends[fold.train] < times[fold.test[0]])
        assert fold.train[-1] < fold.test[0] and not set(fold.train) & set(fold.test)
    date = np.datetime64("2020-01-01") + times.astype("timedelta64[D]")
    assert len(list(f.PurgedWalkForward(8, 4).split(date, date + np.timedelta64(3, "D")))) == 4


@pytest.mark.parametrize("cd", ["none", "local", "full"])
def test_quantum_cardinality_and_financial_hamiltonian_against_full_basis(cd):
    rng = np.random.default_rng(90)
    factor = rng.normal(size=(4, 4)) * 0.02
    covariance = factor.T @ factor
    mean, previous = rng.normal(size=4) * 0.01, rng.dirichlet(np.ones(4))
    selection = f.PortfolioSelection(
        mean, covariance, 2, previous=previous, fees=0.002, turnover=0.01, ridge=0.001
    )
    problem = selection.problem
    plan = problem.qaoa(depth=2, counterdiabatic=cd)
    theta = rng.normal(size=len(plan.to_spec().weight_parameters))
    state = q.statevector(plan.bound(theta))
    assert (
        sum(abs(amplitude) ** 2 for i, amplitude in enumerate(state) if i.bit_count() != 2) < 1e-24
    )
    expected = 0
    for integer in range(16):
        bits = np.array([int(q) for q in format(integer, "04b")])
        w = bits / 2
        economic = 0.5 * w @ covariance @ w - mean @ w + 0.5 * 0.001 * (w @ w)
        economic += 0.5 * 0.01 * np.sum((w - previous) ** 2) + 0.002 * sum(abs(w - previous))
        assert problem.cost(bits, require_feasible=False) == pytest.approx(economic, abs=1e-14)
        diagonal_energy = sum(
            ob.coefficient * (-1) ** sum(bits[j] for j, axis in enumerate(ob.paulis) if axis == "Z")
            for ob in plan.to_spec().observables[0].terms
        )
        assert diagonal_energy == pytest.approx(economic, abs=1e-14)
        expected += abs(state[integer]) ** 2 * economic
    assert plan.expectation(theta) == pytest.approx(expected, abs=2e-14)
    sampled = plan.sample(theta, shots=256, backend="numpy", seed=3)
    assert sampled.invalid_shots == 0 and sampled.feasible_shots == 256


def test_conditional_quantum_blocks_keep_global_cross_cost_and_cardinality():
    rng = np.random.default_rng(23)
    m = rng.normal(size=(7, 7))
    m = (m + m.T) / 2
    problem = f.CardinalityProblem(m, rng.normal(size=7), 3, 0.1)
    fixed = np.array([1, 0, 1, 0, 0, 1, 0])
    indices = (0, 1, 3, 6)
    child = problem.condition(indices, fixed)
    for chosen in itertools.combinations(range(4), child.cardinality):
        bits = np.zeros(4, int)
        bits[list(chosen)] = 1
        global_bits = problem.merge(indices, bits, fixed)
        assert child.cost(bits) == pytest.approx(problem.cost(global_bits), abs=3e-14)
    optimum = problem.exact()
    greedy = problem.refine(fixed)
    assert optimum.cost <= greedy.cost + 1e-14 <= problem.cost(fixed) + 1e-14
    all_free = problem.condition(range(7), fixed)
    assert all_free.offset == problem.offset
    sampled = problem.qaoa().decode_counts({"1110000": 4, "1111000": 3})
    assert sampled.feasible_shots == 4 and sampled.invalid_shots == 3


def test_quantum_portfolio_native_autodiff_and_existing_layer_transform(native):
    torch, jax, jnp = native
    rng = np.random.default_rng(5)
    a = rng.normal(size=(4, 4)) * 0.02
    plan = f.PortfolioSelection(rng.normal(size=4) * 0.01, a.T @ a, 2).problem.qaoa(
        depth=2, counterdiabatic="full"
    )
    theta = plan.initial_parameters() + rng.normal(size=len(plan.initial_parameters())) * 0.2
    tw = torch.tensor(theta, requires_grad=True)
    value = plan.expectation(tw)
    grad = torch.autograd.grad(value, tw)[0]
    jgrad = jax.jit(jax.grad(lambda w: plan.expectation(w)))(jnp.asarray(theta))
    np.testing.assert_allclose(grad, jgrad, atol=2e-11)
    layer = q.torch.QuantumLayer.from_spec(plan.to_spec(), backend="torch", dtype=torch.float64)
    np.testing.assert_allclose(
        layer.functional(tw, torch.empty(0, dtype=torch.float64)).detach(),
        [float(value.detach())],
        atol=2e-13,
    )
    initial = float(value.detach())
    optimizer = torch.optim.Adam([tw], lr=0.08)
    achieved = initial
    for _ in range(10):
        optimizer.zero_grad()
        loss = plan.expectation(tw)
        achieved = min(achieved, float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert achieved < initial - 1e-7


def test_full_counterdiabatic_generator_matches_independent_dense_commutator():
    from openqml.simulator import _evolve_unitary, _TensorOps

    rng = np.random.default_rng(45)
    matrix = rng.normal(size=(4, 4))
    matrix = (matrix + matrix.T) * 0.01
    problem = f.CardinalityProblem(matrix, rng.normal(size=4) * 0.01, 2)
    plan = problem.qaoa(depth=1, counterdiabatic="full", initial_mixing=0)
    assert any(k is not None for _, _, k, _ in plan._cd_pairs)
    identity = np.eye(2)
    paulis = {"X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]])}

    def word(sites):
        value = np.array([[1]], complex)
        for j in range(4):
            value = np.kron(value, paulis[sites[j]] if j in sites else identity)
        return value

    mixer = sum(
        (word({i: "X", j: "X"}) + word({i: "Y", j: "Y"})) * 0.5
        for i, j in ((0, 1), (1, 2), (2, 3), (3, 0))
    )
    energies = [
        problem.cost(np.array(list(map(int, format(i, "04b")))), require_feasible=False)
        for i in range(16)
    ]
    cost = np.diag(energies) / plan.energy_scale
    expected = 1j * (mixer @ cost - cost @ mixer)
    zero = np.zeros(len(plan.to_spec().weight_parameters))
    index = plan.to_spec().weight_parameters.index("cd_0")

    def unitary(weights):
        c = plan.bound(weights)
        return _evolve_unitary(np.eye(16, dtype=complex), c.operations, {}, 4, _TensorOps(), None).T

    epsilon = 1e-6
    direction = np.eye(len(zero))[index] * epsilon
    derivative = (unitary(zero + direction) - unitary(zero - direction)) / (2 * epsilon)
    np.testing.assert_allclose(derivative @ unitary(zero).conj().T, -1j * expected, atol=2e-9)


@pytest.mark.parametrize("model", ["recurrent", "state-attention", "tensor"])
def test_market_forecasts_use_shared_quantum_tensor_program_and_native_gradients(model, native):
    torch, jax, jnp = native
    plan = f.MarketForecastPlan(
        3, 2, depth=1, hidden_features=4, model=model, projection_policy="tensor-train"
    )
    architecture = plan.architecture()
    p = architecture.initial_parameters(8)
    for n in p:
        if "gain_" in n:
            p[n][...] = 0.2
    x = np.random.default_rng(3).normal(size=(2, 3, 3))
    target = np.ones((2, 3, 2)) * 0.002
    tm = architecture.build("torch", backend="torch", dtype=torch.float64, device="cpu")
    with torch.no_grad():
        for n, v in tm.values.items():
            v.copy_(torch.tensor(p[n]))
    output = plan.decode(tm(torch.tensor(x)))
    assert torch.all(output.quantiles[..., 1:] >= output.quantiles[..., :-1])
    loss = plan.loss(tm(torch.tensor(x)), torch.tensor(target)).sum()
    loss.backward()
    jp = jax.tree_util.tree_map(jnp.asarray, p)
    jm = architecture.build("jax", backend="jax")
    jy = plan.decode(jm(jp, jnp.asarray(x)))
    jg = jax.jit(
        jax.grad(
            lambda weights: jnp.sum(plan.loss(jm(weights, jnp.asarray(x)), jnp.asarray(target)))
        )
    )(jp)
    np.testing.assert_allclose(output.quantiles.detach(), jy.quantiles, atol=1e-11)
    for n, v in tm.values.items():
        assert v.grad is not None
        np.testing.assert_allclose(v.grad, jg[n], atol=1e-10)
    assert any(projection is not None for projection in architecture.projections.values())


def test_born_joint_scenarios_factor_decisions_and_training_connect_end_to_end(native, tmp_path):
    torch, jax, jnp = native
    atoms = np.array(
        [[-0.03, 0.01, 0.02], [0.04, -0.02, 0.03], [0.01, 0.03, -0.04], [-0.01, -0.01, -0.02]]
    )
    scenarios = f.ScenarioPlan(atoms)
    architecture = scenarios.architecture(2, depth=1, hidden_features=4)
    model = architecture.build("torch", backend="torch", dtype=torch.float64, device="cpu")
    # Keep this short optimization check independent of test ordering and the
    # framework's random initializer; a large Adam step can leave a good basin.
    with torch.no_grad():
        for name, value in architecture.initial_parameters(17, dtype="float64").items():
            model.values[name].copy_(torch.as_tensor(value, dtype=torch.float64))
    allocation = f.AllocationPlan(turnover=0.5)
    x = torch.ones((4, 3, 2), dtype=torch.float64) * 0.1
    target = torch.tensor(np.broadcast_to(atoms[1], (4, 3, 3)).copy())

    def loss(network, batch):
        probabilities = network(batch[0])
        risk = scenarios.risk(probabilities)
        weights = allocation.solve(risk.mean, risk, previous=np.ones(3) / 3, caps=0.7).weights
        return (
            scenarios.energy_score(probabilities, batch[1]) - 0.02 * (weights * batch[1]).sum(-1)
        ).mean()

    first = float(loss(model, (x, target)).detach())
    trainer = model.trainer(lr=0.003, microbatch_size=2)
    for _ in range(6):
        metrics = trainer.step((x, target), loss)
        assert not metrics["skipped"]
    assert float(loss(model, (x, target)).detach()) < first
    path = tmp_path / "financial.pt"
    trainer.save(path)
    restored = architecture.build("torch", backend="torch", dtype=torch.float64, device="cpu")
    other = restored.trainer(lr=0.003, microbatch_size=2)
    other.load(path)
    trainer.step((x, target), loss)
    other.step((x, target), loss)
    for name in model.values:
        assert torch.equal(model.values[name], restored.values[name])

    def jf(p):
        risk = scenarios.risk(p)
        weights = allocation.solve(risk.mean, risk, previous=np.ones(3) / 3).weights
        return scenarios.energy_score(p, jnp.asarray(atoms[1])) - 0.02 * jnp.sum(
            weights * jnp.asarray(atoms[1])
        )

    prob = np.array([0.1, 0.3, 0.4, 0.2])
    tp = torch.tensor(prob, requires_grad=True)
    risk = scenarios.risk(tp)
    weights = allocation.solve(risk.mean, risk, previous=np.ones(3) / 3).weights
    value = (
        scenarios.energy_score(tp, torch.tensor(atoms[1]))
        - 0.02 * (weights * torch.tensor(atoms[1])).sum()
    )
    np.testing.assert_allclose(
        torch.autograd.grad(value, tp)[0], jax.jit(jax.grad(jf))(jnp.asarray(prob)), atol=1e-10
    )


def test_path_scenarios_and_proper_score_reference():
    rng = np.random.default_rng(10)
    paths = rng.normal(size=(4, 3, 2)) * 0.01
    plan = f.ScenarioPlan(paths)
    p = np.array([0.1, 0.2, 0.3, 0.4])
    target = paths[2]
    distance = np.linalg.norm(
        paths.reshape((4, -1))[:, None] - paths.reshape((4, -1))[None, :], axis=-1
    )
    reference = (
        p @ np.linalg.norm((paths - target).reshape((4, -1)), axis=-1) - 0.5 * p @ distance @ p
    )
    assert plan.energy_score(p, target) == pytest.approx(reference, abs=1e-14)
    np.testing.assert_allclose(plan.returns(), np.prod(1 + paths, axis=1) - 1)
    np.testing.assert_array_equal(plan.returns(step=1), paths[:, 1])
    assert plan.expected_payoff(p, np.array([1, 2, 3, 4]), discount=0.98) == pytest.approx(
        0.98 * p @ np.array([1, 2, 3, 4])
    )


def test_financial_domain_failures_and_budget_admission():
    with pytest.raises(q.ValidationError):
        f.cost_aware_projection(np.ones(3), caps=0.2)
    with pytest.raises(q.ResourceError):
        f.AllocationPlan(max_memory_mb=0.001).resource_profile(1000, 20)
    with pytest.raises(q.ResourceError):
        f.AllocationPlan(max_work=1).resource_profile(20, 4)
    with pytest.raises(q.ValidationError):
        f.SelfFinancingPlan(fees=0.5, impact=0.3)
    with pytest.raises(q.ValidationError):
        f.SelfFinancingPlan().rollout(np.zeros((2, 2)), np.ones((2, 2)))
    with pytest.raises(q.ValidationError):
        f.PurgedWalkForward(4, 2).split(np.ones(8), np.ones(8)).__next__()
    with pytest.raises(q.ValidationError):
        f.ScenarioPlan(np.zeros((3, 2)))
    problem = f.CardinalityProblem(np.eye(8), np.zeros(8), 4)
    with pytest.raises(q.ResourceError):
        problem.exact(max_states=1)
    with pytest.raises(q.ResourceError):
        problem.qaoa(max_operations=2)
    with pytest.raises(q.ValidationError):
        problem.qaoa().decode_counts({"00000000": 1})


def test_masked_financial_targets_and_traced_domain_failures(native):
    torch, jax, jnp = native
    plan = f.MarketForecastPlan(2, 2)
    raw = torch.zeros((1, 2 * 5), dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0.01, float("nan")]], dtype=torch.float64)
    loss = plan.loss(raw, target, mask=torch.tensor([[True, False]])).sum()
    loss.backward()
    assert torch.isfinite(raw.grad).all()
    risk = f.factor_risk(np.eye(3) * 0.01)
    invalid = jax.jit(lambda mu: f.AllocationPlan().solve(mu, risk, caps=0.2).weights)(jnp.zeros(3))
    assert np.isnan(invalid).all()


def test_cuda_financial_solver_value_and_gradient_match_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("A physical CUDA GPU is required")
    risk = f.factor_risk(np.random.default_rng(4).normal(size=(12, 5)) * 0.01)
    mean = np.random.default_rng(5).normal(size=5) * 0.0001
    plan = f.AllocationPlan(turnover=0.1)
    outputs = []
    for device in ("cpu", "cuda"):
        mu = torch.tensor(mean, dtype=torch.float64, device=device, requires_grad=True)
        w = plan.solve(mu, risk, previous=np.ones(5) / 5, caps=0.5).weights
        gradient = torch.autograd.grad(w.square().sum(), mu)[0]
        assert w.device.type == device
        outputs.append((w.detach().cpu().numpy(), gradient.cpu().numpy()))
    np.testing.assert_allclose(outputs[0][0], outputs[1][0], atol=1e-11)
    np.testing.assert_allclose(outputs[0][1], outputs[1][1], atol=1e-10)
