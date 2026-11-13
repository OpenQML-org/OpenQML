"""Independent DA-QNM trajectories, resumable state and native error boundaries."""

import numpy as np
import pytest

import openqml as oq


def _plan():
    a, b, c, p = (oq.Parameter(n) for n in ("a", "b", "c", "phase"))
    circuit = oq.Circuit(2).ry(a, 0).cx(0, 1).rz(b, 1).ry(c, 1).with_global_phase(p)
    return oq.quantum_metric(circuit, parameters=("a", "b", "c", "phase"))


def _array(value, backend):
    return value.detach().cpu().numpy() if backend == "torch" else np.asarray(value)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_stateful_trajectory_matches_independent_dense_fisher_momentum_and_polar(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    dtype = xp.float64 if backend == "torch" else xp.float32
    plan = _plan()
    optimizer = oq.DistanceAwareQNM(
        plan,
        ((2, 2),),
        damping=0.07,
        momentum=0.6,
        mixing=0.25,
        learning_rate=0.8,
        initial_radius=0.02,
        distance_growth=0.3,
        max_step_norm=0.07,
        max_iterations=8,
        alignment_guard=False,
    )
    parameters = np.array([0.2, 0.3, -0.4, 0.7])
    initial, momentum = parameters.copy(), np.zeros(4)
    maximum = 0.0
    state = None

    def native(theta, g, carry):
        return optimizer.step(
            dict(zip(plan.parameters, theta, strict=True)), g, state=carry, backend=backend
        )

    run = native if backend == "torch" else lib.jit(native)
    for i in range(5):
        gradient = np.array([0.3 + 0.02 * i, -0.2, 0.1 - 0.01 * i, 0.0])
        bindings = dict(zip(plan.parameters, parameters, strict=True))
        fisher = 4 * plan.matrix(bindings)
        natural = np.linalg.solve(fisher + 0.07 * np.eye(4), gradient)
        momentum = 0.6 * momentum + 0.4 * natural
        matrix = momentum.reshape(2, 2)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        active = s > 1e-12 * s.max()
        polar = (u * active[None, :]) @ vh
        direction = (
            0.75 * matrix + 0.25 * np.linalg.norm(matrix) * polar / np.sqrt(active.sum())
        ).ravel()
        maximum = max(maximum, np.linalg.norm(parameters - initial))
        radius = min(0.8 * np.linalg.norm(momentum), 0.02 + 0.3 * maximum, 0.07)
        step = radius * direction / (np.linalg.norm(direction) + 1e-8)
        result = run(xp.asarray(parameters, dtype=dtype), xp.asarray(gradient, dtype=dtype), state)
        np.testing.assert_allclose(
            _array(result.natural.direction, backend), natural, rtol=3e-5, atol=2e-7
        )
        np.testing.assert_allclose(_array(result.step, backend), step, rtol=5e-5, atol=3e-7)
        np.testing.assert_allclose(
            _array(result.state.momentum, backend), momentum, rtol=4e-5, atol=3e-7
        )
        assert float(_array(result.state.max_distance, backend)) == pytest.approx(maximum, abs=1e-6)
        assert int(result.state.steps) == i + 1 and bool(result.natural.converged)
        assert np.linalg.norm(_array(result.step, backend)) <= radius + 2e-7
        parameters -= step
        # A copied serialized tensor tree resumes the same trajectory.
        state = oq.DistanceAwareState(*(xp.asarray(_array(v, backend)) for v in result.state))
        if backend == "torch":
            assert not state.momentum.requires_grad


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_momentum_restart_zero_gradient_and_state_identity_checks(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    dtype = xp.float64 if backend == "torch" else xp.float32
    plan = _plan()
    optimizer = oq.DistanceAwareQNM(plan, ((2, 2),), max_iterations=8)
    theta = xp.asarray([0.2, 0.3, -0.4, 0.7], dtype=dtype)
    gradient = xp.asarray([0.3, -0.2, 0.1, 0.0], dtype=dtype)
    bindings = dict(zip(plan.parameters, theta, strict=True))
    first = optimizer.step(bindings, gradient, backend=backend)
    reversed_state = first.state._replace(momentum=-100 * first.natural.direction)
    result = optimizer.step(bindings, gradient, state=reversed_state, backend=backend)
    assert bool(result.restarted) and float(_array(result.alignment, backend)) > 0
    np.testing.assert_allclose(
        _array(result.state.momentum, backend), _array(result.natural.direction, backend)
    )
    zero = optimizer.step(bindings, gradient * 0, state=first.state, backend=backend)
    assert np.isfinite(_array(zero.step, backend)).all() and not np.any(_array(zero.step, backend))
    other = oq.DistanceAwareQNM(plan, ((2, 2),), mixing=0.2, max_iterations=8)
    with pytest.raises(oq.ValidationError):
        other.step(bindings, gradient, state=first.state, backend=backend)
    invalid = first.state._replace(
        momentum=first.state.momentum * xp.asarray(float("nan"), dtype=dtype)
    )
    with pytest.raises(oq.ValidationError):
        optimizer.step(bindings, gradient, state=invalid, backend=backend)
    exhausted = first.state._replace(steps=xp.asarray(2_147_483_647, dtype=xp.int32))
    with pytest.raises(oq.ResourceError):
        optimizer.step(bindings, gradient, state=exhausted, backend=backend)
    if backend == "jax":
        wrong = first.state._replace(signature=first.state.signature + 1)
        raw = lib.jit(lambda s: optimizer.step(bindings, gradient, state=s, backend=backend))(wrong)
        assert not np.isfinite(raw.step).all()
        checked = oq.compile_compute(
            lambda s: optimizer.step(bindings, gradient, state=s, backend=backend), backend=backend
        )
        with pytest.raises(oq.ValidationError):
            checked(wrong)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_weighted_feature_samples_keep_shared_optimizer_coordinates(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    dtype = xp.float64 if backend == "torch" else xp.float32
    x, w = oq.Parameter("x"), oq.Parameter("w")
    plan = oq.quantum_metric(oq.Circuit(1).ry(x, 0).rz(w, 0), parameters=("w",))
    optimizer = oq.DistanceAwareQNM(plan, ((1, 1),), max_iterations=4)
    features = xp.asarray([0.2, 0.4, 0.7], dtype=dtype)
    weights = xp.asarray([0.2, 0.3, 0.5], dtype=dtype)
    gradient = xp.asarray([0.3], dtype=dtype)
    result = optimizer.step(
        {"x": features, "w": xp.asarray(0.3, dtype=dtype)},
        gradient,
        backend=backend,
        sample_weights=weights,
    )
    fisher = sum(p * np.sin(v) ** 2 for p, v in zip([0.2, 0.3, 0.5], [0.2, 0.4, 0.7], strict=True))
    assert float(_array(result.natural.direction, backend)[0]) == pytest.approx(
        0.3 / (fisher + 0.01), rel=3e-5
    )
    with pytest.raises(oq.ValidationError, match="scalar"):
        optimizer.step({"x": features, "w": features}, gradient, backend=backend)


def test_torch_compiled_state_transition_and_unchanged_previous_state():
    torch = pytest.importorskip("torch")
    if tuple(int(v) for v in torch.__version__.split(".")[:2]) < (2, 4):
        pytest.skip("Compiled native JVP transforms require supported Torch>=2.4.")
    x = oq.Parameter("x")
    plan = oq.quantum_metric(oq.Circuit(1).ry(x, 0))
    optimizer = oq.DistanceAwareQNM(plan, ((1, 1),), mixing=0, max_iterations=2)
    theta = torch.tensor([0.3], dtype=torch.float64)
    g = torch.tensor([0.2], dtype=torch.float64)
    first = optimizer.step({"x": theta[0]}, g)
    previous = first.state.momentum.clone()
    run = oq.compile_compute(
        lambda t, s: optimizer.step({"x": t[0]}, g, state=s),
        backend="torch",
        compiler="eager",
        fullgraph=True,
    )
    result = run(theta - first.step, first.state)
    assert int(result.state.steps) == 2
    torch.testing.assert_close(first.state.momentum, previous)


def test_stateful_qnm_runs_on_physical_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("A physical CUDA device is required.")
    plan = _plan()
    optimizer = oq.DistanceAwareQNM(plan, ((2, 2),), max_iterations=8)
    theta = torch.tensor([0.2, 0.3, -0.4, 0.7], dtype=torch.float64, device="cuda")
    result = optimizer.step(dict(zip(plan.parameters, theta, strict=True)), theta * 0.1)
    assert result.step.is_cuda and all(v.is_cuda for v in result.state)
