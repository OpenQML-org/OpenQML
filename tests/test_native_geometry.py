"""Native state geometry matches an independent CPU adjoint metric and derivatives."""

import numpy as np
import pytest

import openqml as oq


def _plan():
    a, b, phase = (oq.Parameter(n) for n in ("a", "b", "phase"))
    circuit = oq.Circuit(6).ry(a * b, 0).rz(a.sin(), 0).ry(b, 5).with_global_phase(phase)
    return oq.quantum_metric(circuit, parameters=("a", "b", "phase"))


def _array(value, backend):
    return value.detach().cpu().numpy() if backend == "torch" else np.asarray(value)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_native_metric_factorization_phase_nullspace_and_parameter_derivatives(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    plan = _plan()
    reference = plan.matrix({"a": 0.3, "b": 0.4, "phase": 0.7})
    theta = xp.asarray([0.3, 0.4, 0.7], dtype=xp.float64 if backend == "torch" else xp.float32)
    vector = xp.asarray([0.2, -0.4, 0.8], dtype=theta.dtype)

    def function(value):
        return plan.native_matvec(
            dict(zip(plan.parameters, value, strict=True)), vector, backend=backend
        )

    actual = function(theta)
    np.testing.assert_allclose(
        _array(actual, backend), reference @ _array(vector, backend), rtol=3e-5, atol=2e-7
    )
    assert abs(float(actual[-1])) < 2e-7
    if backend == "torch":
        theta.requires_grad_()
        derivative = lib.autograd.grad(function(theta).sum(), theta)[0]
        assert actual.device == theta.device
    else:
        derivative = lib.jit(lib.grad(lambda x: function(x).sum()))(theta)
        np.testing.assert_allclose(lib.jit(function)(theta), actual, rtol=3e-5, atol=2e-7)
    h = 1e-5
    expected = []
    for i in range(3):
        plus, minus = np.array([0.3, 0.4, 0.7]), np.array([0.3, 0.4, 0.7])
        plus[i] += h
        minus[i] -= h
        matrices = [plan.matrix(dict(zip(plan.parameters, x, strict=True))) for x in (plus, minus)]
        expected.append(
            (((matrices[0] - matrices[1]) / (2 * h)) @ np.array([0.2, -0.4, 0.8])).sum()
        )
    np.testing.assert_allclose(_array(derivative, backend), expected, atol=2e-6, rtol=3e-4)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_native_cg_certifies_the_actual_system_and_preserves_native_results(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    plan, bindings = _plan(), {"a": 0.3, "b": 0.4, "phase": 0.7}
    gradient = xp.asarray([0.2, -0.4, 0.8], dtype=xp.float64 if backend == "torch" else xp.float32)

    def run(value):
        return plan.native_natural_gradient(
            bindings, value, backend=backend, damping=0.05, max_iterations=8, tolerance=1e-5
        )

    result = run(gradient) if backend == "torch" else lib.jit(run)(gradient)
    matrix = plan.matrix(bindings) + 0.05 * np.eye(3)
    expected = np.linalg.solve(matrix, _array(gradient, backend))
    assert bool(result.converged)
    np.testing.assert_allclose(_array(result.direction, backend), expected, rtol=3e-5, atol=3e-6)
    residual = np.linalg.norm(
        matrix @ _array(result.direction, backend) - _array(gradient, backend)
    )
    assert residual < 1e-5
    assert float(result.residual_norm) < 1e-5
    assert int(result.metric_products) >= int(result.iterations) + 1


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_spectral_step_preserves_inactive_subspaces_and_certifies_only_the_original_solve(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    parameters = tuple(oq.Parameter(f"p{i}") for i in range(4))
    circuit = oq.Circuit(4)
    for i, parameter in enumerate(parameters):
        circuit = circuit.ry(parameter, i)
    plan = oq.quantum_metric(circuit)
    bindings = {name: 0.2 for name in plan.parameters}
    gradient = xp.asarray([1.0, 0, 0, 0], dtype=xp.float64 if backend == "torch" else xp.float32)
    result = plan.spectral_step(
        bindings,
        gradient,
        block_shapes=((2, 2),),
        backend=backend,
        mixing=1,
        max_step_norm=0.02,
        learning_rate=1,
        damping=0.05,
    )
    np.testing.assert_allclose(_array(result.step, backend), [0.02, 0, 0, 0], atol=2e-7)
    assert bool(result.natural.converged) and float(result.alignment) > 0
    assert not bool(result.used_natural_direction)
    np.testing.assert_allclose(
        (plan.matrix(bindings) + 0.05 * np.eye(4)) @ _array(result.natural.direction, backend),
        [1, 0, 0, 0],
        atol=3e-6,
    )
    zero = plan.spectral_step(bindings, gradient * 0, block_shapes=((2, 2),), backend=backend)
    np.testing.assert_array_equal(_array(zero.step, backend), np.zeros(4))
    assert bool(zero.natural.converged)


def test_native_geometry_keeps_higher_torch_derivatives_and_fullgraph_capture():
    torch = pytest.importorskip("torch")
    plan = _plan()
    theta = torch.tensor([0.3, 0.4, 0.7], dtype=torch.float64, requires_grad=True)
    vector = torch.tensor([0.2, -0.4, 0.8], dtype=torch.float64)

    def f(value):
        return plan.native_matvec(dict(zip(plan.parameters, value, strict=True)), vector)

    assert torch.autograd.gradcheck(f, (theta,))
    assert torch.autograd.gradgradcheck(f, (theta,))
    if tuple(int(v) for v in torch.__version__.split(".")[:2]) >= (2, 4):
        compiled = oq.compile_compute(f, backend="torch", compiler="eager", fullgraph=True)
        torch.testing.assert_close(compiled(theta), f(theta))


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_native_sample_geometry_is_the_weighted_mean_without_per_sample_metric_matrices(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    plan = _plan()
    dtype = xp.float64 if backend == "torch" else xp.float32
    a = xp.asarray([0.1, 0.3, -0.2], dtype=dtype)
    b = xp.asarray(0.4, dtype=dtype)
    mass = xp.asarray([0.2, 0.3, 0.5], dtype=dtype)
    vector = xp.asarray([0.2, -0.4, 0.8], dtype=dtype)

    def run(weights):
        return plan.native_matvec(
            {"a": a, "b": b, "phase": 0.7}, vector, sample_weights=weights, backend=backend
        )

    matrices = [plan.matrix({"a": value, "b": 0.4, "phase": 0.7}) for value in (0.1, 0.3, -0.2)]
    expected = sum(p * matrix for p, matrix in zip((0.2, 0.3, 0.5), matrices, strict=True))
    result = run(mass) if backend == "torch" else lib.jit(run)(mass)
    np.testing.assert_allclose(
        _array(result, backend), expected @ np.array([0.2, -0.4, 0.8]), rtol=3e-5, atol=2e-7
    )
    if backend == "torch":
        mass.requires_grad_()
        derivative = lib.autograd.grad(run(mass).sum(), mass)[0]
        assert lib.isfinite(derivative).all() and bool(derivative.abs().sum() > 0)
    else:
        derivative = lib.jit(lib.grad(lambda weights: run(weights).sum()))(mass)
        assert np.isfinite(derivative).all() and np.abs(derivative).sum() > 0
    with pytest.raises(oq.ValidationError):
        run(mass * 0)


def test_native_geometry_admits_work_before_state_allocation_and_validates_bindings():
    torch = pytest.importorskip("torch")
    x = oq.Parameter("x")
    circuit = oq.Circuit(5).h(0)
    for i in range(4):
        circuit = circuit.cx(i, i + 1)
    for _ in range(20):
        circuit = circuit.ry(x, 0)
    plan = oq.quantum_metric(circuit)
    with oq.compute_context(policy=oq.ComputePolicy(max_memory_mb=0.001)):
        with pytest.raises(oq.ResourceError):
            plan.native_matvec({"x": 0.2}, torch.ones(1, dtype=torch.float64))
    plan = _plan()
    for bindings, vector in (
        ({}, torch.ones(3)),
        ({"a": 0.2, "b": 0.3, "phase": 0.1}, torch.ones(2)),
    ):
        with pytest.raises(oq.ValidationError):
            plan.native_matvec(bindings, vector)


def test_native_geometry_runs_on_a_physical_cuda_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("A physical CUDA device is required.")
    plan = _plan()
    vector = torch.tensor([0.2, -0.4, 0.8], dtype=torch.float64, device="cuda")
    result = plan.native_natural_gradient(
        {"a": 0.3, "b": 0.4, "phase": 0.7}, vector, damping=0.05, max_iterations=8
    )
    assert result.direction.is_cuda and result.residual_norm.is_cuda
    assert bool(result.converged)
