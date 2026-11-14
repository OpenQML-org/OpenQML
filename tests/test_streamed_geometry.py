"""Streaming Fisher products against independent per-sample dense metrics."""

import numpy as np
import pytest

import openqml as oq


def _plan(budget=256):
    a, b, x = (oq.Parameter(n) for n in ("a", "b", "x"))
    circuit = oq.Circuit(3).ry(a + x.log(), 0).cx(0, 1).rx(a * b, 1).cx(1, 2).rz(b, 2)
    return oq.quantum_metric(circuit, parameters=("a", "b"), max_memory_mb=budget)


def _numpy(value, backend):
    return value.detach().numpy() if backend == "torch" else np.asarray(value)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("chunk", [1, 3, 8])
def test_streaming_matches_weighted_dense_reference_with_nonlinear_auxiliary_inputs(backend, chunk):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    dtype = xp.float64 if backend == "torch" else xp.float32
    plan = _plan()
    features, masses = np.linspace(0.4, 1.6, 7), np.array([1, 0, 2, 4, 0, 3, 1.0])
    vector = np.array([0.2, -0.3])
    bindings = {
        "a": xp.asarray(0.3, dtype=dtype),
        "b": xp.asarray(-0.2, dtype=dtype),
        "x": xp.asarray(features, dtype=dtype),
    }
    reference = sum(
        mass * (plan.matrix({"a": 0.3, "b": -0.2, "x": value}) @ vector)
        for value, mass in zip(features, masses / masses.sum(), strict=True)
    )

    def product(values, direction):
        return plan.native_matvec(
            values,
            direction,
            backend=backend,
            sample_weights=xp.asarray(masses, dtype=dtype),
            sample_chunk_size=chunk,
        )

    run = product if backend == "torch" else lib.jit(product)
    actual = run(bindings, xp.asarray(vector, dtype=dtype))
    np.testing.assert_allclose(_numpy(actual, backend), reference, rtol=2e-5, atol=2e-7)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_streaming_native_gradient_and_hessian_match_dense_finite_differences(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    dtype = xp.float64 if backend == "torch" else xp.float32
    plan = _plan()
    features = np.linspace(0.5, 1.8, 5)
    vector = np.array([0.4, -0.2])
    theta = xp.asarray([0.3, -0.2], dtype=dtype)

    def objective(value):
        product = plan.native_matvec(
            {"a": value[0], "b": value[1], "x": xp.asarray(features, dtype=dtype)},
            xp.asarray(vector, dtype=dtype),
            backend=backend,
            sample_chunk_size=2,
        )
        return product.sum()

    def dense(value):
        return sum(
            (plan.matrix({"a": value[0], "b": value[1], "x": x}) @ vector).sum() for x in features
        ) / len(features)

    if backend == "torch":
        theta.requires_grad_()
        assert lib.autograd.gradcheck(objective, (theta,))
        assert lib.autograd.gradgradcheck(objective, (theta,))
        gradient = lib.autograd.grad(objective(theta), theta)[0]
    else:
        gradient = lib.jit(lib.grad(objective))(theta)
        assert np.isfinite(lib.jit(lib.hessian(objective))(theta)).all()
    center, step = np.array([0.3, -0.2]), 1e-5
    expected = [
        (dense(center + np.eye(2)[i] * step) - dense(center - np.eye(2)[i] * step)) / (2 * step)
        for i in range(2)
    ]
    np.testing.assert_allclose(_numpy(gradient, backend), expected, rtol=4e-4, atol=3e-6)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_chunking_admits_a_batch_that_exceeds_the_full_state_workspace(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    plan = _plan(0.25)
    bindings = {
        "a": xp.asarray(0.3),
        "b": xp.asarray(-0.2),
        "x": xp.asarray(np.linspace(0.5, 2, 128)),
    }
    vector = xp.asarray([0.2, -0.3])
    with pytest.raises(oq.ResourceError):
        plan.native_matvec(bindings, vector, backend=backend)
    value = plan.native_matvec(bindings, vector, backend=backend, sample_chunk_size=8)
    assert np.isfinite(_numpy(value, backend)).all()
    result = plan.native_natural_gradient(
        bindings, vector, backend=backend, sample_chunk_size=8, damping=0.1, max_iterations=8
    )
    assert bool(result.converged)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_distance_state_can_resume_with_a_different_sample_chunk_size(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    plan = _plan()
    full = oq.DistanceAwareQNM(plan, ((1, 2),), damping=0.1, max_iterations=8)
    chunked = oq.DistanceAwareQNM(
        plan, ((1, 2),), damping=0.1, max_iterations=8, sample_chunk_size=2
    )
    assert full._signature == chunked._signature
    local = {
        "a": xp.asarray(0.3),
        "b": xp.asarray(-0.2),
        "x": xp.asarray(np.linspace(0.5, 2, 5)),
    }
    gradient = xp.asarray([0.2, -0.3])
    carry = full.step(local, gradient, backend=backend).state
    a = full.step(local, gradient, state=carry, backend=backend)
    b = chunked.step(local, gradient, state=carry, backend=backend)
    np.testing.assert_allclose(
        _numpy(a.step, backend), _numpy(b.step, backend), rtol=5e-5, atol=1e-7
    )


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_streaming_weights_keep_exact_probability_gradients(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    dtype = xp.float64 if backend == "torch" else xp.float32
    plan = _plan()
    features = np.linspace(0.5, 1.8, 5)
    vector = np.array([0.2, -0.3])
    coefficients = np.array(
        [(plan.matrix({"a": 0.3, "b": -0.2, "x": x}) @ vector).sum() for x in features]
    )
    weights = np.array([0.3, 0.1, 0.9, 0.5, 0.2])

    def objective(mass):
        local = {
            "a": xp.asarray(0.3, dtype=dtype),
            "b": xp.asarray(-0.2, dtype=dtype),
            "x": xp.asarray(features, dtype=dtype),
        }
        return plan.native_matvec(
            local,
            xp.asarray(vector, dtype=dtype),
            backend=backend,
            sample_weights=mass,
            sample_chunk_size=2,
        ).sum()

    mass = xp.asarray(weights, dtype=dtype)
    if backend == "torch":
        mass.requires_grad_()
        assert lib.autograd.gradcheck(objective, (mass,))
        assert lib.autograd.gradgradcheck(objective, (mass,))
        gradient = lib.autograd.grad(objective(mass), mass)[0]
    else:
        gradient = lib.jit(lib.grad(objective))(mass)
    expected = (coefficients - weights @ coefficients / weights.sum()) / weights.sum()
    np.testing.assert_allclose(_numpy(gradient, backend), expected, rtol=1e-3, atol=1e-7)


def test_streaming_preserves_torch_fullgraph_capture():
    torch = pytest.importorskip("torch")
    if tuple(map(int, torch.__version__.split(".")[:2])) < (2, 4):
        pytest.skip("Native JVP fullgraph requires supported Torch>=2.4.")
    plan = _plan()
    inputs = {"a": torch.tensor(0.3), "b": torch.tensor(-0.2), "x": torch.linspace(0.5, 2, 5)}
    vector = torch.tensor([0.2, -0.3])
    run = oq.compile_compute(
        lambda b, v: plan.native_matvec(b, v, sample_chunk_size=2),
        backend="torch",
        compiler="eager",
        fullgraph=True,
    )
    torch.testing.assert_close(run(inputs, vector), plan.native_matvec(inputs, vector))


def test_streamed_geometry_on_physical_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("A physical CUDA device is required.")
    plan = _plan()
    bindings = {
        "a": torch.tensor(0.3, device="cuda"),
        "b": torch.tensor(-0.2, device="cuda"),
        "x": torch.linspace(0.5, 2, 17, device="cuda"),
    }
    gradient = torch.tensor([0.2, -0.3], device="cuda")
    a = plan.native_natural_gradient(bindings, gradient, damping=0.1, max_iterations=8)
    b = plan.native_natural_gradient(
        bindings, gradient, damping=0.1, max_iterations=8, sample_chunk_size=4
    )
    assert b.direction.is_cuda and bool(b.converged)
    torch.testing.assert_close(a.direction, b.direction, atol=2e-6, rtol=2e-5)
