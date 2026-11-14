"""Noisy-state SLD values, implicit derivatives and hybrid optimizer ownership."""

import copy

import numpy as np
import pytest

import openqml as oq


def _native(backend, value):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    return xp.asarray(value, dtype=xp.float64 if backend == "torch" else xp.float32)


def _numpy(backend, value):
    return value.detach().numpy() if backend == "torch" else np.asarray(value)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("probability", [0.0, 0.2, 0.75])
def test_depolarized_qubit_matches_analytic_sld_including_repeated_eigenvalues(
    backend, probability
):
    theta, noise, phase = (oq.Parameter(n) for n in ("theta", "noise", "phase"))
    circuit = oq.Circuit(1).ry(theta, 0).depolarizing(noise, 0).with_global_phase(phase)
    plan = oq.quantum_fisher(circuit, parameters=("noise", "theta", "phase"), state_damping=0.002)
    vector = _native(backend, [0.4, -0.6, 0.8])
    p, width = probability, 1.004
    radius = 1 - 4 * p / 3
    expected = np.diag([16 * width / (9 * (width**2 - radius**2)), radius**2 / width, 0])
    result = plan.native_matvec({"noise": p, "theta": 0.47, "phase": 0.2}, vector, backend=backend)
    np.testing.assert_allclose(
        _numpy(backend, result), expected @ [0.4, -0.6, 0.8], rtol=2e-4, atol=2e-5
    )


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_mixed_geometry_chunking_shared_coordinates_and_natural_residual(backend):
    x, w, v = (oq.Parameter(n) for n in ("x", "w", "v"))
    circuit = oq.Circuit(2).ry(w * x, 0).rz(v, 0).cx(0, 1).amplitude_damping(0.3, 1)
    plan = oq.quantum_fisher(circuit, parameters=("w", "v"), state_damping=0.003)
    gradient = _native(backend, [0.2, -0.4])
    bindings = {"x": _native(backend, [0.2, -0.1, 0.9, 0.4, 0.7]), "w": 0.4, "v": 0.3}
    weights = _native(backend, [1, 0, 3, 2, 1])
    a = plan.native_matvec(bindings, gradient, backend=backend, sample_weights=weights)
    b = plan.native_matvec(
        bindings, gradient, backend=backend, sample_weights=weights, sample_chunk_size=2
    )
    np.testing.assert_allclose(_numpy(backend, a), _numpy(backend, b), rtol=2e-5, atol=2e-7)
    solve = plan.native_natural_gradient(
        bindings,
        gradient,
        backend=backend,
        sample_weights=weights,
        sample_chunk_size=2,
        damping=0.1,
        tolerance=2e-5,
        max_iterations=8,
    )
    residual = (
        plan.native_matvec(bindings, solve.direction, backend=backend, sample_weights=weights)
        + 0.1 * solve.direction
        - gradient
    )
    assert bool(solve.converged)
    assert np.linalg.norm(_numpy(backend, residual)) < 2e-5


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_density_fisher_derivatives_are_finite_at_maximally_mixed_state(backend):
    noise = oq.Parameter("noise")
    plan = oq.quantum_fisher(oq.Circuit(1).depolarizing(noise, 0), state_damping=0.003)
    vector = _native(backend, [1])

    def function(p):
        return plan.native_matvec({"noise": p}, vector, backend=backend).sum()

    lib = pytest.importorskip(backend)
    point = _native(backend, 0.75)
    if backend == "torch":
        point.requires_grad_()
        value = function(point)
        first = lib.autograd.grad(value, point, create_graph=True)[0]
        second = lib.autograd.grad(first, point)[0]
    else:
        value = lib.jit(function)(point)
        first = lib.jit(lib.grad(function))(point)
        second = lib.jit(lib.grad(lib.grad(function)))(point)
    width = 1.006
    np.testing.assert_allclose(_numpy(backend, value), 16 / (9 * width), rtol=3e-5)
    assert abs(float(first)) < 1e-5
    np.testing.assert_allclose(_numpy(backend, second), 512 / (81 * width**3), rtol=2e-4)


def test_torch_sld_complex_derivatives_and_forward_transform():
    torch = pytest.importorskip("torch")
    a, b = oq.Parameter("a"), oq.Parameter("b")
    plan = oq.quantum_fisher(oq.Circuit(2).ry(a, 0).rx(b, 1).cx(0, 1).depolarizing(0.2, 0))
    vector = torch.tensor([0.7, -0.4], dtype=torch.float64)

    def function(value):
        return plan.native_matvec({"a": value[0], "b": value[1]}, vector)

    value = torch.tensor([0.4, 0.7], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(function, (value,))
    assert torch.autograd.gradgradcheck(function, (value,))
    _, tangent = torch.func.jvp(function, (value,), (vector,))
    expected = (function(value + 1e-5 * vector) - function(value - 1e-5 * vector)) / 2e-5
    torch.testing.assert_close(tangent, expected, rtol=1e-4, atol=1e-7)


def test_torch_density_metric_fullgraph_backward():
    torch = pytest.importorskip("torch")
    if tuple(int(v) for v in torch.__version__.split(".")[:2]) < (2, 4):
        pytest.skip("Compiled native JVP transforms require supported Torch>=2.4.")
    w = oq.Parameter("w")
    plan = oq.quantum_fisher(oq.Circuit(1).ry(w, 0).amplitude_damping(0.3, 0))

    def function(value):
        return plan.native_matvec({"w": value}, torch.ones(1, dtype=value.dtype))

    value = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    compiled = torch.compile(function, backend="aot_eager", fullgraph=True)
    expected, actual = function(value), compiled(value)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), value)[0], torch.autograd.grad(expected.sum(), value)[0]
    )


def test_measurement_record_is_marginalized_and_factorization_admits_disconnected_width():
    pytest.importorskip("torch")
    w = oq.Parameter("w")
    first = oq.Circuit(1, num_clbits=1).ry(w, 0).measure(0, 0)
    circuit = first.if_else(oq.ClassicalCondition((0,), 1), oq.Circuit(1).x(0))
    plan = oq.quantum_fisher(circuit)
    result = plan.native_matvec({"w": 0.7}, _native("torch", [1]))
    np.testing.assert_allclose(result, [0], atol=1e-12)
    wide = oq.quantum_fisher(oq.Circuit(128).ry(w, 127).depolarizing(0.1, 127), max_memory_mb=1)
    assert wide.to_dict()["max_component_qubits"] == 1
    with pytest.raises(oq.ResourceError):
        oq.quantum_fisher(oq.Circuit(10).ry(w, 0).cx(0, 9), max_memory_mb=1e-5)
    with pytest.raises(oq.ValidationError):
        oq.quantum_fisher(circuit, state_damping=0)


def test_noisy_hybrid_qnm_updates_weights_and_round_trips_geometry_state():
    torch = pytest.importorskip("torch")
    from openqml.qml.interfaces.torch import QuantumLayer

    x, w = oq.Parameter("x"), oq.Parameter("w")
    circuit = oq.Circuit(1).ry(x * w, 0).amplitude_damping(0.2, 0)
    model = QuantumLayer(circuit, ("x",), ("w",), dtype=torch.float64, device="cpu")
    optimizer = oq.hybrid_qnm(model, quantum_options={"sample_chunk_size": 2})
    inputs = torch.tensor([[0.2], [0.4], [0.6]], dtype=torch.float64)
    before = model.weights.detach().clone()
    model(inputs).square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, model.weights)
    state = copy.deepcopy(optimizer.state_dict())
    optimizer.load_state_dict(state)
    assert optimizer.state[model.weights]["openqml_qnm"]["steps"] == 1
    other = oq.hybrid_qnm(model, fisher_options={"state_damping": 0.01})
    with pytest.raises(oq.ValidationError):
        other.load_state_dict(state)
    optimizer.close()
    other.close()
