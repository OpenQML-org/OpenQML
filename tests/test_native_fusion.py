"""Native fusion must preserve batching, transforms and bounded execution."""

import numpy as np
import pytest

import openqml as oq


def _circuit(repeats=3):
    x, y = oq.Parameter("x"), oq.Parameter("y")
    circuit = oq.Circuit(6)
    for _ in range(repeats):
        circuit = circuit.h(5).ry(x * y, 0).cx(5, 0).rz(x.sin(), 5).crx(y, 5, 0)
    return circuit.with_global_phase(x / 3)


def _numpy(value, backend):
    return value.detach().numpy() if backend == "torch" else np.asarray(value)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_native_fusion_batches_reversed_wires_and_reduces_full_state_updates(backend, monkeypatch):
    xp = pytest.importorskip(backend)
    from openqml.sim import dense

    if backend == "jax":
        xp = xp.numpy
    bindings = {"x": xp.asarray([[0.2], [-0.3]]), "y": xp.asarray([0.1, 0.3, -0.2])}
    rng = np.random.default_rng(12)
    unitary = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    circuit = _circuit(14).barrier().ccx(5, 0, 3).h(3).unitary(unitary, 5, 3)
    expected = oq.statevector(circuit, bindings, backend=backend, fusion=False)
    apply, calls = dense._apply_matrix, []

    def observe(state, matrix, wires, width, ops):
        calls.append(wires)
        return apply(state, matrix, wires, width, ops)

    monkeypatch.setattr(dense, "_apply_matrix", observe)
    result = oq.statevector(circuit, bindings, backend=backend)
    assert len(calls) == 4  # Two bounded groups, the three-wire gate and final fixed group.
    np.testing.assert_allclose(_numpy(result, backend), _numpy(expected, backend), atol=3e-6)
    np.testing.assert_allclose(np.sum(abs(_numpy(result, backend)) ** 2, axis=-1), 1, atol=6e-6)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_native_fusion_keeps_first_second_and_directional_derivatives(backend):
    xp = pytest.importorskip(backend)
    circuit, observable = _circuit(), oq.PauliString("XIIIIZ")

    def objective(value, fusion=True, checkpoint=None):
        state = oq.statevector(
            circuit,
            {"x": value[0], "y": value[1]},
            backend=backend,
            fusion=fusion,
            checkpoint_interval=checkpoint,
        )
        # Include phase-sensitive amplitudes, not only real expectations.
        return state.real.sum() + state.imag.sum() / 5

    if backend == "torch":
        value = xp.tensor([0.2, -0.3], dtype=xp.float64)
        grad, hessian, jvp, vectorize = xp.func.grad, xp.func.hessian, xp.func.jvp, xp.func.vmap
    else:
        value = xp.numpy.array([0.2, -0.3])
        grad, hessian, jvp, vectorize = xp.grad, xp.hessian, xp.jvp, xp.vmap

    def reference(v):
        return objective(v, fusion=False)

    for transform in (grad, hessian):
        np.testing.assert_allclose(
            _numpy(transform(objective)(value), backend),
            _numpy(transform(reference)(value), backend),
            atol=3e-6,
        )
    for checkpoint in (None, 7):

        def candidate(v, checkpoint=checkpoint):
            return objective(v, checkpoint=checkpoint)

        actual = jvp(candidate, (value,), (value * 0 + 1,))
        expected = jvp(reference, (value,), (value * 0 + 1,))
        np.testing.assert_allclose(
            [_numpy(v, backend) for v in actual],
            [_numpy(v, backend) for v in expected],
            atol=3e-6,
        )
    np.testing.assert_allclose(
        _numpy(vectorize(grad(objective))(value[None] * value[:, None]), backend),
        _numpy(vectorize(grad(reference))(value[None] * value[:, None]), backend),
        atol=3e-6,
    )
    options = {"compiler": "eager"} if backend == "torch" else {}
    transform = xp.func.jacrev if backend == "torch" else xp.jacrev
    if backend == "torch" and tuple(int(v) for v in xp.__version__.split(".")[:2]) < (2, 4):
        transform = None
    compiled = oq.compile_circuit(
        circuit, [observable], backend=backend, transform=transform, **options
    )
    bindings = {"x": value[0], "y": value[1]}
    actual = compiled(bindings)

    def function(v):
        return oq.expectations(circuit, [observable], v, backend=backend, fusion=False)

    expected = function(bindings) if transform is None else transform(function)(bindings)
    for a, b in zip(
        actual.values() if isinstance(actual, dict) else (actual,),
        expected.values() if isinstance(expected, dict) else (expected,),
        strict=True,
    ):
        np.testing.assert_allclose(_numpy(a, backend), _numpy(b, backend), atol=3e-6)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_native_fusion_respects_small_workspace_and_repeated_cold_caches(backend, monkeypatch):
    xp = pytest.importorskip(backend)
    from openqml.sim import dense

    circuit = oq.Circuit(3).h(2).x(0).cx(2, 0).with_global_phase(oq.Parameter("x"))
    value = xp.tensor(0.2, dtype=xp.float64) if backend == "torch" else xp.numpy.asarray(0.2)
    bindings = {"x": value}
    expected = oq.statevector(circuit, bindings, backend=backend, fusion=False)
    # Four complex128 states fit; the extra 4x4 products do not.
    budget = 8 * 64 / 2**20
    apply, calls = dense._apply_matrix, []

    def observe(state, matrix, wires, width, ops):
        calls.append(wires)
        return apply(state, matrix, wires, width, ops)

    monkeypatch.setattr(dense, "_apply_matrix", observe)
    for _ in range(2):
        with oq.compute_context(policy=oq.ComputePolicy(max_memory_mb=budget)):
            result = oq.statevector(circuit, bindings, backend=backend, max_memory_mb=budget)
            np.testing.assert_allclose(
                _numpy(result, backend), _numpy(expected, backend), atol=2e-7
            )
    assert len(calls) == 6
