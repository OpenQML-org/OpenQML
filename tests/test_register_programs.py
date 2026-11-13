"""Exact register reuse, native derivatives and checked accelerator graph caching."""

import weakref

import numpy as np
import pytest

import openqml as oq
from openqml.compute._program import lower_program
from openqml.sim.dense import _TensorOps
from openqml.sim.pauli import PauliPlan, _evaluate_program


def test_liveness_releases_dead_arrays_without_mutating_caller_storage(monkeypatch):
    nodes = (("param", "x"),) + tuple(("sin", i) for i in range(500))
    roots = (len(nodes) - 1,)
    program = lower_program(nodes, roots)
    assert program.registers == 2
    x = np.linspace(-0.3, 0.4, 64)
    original = x.copy()
    references, peak, sine = [], [], np.sin

    def tracked(value):
        result = sine(value)
        references.append(weakref.ref(result))
        peak.append(sum(reference() is not None for reference in references))
        return result

    monkeypatch.setattr(np, "sin", tracked)
    actual = _evaluate_program(nodes, roots, {"x": x}, _TensorOps(), x.shape, program=program)
    expected = x.copy()
    for _ in range(500):
        expected = sine(expected)
    np.testing.assert_array_equal(actual[:, 0], expected)
    np.testing.assert_array_equal(x, original)
    assert max(peak) <= 3 and not np.shares_memory(actual, x)


def test_register_roots_and_repeated_inputs_remain_live_through_the_last_use():
    nodes = (("param", "x"), ("mul", 0, 0), ("sin", 1), ("add", 0, 2), ("mul", 3, 1))
    roots, x = (0, 4, 2, 0), np.linspace(-0.5, 0.7, 8)
    p = lower_program(nodes, roots)
    actual = _evaluate_program(nodes, roots, {"x": x}, _TensorOps(), x.shape, program=p)
    np.testing.assert_allclose(
        actual, np.stack((x, (x + np.sin(x * x)) * x * x, np.sin(x * x), x), axis=-1)
    )


def _plan():
    x = oq.Parameter("x")
    circuit = oq.Circuit(1)
    for _ in range(12):
        circuit = circuit.ry(x, 0).rz(x / 3, 0)
    return PauliPlan(circuit, (oq.PauliString("Z"), oq.PauliString("X")))


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_register_evaluation_preserves_higher_derivatives_and_compilation(framework):
    xp = pytest.importorskip(framework)
    plan = _plan()
    assert plan._program.registers < len(plan._nodes)
    if framework == "torch":
        value = xp.tensor(0.23, dtype=xp.float64, requires_grad=True)
        result = plan.evaluate({"x": value}, backend="torch", like=value)
        first = xp.autograd.grad(result.sum(), value, create_graph=True)[0]
        second = xp.autograd.grad(first, value)[0]
        fn = xp.compile(
            lambda a: plan.evaluate({"x": a}, backend="torch", like=a),
            backend="eager",
            fullgraph=True,
        )
        xp.testing.assert_close(fn(value), result)
        numbers = float(first), float(second)
    else:
        import jax.numpy as jnp

        def fn(a):
            return plan.evaluate({"x": a}, backend="jax", like=a).sum()

        first = xp.jit(xp.grad(fn))(jnp.asarray(0.23))
        second = xp.jit(xp.grad(xp.grad(fn)))(jnp.asarray(0.23))
        numbers = float(first), float(second)
    h = 1e-4

    def f(a):
        return plan.evaluate({"x": a}).sum()

    reference = (
        (f(0.23 + h) - f(0.23 - h)) / (2 * h),
        (f(0.23 + h) - 2 * f(0.23) + f(0.23 - h)) / h**2,
    )
    np.testing.assert_allclose(numbers, reference, rtol=2e-4, atol=2e-3)


def test_checked_native_readout_compiles_once_and_never_reuses_parameter_values():
    torch = pytest.importorskip("torch")
    plan = PauliPlan(oq.Circuit(1).ry(oq.Parameter("x"), 0), (oq.PauliString("Z"),))
    backend, like = oq.CUDABackend(compile_mode="default"), torch.empty((), dtype=torch.float64)
    function, lock, names = backend._compiled_readout(plan, 3, like)
    assert names == ("x",)
    assert backend._compiled_readout(plan, 3, like)[0] is function
    with torch.no_grad(), lock:
        for angles in ([0.2, 0.4, -0.3], [0.8, -0.7, 0.1]):
            packed = torch.tensor(angles, dtype=torch.float64)[:, None]
            np.testing.assert_allclose(function(packed).numpy()[:, 0], np.cos(angles), atol=2e-12)
    with oq.compute_context(policy=oq.ComputePolicy(validation_atol=1e-5)):
        assert backend._compiled_readout(plan, 3, like)[0] is not function
    backend.clear_cache()
    assert backend._compiled._cache.info()["entries"] == 0


def test_packed_native_inputs_can_omit_eliminated_parameter_columns():
    torch = pytest.importorskip("torch")
    from openqml.interop._parameter_batches import parameter_batches
    from openqml.sim._cuda import _native_parameter_matrix

    circuits = [oq.Circuit(1).ry(v, 0).rz(2 * v, 0) for v in (0.1, 0.2, 0.3)]
    _, prototype, bindings = next(parameter_batches(circuits))
    plan = PauliPlan(prototype, (oq.PauliString("Z"),))
    assert len(plan.active_parameters) == 1 and len(bindings) == 2
    packed = _native_parameter_matrix(
        bindings, 0, 3, torch.empty((), dtype=torch.float64), plan.active_parameters
    )
    assert packed.shape == (3, 1)
    np.testing.assert_allclose(packed.numpy()[:, 0], [0.1, 0.2, 0.3])


@pytest.mark.parametrize("mode", [[], 1, "unknown"])
def test_invalid_gpu_compile_modes_fail_before_device_startup(mode):
    with pytest.raises(oq.ValidationError, match="compile_mode"):
        oq.CUDABackend(compile_mode=mode)


def test_explicit_cuda_indices_are_normalized_without_loading_cuda():
    assert oq.CUDABackend(device="cuda:02").device == "cuda:2"
