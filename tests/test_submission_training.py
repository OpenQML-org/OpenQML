"""Independent derivatives and actual submission counts for packed training."""

from dataclasses import replace

import numpy as np
import pytest

import openqml as oq


class CountingBackend:
    name = "stencil-counter"

    def __init__(self, deterministic=False):
        self.calls = []
        self.capabilities = oq.BackendCapabilities(
            native_expectations=True,
            exact_expectations=True,
            deterministic_expectations=deterministic,
            channels=True,
        )

    def expectations_batch(self, circuits, observables, *, options):
        self.calls.append(len(circuits))
        return oq.expectations_batch(circuits, observables, backend="numpy")


@pytest.mark.parametrize("deterministic", [False, True])
def test_linearization_matches_finite_differences_and_does_not_replay_products(deterministic):
    x, y, p = (oq.Parameter(n) for n in ("x", "y", "p"))
    circuit = oq.Circuit(2).ry(x * x + y, 0).crx(x, 0, 1).amplitude_damping(p, 1).ry(y, 1)
    obs, backend = (oq.PauliString("ZZ"), oq.PauliString("IX")), CountingBackend(deterministic)
    rows = [
        {"x": 0.2, "y": 0.3, "p": 0.4},
        {"x": 0.2, "y": 0.3, "p": 0.4},
        {"x": -0.4, "y": 0.1, "p": 0.2},
    ]
    plan = oq.ShiftPlan(circuit, obs, parameters=("y", "x", "p"), backend=backend)
    result = plan.linearize(rows, chunk_size=1000)
    expected = np.empty((3, 2, 3))
    for i, row in enumerate(rows):
        for j, name in enumerate(result.parameters):
            plus, minus = dict(row), dict(row)
            plus[name] += 1e-6
            minus[name] -= 1e-6
            expected[i, :, j] = (
                oq.expectations(circuit, obs, plus) - oq.expectations(circuit, obs, minus)
            ) / 2e-6
    np.testing.assert_allclose(result.jacobian, expected, atol=3e-10)
    np.testing.assert_allclose(result.values, oq.expectations_batch(circuit, obs, rows))
    count = (2 if deterministic else 3) * (1 + plan.shift_evaluations)
    assert backend.calls == [count] and result.submission_batches == 1
    assert result.circuit_evaluations == count
    tangent, cotangent = np.ones((3, 3)), np.ones((3, 2))
    np.testing.assert_allclose(
        result.vjp(cotangent), np.einsum("bm,bmp->bp", cotangent, expected), atol=3e-10
    )
    np.testing.assert_allclose(
        result.jvp(tangent), np.einsum("bmp,bp->bm", expected, tangent), atol=3e-10
    )
    assert backend.calls == [count]
    assert not result.values.flags.writeable and not result.jacobian.flags.writeable


def test_linearization_preflights_all_rows_and_memory_before_external_submission():
    backend = CountingBackend()
    p = oq.Parameter("p")
    plan = oq.ShiftPlan(
        oq.Circuit(1).amplitude_damping(p, 0), [oq.PauliString("Z")], backend=backend
    )
    with pytest.raises(oq.ValidationError):
        plan.linearize({"p": [0.2, -0.1]})
    with pytest.raises(oq.ValidationError):
        plan.linearize({"p": [0.2, 1.0]}, chunk_size=2)
    with pytest.raises(oq.ResourceError):
        plan.linearize({"p": [0.2, 0.3]}, max_memory_mb=1e-8)
    assert not backend.calls
    result = plan.linearize({"p": [0.2, 0.3]}, parameters=(), chunk_size=2)
    assert result.jacobian.shape == (2, 1, 0) and result.circuit_evaluations == 2


def _layer(backend, interface):
    x, w, v = (oq.Parameter(n) for n in ("x", "w", "v"))
    circuit = oq.Circuit(1).ry(x * w, 0).rz(v, 0).ry(w, 0)
    if interface == "torch":
        import torch

        from openqml.qml.interfaces.torch import QuantumLayer

        options = {"dtype": torch.float64, "device": "cpu"}
    else:
        from openqml.qml.interfaces.jax import QuantumLayer

        options = {}
    return QuantumLayer(
        circuit,
        ("x",),
        ("w", "v"),
        backend=backend,
        gradient_method="batched-shift",
        gradient_chunk_size=1000,
        **options,
    )


def test_torch_broadcast_gradients_and_transforms_reuse_the_primal_stencil():
    torch = pytest.importorskip("torch")
    backend = CountingBackend()
    layer = _layer(backend, "torch")
    x = torch.tensor([[0.2], [0.5], [-0.3]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([[[0.4, 0.7]], [[0.2, -0.3]]], dtype=torch.float64, requires_grad=True)
    reference = _layer("numpy", "torch")
    reference.gradient_method = "parameter-shift"

    def objective(a, b):
        return layer.functional(a, b).sin().square().sum()

    value = objective(w, x)
    gradients = torch.autograd.grad(value, (w, x))
    assert len(backend.calls) == 1
    expected = torch.autograd.grad(reference.functional(w, x).sin().square().sum(), (w, x))
    for actual, want in zip(gradients, expected, strict=True):
        torch.testing.assert_close(actual, want)
    backend.calls.clear()
    jac = torch.func.jacrev(lambda a: layer.functional(a, x))(w)
    assert len(backend.calls) == 1 and torch.isfinite(jac).all()
    backend.calls.clear()
    forward_jacobian = torch.func.jacfwd(lambda a: layer.functional(a, x))(w)
    assert len(backend.calls) == 1
    torch.testing.assert_close(forward_jacobian, jac)
    backend.calls.clear()
    _, tangent = torch.func.jvp(lambda a: layer.functional(a, x), (w,), (torch.ones_like(w),))
    assert len(backend.calls) == 1 and torch.isfinite(tangent).all()
    backend.calls.clear()
    mapped = torch.vmap(lambda a: layer.functional(a, x))(w[:, 0, :])
    assert len(backend.calls) == 1 and mapped.shape == (2, 3, 1)
    gradient = torch.autograd.grad(objective(w, x), w, create_graph=True)[0]
    with pytest.raises(oq.ValidationError, match="higher derivatives"):
        torch.autograd.grad(gradient.sum(), w)


def test_jax_jit_value_gradient_jvp_and_vmap_use_one_stencil():
    jax = pytest.importorskip("jax")
    xp, backend = jax.numpy, CountingBackend()
    layer = _layer(backend, "jax")
    x = xp.asarray([[0.2], [0.5], [-0.3]])
    w = xp.asarray([[[0.4, 0.7]], [[0.2, -0.3]]])

    def objective(a, b):
        return xp.square(xp.sin(layer(a, b))).sum()

    value, gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(w, x)
    value.block_until_ready()
    assert len(backend.calls) == 1
    reference = _layer("numpy", "jax")
    reference.gradient_method = "parameter-shift"
    expected = jax.grad(lambda a, b: xp.square(xp.sin(reference(a, b))).sum(), argnums=(0, 1))(w, x)
    for actual, want in zip(gradients, expected, strict=True):
        np.testing.assert_allclose(actual, want, rtol=2e-5, atol=2e-7)
    backend.calls.clear()
    _, tangent = jax.jit(lambda a: jax.jvp(lambda z: layer(z, x), (a,), (xp.ones_like(a),)))(w)
    tangent.block_until_ready()
    assert len(backend.calls) == 1
    backend.calls.clear()
    mapped = jax.jit(jax.vmap(lambda a: layer(a, x)))(w[:, 0, :])
    mapped.block_until_ready()
    assert len(backend.calls) == 1 and mapped.shape == (2, 3, 1)
    with pytest.raises(oq.ValidationError, match="higher derivatives"):
        jax.hessian(lambda a: objective(a, x))(w)


def test_qiskit_v2_estimator_linearization_submits_one_real_job():
    pytest.importorskip("qiskit")
    from qiskit.primitives import StatevectorEstimator

    class Estimator:
        calls = 0

        def run(self, *args, **kwargs):
            self.calls += 1
            return StatevectorEstimator().run(*args, **kwargs)

    estimator = Estimator()
    backend = oq.QiskitEstimatorBackend(estimator, precision=0)
    p = oq.ShiftPlan(oq.Circuit(1).ry(oq.Parameter("x"), 0), [oq.PauliString("Z")], backend=backend)
    result = p.linearize({"x": [0.2, 0.4]}, chunk_size=64)
    np.testing.assert_allclose(result.values[:, 0], np.cos([0.2, 0.4]))
    np.testing.assert_allclose(result.vjp(np.ones((2, 1)))[:, 0], -np.sin([0.2, 0.4]))
    assert estimator.calls == 1


def test_jax_stochastic_submissions_are_effects_even_if_results_are_unused_or_equal():
    jax = pytest.importorskip("jax")
    backend = CountingBackend()
    layer = _layer(backend, "jax")
    w, x = jax.numpy.asarray([0.4, 0.7]), jax.numpy.asarray([[0.2]])

    def unused(weights):
        layer(weights, x)
        return weights.sum() * 0

    jax.jit(unused)(w).block_until_ready()
    jax.effects_barrier()
    assert len(backend.calls) == 1
    backend.calls.clear()
    jax.jit(lambda weights: layer(weights, x) + layer(weights, x))(w).block_until_ready()
    jax.effects_barrier()
    assert len(backend.calls) == 2


def test_joint_workload_prices_one_submission_instead_of_separate_phases():
    from openqml.runtime.costs import estimate_cost

    profile = oq.ExecutionProfile(
        circuit_seconds=0.01, queue_seconds=3, cost_per_job=1, max_batch_size=64
    )
    workload = oq.TrainingWorkload(
        batch_size=5, steps=2, gradient_evaluations=10, microbatch_size=2, gradient_chunk_size=64
    )
    request = oq.SchedulingRequest(
        (oq.Circuit(1),), (oq.PauliString("Z"),), "expectations", oq.ExecutionOptions(), workload
    )
    separate = estimate_cost(profile, request, oq.SchedulingPolicy())
    joint = estimate_cost(
        profile,
        replace(request, workload=replace(workload, joint_primal_gradient=True)),
        oq.SchedulingPolicy(),
    )
    assert separate.provider_jobs == 12 and joint.provider_jobs == 6
    assert separate.circuit_evaluations == joint.circuit_evaluations == 110
    assert separate.physical_shots == joint.physical_shots
    assert joint.cost == 6 and joint.seconds < separate.seconds


def test_flax_parameter_tree_trains_through_the_packed_stencil():
    pytest.importorskip("flax")
    jax = pytest.importorskip("jax")
    from openqml.qml.interfaces.flax import QuantumLayer

    x, w = oq.Parameter("x"), oq.Parameter("w")
    circuit = oq.Circuit(1).ry(x * w, 0)
    backend = CountingBackend()
    layer = QuantumLayer(circuit, ("x",), ("w",), backend=backend, gradient_method="batched-shift")
    inputs = jax.numpy.asarray([[0.2], [0.5]])
    parameters = layer.init(jax.random.key(3), inputs)
    jax.effects_barrier()
    backend.calls.clear()
    value, gradient = jax.jit(
        jax.value_and_grad(lambda p: jax.numpy.square(layer.apply(p, inputs)).sum())
    )(parameters)
    value.block_until_ready()
    assert len(backend.calls) == 1
    assert all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(gradient))
