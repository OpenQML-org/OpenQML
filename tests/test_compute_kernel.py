"""Shared execution, bounded metadata ownership and numerical failure contracts."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import openqml as oq
from openqml import compute
from openqml._contraction_program import ContractionKernel
from openqml._validation import validate_expectations, validate_state
from openqml.finance._array import Ops, admission
from openqml.rl import _Ops
from openqml.simulator import _TensorOps


@pytest.fixture(autouse=True)
def restore_kernel(monkeypatch):
    monkeypatch.setattr(compute, "_KERNEL", oq.ComputeKernel())


@pytest.mark.parametrize(
    "options",
    [
        {"max_memory_mb": 0},
        {"max_memory_mb": True},
        {"max_memory_mb": float("inf")},
        {"max_memory_mb": float("nan")},
        {"cache_entries": -1},
        {"cache_entries": True},
        {"cache_entries": 1.5},
        {"cache_bytes": -1},
        {"validation_atol": -1},
        {"validation_rtol": float("nan")},
    ],
)
def test_invalid_policies_fail_before_execution(options):
    with pytest.raises(oq.ValidationError):
        oq.ComputePolicy(**options)


def test_quantum_rl_and_finance_share_one_kernel_and_primitive_dispatch(monkeypatch):
    kernel = oq.get_compute_kernel()
    assert _TensorOps().kernel is _Ops(np.ones(2)).kernel is Ops(np.ones(2)).kernel is kernel
    calls = []
    original = kernel.matmul

    def observed(left, right, **kwargs):
        calls.append((left.shape, right.shape))
        return original(left, right, **kwargs)

    monkeypatch.setattr(kernel, "matmul", observed)
    np.testing.assert_allclose(
        oq.statevector(oq.Circuit(2).h(0).cx(0, 1)), [2**-0.5, 0, 0, 2**-0.5]
    )
    assert calls
    first = len(calls)
    plan = oq.TensorNetworkPlan(oq.Circuit(2).h(0).cx(0, 1), (oq.PauliString("ZZ"),))
    np.testing.assert_allclose(plan.evaluate(), [1], atol=1e-12)
    assert len(calls) > first
    np.testing.assert_allclose(Ops(np.eye(2)).matmul(np.eye(2), np.ones((2, 1))), 1)


def test_shared_ceiling_applies_to_direct_execution_plans_and_financial_work():
    oq.configure_compute(policy=oq.ComputePolicy(max_memory_mb=0.01))
    assert oq.ExecutionOptions(max_memory_mb=100).max_memory_mb == 0.01
    assert _TensorOps().kernel is oq.get_compute_kernel()
    with pytest.raises(oq.ResourceError):
        oq.statevector(oq.Circuit(12).h(0), max_memory_mb=100)
    with pytest.raises(oq.ResourceError):
        oq.density_matrix(oq.Circuit(7).h(0), max_memory_mb=100)
    with pytest.raises(oq.ResourceError):
        oq.TensorNetworkPlan(oq.Circuit(4).h(0).cx(0, 1), (oq.PauliString("ZZII"),))
    with pytest.raises(oq.ResourceError):
        admission(10_000, max_memory_mb=100)
    # An explicit lower caller budget remains binding.
    with pytest.raises(oq.ResourceError):
        admission(200, max_memory_mb=0.001)


def test_existing_plan_obeys_a_new_ceiling_before_tensor_allocation():
    plan = oq.TensorTrainPlan.from_features(16, 16, bond_dimension=2)
    cores = tuple(np.ones(s) for s in plan.core_shapes)
    oq.configure_compute(policy=oq.ComputePolicy(max_memory_mb=0.001))
    with pytest.raises(oq.ResourceError):
        plan.apply(cores, np.ones((100, 16)))


def test_exact_byte_admission_and_work_limits():
    kernel = oq.get_compute_kernel()
    kernel.admit(64, 1 / 1024)
    with pytest.raises(oq.ResourceError):
        kernel.admit(65, 1 / 1024)
    with pytest.raises(oq.ResourceError):
        kernel.admit(0, work=11, max_work=10)
    for values in ((-1, 16), (1, 0), (True, 16)):
        with pytest.raises(oq.ValidationError):
            kernel.admit(values[0], itemsize=values[1])


@pytest.mark.parametrize("operation", ["matmul", "einsum"])
def test_broadcast_output_is_admitted_before_the_primitive(monkeypatch, operation):
    kernel = oq.ComputeKernel(oq.ComputePolicy(max_memory_mb=0.1))
    left, right = np.ones((1000, 1)), np.ones((1, 1000))
    monkeypatch.setattr(np, operation, lambda *a, **k: pytest.fail("Allocated rejected output"))
    with pytest.raises(oq.ResourceError):
        if operation == "matmul":
            kernel.matmul(left, right)
        else:
            kernel.einsum("ij,jk->ik", left, right)


@pytest.mark.parametrize(
    "equation,shapes",
    [
        ("...ij,...jk->...ik", ((2, 3, 4), (1, 4, 2))),
        ("ii->", ((3, 3),)),
        ("...i,...i->...", ((2, 3), (1, 3))),
        ("ij,jk", ((2, 3), (3, 4))),
        ("ij,kj->ik", ((2, 0), (3, 0))),
    ],
)
def test_budgeted_einsum_matches_numpy_for_broadcast_diagonal_and_empty_axes(equation, shapes):
    kernel = oq.ComputeKernel(oq.ComputePolicy(max_memory_mb=1))
    values = tuple(np.ones(s) for s in shapes)
    np.testing.assert_allclose(kernel.einsum(equation, *values), np.einsum(equation, *values))


def test_plans_are_reused_across_calls_without_caching_bound_values():
    kernel = oq.get_compute_kernel()
    a, b = np.arange(4096.0).reshape(64, 64), np.ones((64, 32))
    np.testing.assert_allclose(kernel.einsum("ij,jk->ik", a, b), a @ b)
    a[:] = 7
    np.testing.assert_allclose(kernel.einsum("ij,jk->ik", a, b), a @ b)
    assert kernel.cache_info()["hits"] == 1
    p = ContractionKernel.compile((0, 1), (1, 2), (1,))
    assert p is ContractionKernel.compile((0, 1), (1, 2), (1,))
    with pytest.raises(oq.ValidationError, match="immutable metadata"):
        kernel.plan("bad", (), lambda: a)


def test_cache_is_bounded_under_eviction_and_concurrent_compilation():
    kernel = oq.configure_compute(policy=oq.ComputePolicy(cache_entries=2, cache_bytes=4096))
    builds = []

    def build():
        builds.append(1)
        return ("immutable", 2)

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(lambda _: kernel.plan("test", (), build), range(32)))
    assert len(builds) == 1
    assert all(v is values[0] for v in values)
    for k in range(10):
        kernel.plan("test", (k,), lambda k=k: (k,))
    info = kernel.cache_info()
    assert info["entries"] <= 2 and info["bytes"] <= 4096 and info["evictions"] > 0
    kernel.plan("oversized", (), lambda: ("x" * 5000,))
    assert kernel.cache_info()["bytes"] <= 4096
    kernel.clear_cache()
    assert kernel.cache_info()["entries"] == kernel.cache_info()["bytes"] == 0


def test_cache_can_be_disabled():
    kernel = oq.ComputeKernel(oq.ComputePolicy(cache_entries=0))
    for _ in range(3):
        kernel.einsum("ij,jk->ik", np.eye(2), np.eye(2))
    assert kernel.cache_info()["entries"] == 0


def test_large_finite_budgets_do_not_overflow_contraction_planning():
    kernel = oq.ComputeKernel(oq.ComputePolicy(max_memory_mb=1e308))
    np.testing.assert_array_equal(kernel.einsum("ij,jk->ik", np.eye(64), np.eye(64)), np.eye(64))


def test_numeric_failures_are_typed_and_scaled_norms_do_not_overflow():
    kernel = oq.get_compute_kernel()
    assert np.isfinite(kernel.norm(np.array([1e300, 1e300])))
    for matrix in (np.zeros((2, 2)), np.array([[np.nan, 0], [0, 1]])):
        with pytest.raises(oq.NumericalError):
            kernel.solve(matrix, np.ones(2))
    with pytest.raises(oq.NumericalError):
        kernel.svd(np.array([[float("inf")]]))
    with pytest.raises(oq.BackendError):
        validate_state([1e300, 0], oq.Circuit(1))


def test_provider_validation_uses_shared_scale_aware_tolerances():
    oq.configure_compute(policy=oq.ComputePolicy(validation_atol=1e-9, validation_rtol=1e-7))
    observable = oq.PauliString("Z", coefficient=1e9)
    validate_expectations([[1e9 + 1]], 1, (observable,))
    with pytest.raises(oq.BackendError):
        validate_expectations([[1e9 + 200]], 1, (observable,))
    with pytest.raises(oq.BackendError):
        validate_state([1.000001, 0], oq.Circuit(1))


def test_torch_compile_and_gradients_stay_native():
    torch = pytest.importorskip("torch")
    oq.configure_compute(policy=oq.ComputePolicy(max_memory_mb=1))
    a = torch.tensor([[2.0, 0.0], [0.0, 3.0]], dtype=torch.float64, requires_grad=True)

    def objective(value):
        ops = Ops(value)
        result = ops.solve(value, torch.ones((2, 1), dtype=value.dtype))
        return ops.einsum("ij,ij->", result, result)

    compiled = torch.compile(objective, backend="eager", fullgraph=True)
    expected, actual = objective(a), compiled(a)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, a)[0], torch.autograd.grad(expected, a)[0]
    )
    bad = compiled(torch.tensor([[float("nan"), 0.0], [0.0, 1.0]], dtype=torch.float64))
    assert not bool(torch.isfinite(bad))


def test_jax_jit_and_gradients_stay_native():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    oq.configure_compute(policy=oq.ComputePolicy(max_memory_mb=1))

    def objective(value):
        ops = Ops(value)
        solved = ops.solve(value, jnp.ones((2, 1), dtype=value.dtype))
        return ops.einsum("ij,ij->", solved, solved)

    a = jnp.diag(jnp.array([2.0, 3.0]))
    np.testing.assert_allclose(jax.jit(objective)(a), 1 / 4 + 1 / 9, rtol=1e-6)
    assert bool(jnp.isfinite(jax.jit(jax.grad(objective))(a)).all())
    bad = jax.jit(objective)(jnp.diag(jnp.array([float("nan"), 1.0])))
    assert not bool(jnp.isfinite(bad))


def test_jax_vmap_with_closed_over_device_values_preserves_transforms():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    weights = jnp.ones(3)
    ops = _TensorOps("jax", weights)

    def function(row):
        return (ops.array(row, real_value=True) * weights).sum()

    value = jnp.ones((2, 3))
    np.testing.assert_allclose(jax.vmap(function)(value), [3, 3])
    np.testing.assert_allclose(jax.grad(lambda x: jax.vmap(function)(x).sum())(value), 1)


def test_hybrid_classical_projections_preserve_torch_autocast():
    torch = pytest.importorskip("torch")
    plan = oq.HybridArchitecture(
        2,
        1,
        hidden_features=4,
        depth=1,
        quantum=oq.QuantumCellSpec(num_qubits=2),
        projection_policy="dense",
        rematerialize=False,
    )
    model = plan.build("torch", backend="torch", device="cpu", dtype=torch.float32)
    value = torch.ones((3, 2))
    reference = model(value).detach()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(value)
    torch.testing.assert_close(result.float(), reference, atol=0.03, rtol=0.03)
    result.square().sum().backward()
    assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)


def test_hybrid_projections_accept_jax_mixed_precision_inputs():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    plan = oq.HybridArchitecture(2, 1, hidden_features=4, depth=1, projection_policy="dense")
    values = {name: jnp.asarray(value) for name, value in plan.initial_parameters(5).items()}
    result = plan.evaluate(
        values, jnp.ones((3, 2), dtype=jnp.bfloat16), backend="jax", active_depth=0
    )
    assert bool(jnp.isfinite(result).all())
