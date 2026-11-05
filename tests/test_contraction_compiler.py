"""Exact tree surgery, slice-aware common subtrees and native derivative contracts."""

import gc
import sys

import numpy as np
import pytest

import openqml as q
from openqml._contraction_program import ContractionKernel
from openqml.simulator import _TensorOps


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    if "jax" in sys.modules:
        sys.modules["jax"].clear_caches()
    gc.collect()


def circuit(seed=1, n=6, depth=6):
    rng = np.random.default_rng(seed)
    builder = q.CircuitBuilder(n)
    for _ in range(depth):
        for j in range(n):
            builder.ry(float(rng.normal()), j)
        order = rng.permutation(n)
        for a, b in zip(order[::2], order[1::2], strict=True):
            builder.rxx(float(rng.normal()), int(a), int(b))
    return builder.build()


@pytest.mark.parametrize("seed", range(4))
def test_portfolio_reduces_arithmetic_and_preserves_independent_dense_values(seed):
    c, obs = circuit(seed), (q.PauliString("ZZZZZZ"), q.PauliString("XYZXYZ"))
    greedy = q.TensorNetworkPlan(c, obs)
    optimized = q.TensorNetworkPlan(c, obs, optimizer="portfolio")
    before = sum(g[6] for g in greedy._graphs)
    after = sum(g[6] for g in optimized._graphs)
    assert after < before
    expected = q.expectations(c, obs, backend="numpy")
    np.testing.assert_allclose(optimized.evaluate(), expected, atol=3e-14)
    assert optimized.to_dict() == q.TensorNetworkPlan(c, obs, optimizer="portfolio").to_dict()
    assert optimized.working_bytes <= optimized.max_memory_mb * 2**20


def test_compiled_gemm_handles_noncontiguous_modes_and_broadcast_batches():
    rng = np.random.default_rng(3)
    left = rng.normal(size=(2, 1, 2, 2, 2)) + 1j * rng.normal(size=(2, 1, 2, 2, 2))
    right = rng.normal(size=(1, 3, 2, 2, 2)) + 1j * rng.normal(size=(1, 3, 2, 2, 2))
    kernel = ContractionKernel.compile((3, 1, 4), (2, 4, 1), (1, 4))
    actual = kernel.apply(left, right, _TensorOps("numpy"))
    expected = np.einsum("...abc,...dcb->...ad", left, right)
    np.testing.assert_allclose(actual, expected, atol=3e-14)
    scalar = ContractionKernel.compile((), (), ())
    assert scalar.apply(np.asarray(2j), np.asarray(3j), _TensorOps("numpy")) == -6


def shared_circuit():
    a, b = q.Parameter("a"), q.Parameter("b")
    c = q.Circuit(4).ry(a, 0).ry(a, 1).rxx(b, 0, 1).ry(a, 2).cx(1, 2).cx(2, 3)
    obs = (q.PauliString("ZZZZ"), q.PauliString("ZIII"), q.PauliString("XXII"))
    return c, obs


def test_demand_reuse_skips_subtrees_and_never_keeps_previous_bindings():
    c, obs = shared_circuit()
    baseline = q.TensorNetworkPlan(c, obs)
    reused = q.TensorNetworkPlan(c, obs, optimizer="portfolio", reuse_intermediates=True)
    for angle in (0.2, -0.4, 0.7):
        bindings = {"a": np.array([angle, angle + 0.1]), "b": 0.3}
        expected, first = baseline.evaluate(bindings, return_report=True)
        actual, report = reused.evaluate(bindings, return_report=True)
        np.testing.assert_allclose(actual, expected, atol=2e-14)
        assert report["subtree_cache_hits"] > 0
        assert report["contractions_executed"] < first["contractions_executed"]
        assert 0 < report["cache_peak_bytes"] <= report["cache_budget_bytes"]


@pytest.mark.parametrize("reuse", [False, True])
def test_portfolio_slicing_preserves_every_assignment_under_memory_pressure(reuse):
    c = circuit(0, n=10, depth=8)
    obs = (q.PauliString("Z" * 10), q.PauliString("X" + "Z" * 9))
    whole = q.TensorNetworkPlan(c, obs, max_memory_mb=64, max_flops=10**11)
    budget = (whole._metadata_bytes + 0.55 * (whole.working_bytes - whole._metadata_bytes)) / 2**20
    sliced = q.TensorNetworkPlan(
        c,
        obs,
        optimizer="portfolio",
        reuse_intermediates=reuse,
        max_memory_mb=budget,
        max_slices=256,
        max_flops=10**11,
    )
    assert any(len(g[4]) for g in sliced._graphs)
    actual, report = sliced.evaluate(return_report=True)
    np.testing.assert_allclose(actual, q.expectations(c, obs, backend="numpy"), atol=2e-13)
    assert report["slices_evaluated"] == sum(1 << len(g[4]) for g in sliced._graphs)
    assert sliced.working_bytes <= budget * 2**20


@pytest.mark.parametrize("rematerialize", [False, True])
def test_torch_reused_subtree_gradients_hessians_and_report_snapshot(rematerialize):
    torch = pytest.importorskip("torch")
    c, obs = shared_circuit()
    plan = q.TensorNetworkPlan(
        c,
        obs,
        optimizer="portfolio",
        reuse_intermediates=True,
        rematerialize=rematerialize,
    )
    values = torch.tensor([0.2, 0.4], dtype=torch.float64, requires_grad=True)

    def evaluate(v):
        return plan.evaluate({"a": v[0], "b": v[1]}, backend="torch", like=v)

    assert torch.autograd.gradcheck(evaluate, (values,))
    assert torch.autograd.gradgradcheck(evaluate, (values,))
    actual, report = plan.evaluate(
        {"a": values[0], "b": values[1]},
        backend="torch",
        like=values,
        return_report=True,
    )
    snapshot = dict(report)
    gradient = torch.autograd.grad(actual.square().sum(), values)[0]
    reference = q.TensorNetworkPlan(c, obs).evaluate(
        {"a": values[0], "b": values[1]},
        backend="torch",
        like=values,
    )
    torch.testing.assert_close(
        gradient,
        torch.autograd.grad(reference.square().sum(), values)[0],
        atol=1e-13,
        rtol=1e-12,
    )
    assert snapshot == report


def test_jax_reuse_jit_vmap_hessian_and_rematerialization():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    c, obs = shared_circuit()
    plan = q.TensorNetworkPlan(
        c, obs, optimizer="portfolio", reuse_intermediates=True, rematerialize=True
    )
    dense = q.TensorNetworkPlan(c, obs)

    def evaluate(v, current):
        return current.evaluate({"a": v[0], "b": v[1]}, backend="jax", like=v).sum()

    value = jnp.asarray([0.2, 0.4])

    def scalar(v):
        return evaluate(v, plan)

    def reference(v):
        return evaluate(v, dense)

    np.testing.assert_allclose(jax.jit(scalar)(value), reference(value), atol=2e-6)
    np.testing.assert_allclose(jax.hessian(scalar)(value), jax.hessian(reference)(value), atol=5e-6)
    batch = jnp.stack((value, value + 0.1))
    np.testing.assert_allclose(
        jax.jit(jax.vmap(scalar))(batch), jax.vmap(reference)(batch), atol=2e-6
    )


def test_cache_eviction_changes_work_only(monkeypatch):
    from openqml._contraction_reuse import ContractionReuse

    original = ContractionReuse.__init__
    monkeypatch.setattr(ContractionReuse, "__init__", lambda self, budget: original(self, 400))
    c, obs = shared_circuit()
    plan = q.TensorNetworkPlan(c, obs, optimizer="portfolio", reuse_intermediates=True)
    binding = {"a": 0.5, "b": -0.4}
    actual, report = plan.evaluate(binding, return_report=True)
    np.testing.assert_allclose(actual, q.expectations(c, obs, binding, backend="numpy"), atol=2e-14)
    assert report["cache_peak_bytes"] <= 400


def test_portfolio_and_reporting_controls_are_admitted():
    c, obs = shared_circuit()
    with pytest.raises(q.ResourceError, match="max_work"):
        q.TensorNetworkPlan(c, obs, optimizer="portfolio", max_work=20)
    plan = q.TensorNetworkPlan(q.Circuit(1), (q.PauliString("I"),), optimizer="portfolio")
    result, report = plan.evaluate(return_report=True)
    np.testing.assert_array_equal(result, [1])
    assert report["contractions_executed"] == 0
    with pytest.raises(q.ValidationError, match="return_report"):
        plan.evaluate(return_report=1)
