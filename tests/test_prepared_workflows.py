"""Shared preparation must preserve independent results and differentiability."""

from dataclasses import replace

import numpy as np
import pytest

import openqml as oq


def test_parameter_sweeps_share_topology_without_stale_values_or_bound_copies(monkeypatch):
    x = oq.Parameter("x")
    angle = x.sin() + x * x
    circuit = oq.Circuit(3).h(0).cx(0, 2).ry(angle, 1).rz(angle, 2).with_global_phase(x / 3)
    obs = [oq.PauliString("IZI"), oq.PauliString("XIX")]
    rows = [{"x": v} for v in np.linspace(-0.8, 0.9, 259)]
    expected = np.array([oq.expectations(circuit, obs, row, backend="numpy") for row in rows])

    def forbidden(*args, **kwargs):
        pytest.fail("A homogeneous parameter sweep copied a bound circuit")

    with oq.compute_context(policy=oq.ComputePolicy(cache_bytes=0)):
        monkeypatch.setattr(oq.Circuit, "bind", forbidden)
        actual = oq.expectations_batch(circuit, obs, iter(rows), backend="numpy")
        repeated = oq.expectations_batch(circuit, obs, reversed(rows), backend="numpy")
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    np.testing.assert_allclose(repeated, expected[::-1], atol=2e-14)


def test_parameter_batch_preserves_empty_bindings_domains_and_backend_limits(monkeypatch):
    from openqml.runtime import backends
    from openqml.sim import dense

    circuit = oq.Circuit(2).ry(oq.Parameter("x"), 0)
    obs = [oq.PauliString("ZI")]
    assert oq.expectations_batch(circuit, obs, [], backend="numpy").shape == (0, 1)
    for rows in ([{}], [{"x": 0, "extra": 1}], [{"x": float("nan")}], {"x": 0}):
        with pytest.raises(oq.ValidationError):
            oq.expectations_batch(circuit, obs, rows, backend="numpy")
    log = oq.Circuit(2).ry(oq.Parameter("x").log(), 0)
    with np.errstate(invalid="ignore"):
        with pytest.raises(oq.ValidationError):
            oq.expectations_batch(log, obs, [{"x": -1}], backend="numpy")
    device = backends.get_backend("numpy")
    limited = backends._LocalBackend(device.name, replace(device.capabilities, max_qubits=1))
    monkeypatch.setattr(backends, "get_backend", lambda value: limited)
    import openqml.sim.batch as batch

    monkeypatch.setattr(batch, "get_backend", lambda value: limited)
    monkeypatch.setattr(dense, "_initial_state", lambda *a: pytest.fail("Allocated past qubit cap"))
    with pytest.raises(oq.ResourceError, match="limit"):
        oq.expectations_batch(circuit, obs, [{"x": 0}], backend="numpy")


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_native_factorization_reuse_solves_multiple_rhs_and_singular_errors(backend):
    xp = pytest.importorskip("torch" if backend == "torch" else "jax.numpy")
    convert = (lambda x: xp.tensor(x, dtype=xp.float64)) if backend == "torch" else xp.asarray
    matrix, rhs = convert([[2.0, 0.2], [0.3, 1.5]]), convert([[1.0, 2.0], [-1.0, 0.5]])
    solve = oq.get_compute_kernel().factorized_solve(matrix, xp=xp, backend=backend)
    for scale in (1, -0.4):
        actual = solve(rhs * scale)
        np.testing.assert_allclose(matrix @ actual, rhs * scale, rtol=2e-6, atol=2e-6)
    with pytest.raises(oq.NumericalError, match="singular"):
        oq.get_compute_kernel().factorized_solve(
            convert([[1.0, 1.0], [1.0, 1.0]]), xp=xp, backend=backend
        )


def test_woodbury_refinement_factors_once_and_retains_implicit_higher_derivatives(monkeypatch):
    torch = pytest.importorskip("torch")
    from openqml.finance._array import Ops
    from openqml.finance._linear import linear_solve

    generator = torch.Generator().manual_seed(8)
    diagonal = torch.ones(16, dtype=torch.float64, requires_grad=True)
    factors = torch.randn(3, 16, generator=generator, dtype=torch.float64) * 0.1
    mass = torch.ones(3, dtype=torch.float64)
    rhs = torch.randn(16, 2, generator=generator, dtype=torch.float64)
    original, calls = torch.linalg.lu_factor_ex, []

    def counted(value, **options):
        calls.append(value.shape)
        return original(value, **options)

    monkeypatch.setattr(torch.linalg, "lu_factor_ex", counted)
    actual = linear_solve(diagonal, factors, mass, rhs, 64, 1e-10, Ops(diagonal))
    assert calls == [(3, 3)]
    reference = torch.linalg.solve(torch.diag(diagonal) + factors.T @ factors, rhs)
    gradient = torch.autograd.grad(actual.square().sum(), diagonal, create_graph=True)[0]
    expected = torch.autograd.grad(reference.square().sum(), diagonal, create_graph=True)[0]
    torch.testing.assert_close(gradient, expected, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        torch.autograd.grad(gradient.sum(), diagonal)[0],
        torch.autograd.grad(expected.sum(), diagonal)[0],
        rtol=1e-9,
        atol=1e-9,
    )


def test_jax_woodbury_gradients_match_dense_solve_at_zero_scenario_mass():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from openqml.finance._array import Ops
    from openqml.finance._linear import linear_solve

    rng = np.random.default_rng(9)
    factors = jnp.asarray(rng.normal(size=(3, 16)) * 0.1, dtype="float32")
    rhs = jnp.asarray(rng.normal(size=(16, 2)), dtype="float32")
    mass = jnp.array([0.0, 0.5, 1.0])

    def loss(value):
        result = linear_solve(jnp.ones(16), factors, value, rhs, 64, 1e-6, Ops(rhs))
        return jnp.sum(result**2)

    def dense(value):
        matrix = jnp.eye(16) + factors.T @ (value[:, None] * factors)
        return jnp.sum(jnp.linalg.solve(matrix, rhs) ** 2)

    actual = jax.jit(jax.value_and_grad(loss))(mass)
    expected = jax.value_and_grad(dense)(mass)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        jax.jvp(jax.grad(loss), (mass,), (jnp.ones(3),))[1],
        jax.jvp(jax.grad(dense), (mass,), (jnp.ones(3),))[1],
        rtol=2e-5,
        atol=2e-5,
    )
