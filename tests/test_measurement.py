import numpy as np
import pytest

from openqml import (
    BackendCapabilities,
    Circuit,
    Hamiltonian,
    Parameter,
    PauliString,
    ShotBackend,
    ValidationError,
    estimate,
    expectations_batch,
    measurement_plan,
    parameter_shift_jacobian,
    parameter_shift_vjp,
)


def test_shared_shots_capture_cross_output_covariance_and_identity():
    circuit = Circuit(2).h(0).cx(0, 1)
    obs = (
        PauliString("ZI"),
        PauliString("IZ"),
        Hamiltonian((PauliString("ZI", 2), PauliString("IZ", -0.5), PauliString("II", 0.7))),
    )
    result = estimate(circuit, obs, shots=501, seed=19)
    assert len(result.plan.groups) == 1 and result.shots_per_group == (501,)
    assert result.execution_count == 1 and result.shots_per_circuit == 501
    np.testing.assert_allclose(result.values[0, 0], result.values[0, 1], atol=1e-15)
    assert result.values[0, 2] == pytest.approx(0.7 + 1.5 * result.values[0, 0])
    assert result.covariance[0, 0, 1] == pytest.approx(result.covariance[0, 0, 0])
    assert result.covariance[0, 0, 2] == pytest.approx(1.5 * result.covariance[0, 0, 0])
    expected = (1 - result.values[0, 0] ** 2) / 500
    assert result.standard_errors[0, 0] ** 2 == pytest.approx(expected)
    identity = estimate(circuit, (PauliString("II", 3),), shots=1)
    assert identity.execution_count == 0 and identity.shots_per_circuit == 0
    np.testing.assert_array_equal(identity.values, [[3]])
    np.testing.assert_array_equal(identity.standard_errors, [[0]])


def test_pauli_bases_budget_and_uncertainty_against_exact():
    circuit = Circuit(1).ry(Parameter("x"), 0).rz(0.43, 0)
    obs = (PauliString("X"), PauliString("Y"), PauliString("Z"))
    rows = [{"x": 0.37}, {"x": -0.71}]
    sampled = estimate(circuit, obs, rows, shots=30001, seed=31)
    assert sum(sampled.shots_per_group) == 30001
    assert len(sampled.plan.groups) == 3
    exact = expectations_batch(circuit, obs, rows)
    assert np.all(np.abs(sampled.values - exact) < 5 * sampled.standard_errors + 1e-4)
    repeat = estimate(circuit, obs, rows, shots=30001, seed=31)
    np.testing.assert_array_equal(repeat.values, sampled.values)
    np.testing.assert_array_equal(repeat.covariance, sampled.covariance)
    with pytest.raises(ValidationError, match="At least 6 shots"):
        estimate(circuit, obs, rows, shots=5)
    plan = measurement_plan((Hamiltonian((PauliString("X"), PauliString("X", -1))),))
    assert plan.groups == ()


def test_shot_backend_shift_gradients_with_nonlinear_chain_rule():
    circuit = Circuit(1).ry(Parameter("w") * Parameter("x"), 0)
    rows = [{"w": 0.63, "x": 0.41}, {"w": -0.71, "x": 0.37}]
    obs = (PauliString("Z"),)
    backend = ShotBackend(shots=20001, seed=17)
    exact = parameter_shift_jacobian(circuit, obs, rows)
    measured = parameter_shift_jacobian(circuit, obs, rows, backend=backend, chunk_size=4)
    np.testing.assert_allclose(exact, measured, atol=0.02)


def test_vjp_matches_jacobian_and_bounds_queued_jobs():
    class Counting:
        name = "counting"
        capabilities = BackendCapabilities(native_expectations=True)

        def __init__(self):
            self.batch_sizes = []

        def expectations_batch(self, circuits, observables, *, options):
            self.batch_sizes.append(len(circuits))
            return expectations_batch(circuits, observables)

    circuit = Circuit(2).h(0).ry(Parameter("x").sin() * Parameter("w"), 1)
    circuit = circuit.crx(Parameter("x") ** 2, 0, 1).rz(Parameter("frozen"), 0)
    rows = [{"x": x, "w": 0.73, "frozen": -0.23} for x in [0.17, 0.41, -0.63]]
    obs = (PauliString("XY"), PauliString("ZZ"), PauliString("YZ"))
    cotangent = np.random.default_rng(7).normal(size=(3, 3))
    jac = parameter_shift_jacobian(circuit, obs, rows)
    backend = Counting()
    vjp = parameter_shift_vjp(circuit, obs, rows, cotangent, backend=backend, chunk_size=5)
    np.testing.assert_allclose(vjp, np.einsum("bo,bop->bp", cotangent, jac), atol=1e-12)
    assert max(backend.batch_sizes) <= 5 and len(backend.batch_sizes) > 1
    selected = ("w", "x")
    restricted = parameter_shift_vjp(circuit, obs, rows, cotangent, parameters=selected)
    np.testing.assert_allclose(restricted, vjp[:, [circuit.parameters.index(p) for p in selected]])
    zero = parameter_shift_vjp(circuit, obs, rows, cotangent, parameters=())
    assert zero.shape == (3, 0)
    with pytest.raises(ValidationError, match="Cotangents"):
        parameter_shift_vjp(circuit, obs, rows, np.zeros((1, 3)))
    with pytest.raises(ValidationError, match="unique"):
        parameter_shift_vjp(circuit, obs, rows, cotangent, parameters=("w", "w"))


def test_shot_layer_trains_quantum_and_classical_weights_together():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    torch.manual_seed(31)
    layer = QuantumLayer.hardware_efficient(2, 1, backend=ShotBackend(shots=512, seed=7))
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), layer, torch.nn.Linear(2, 1))
    x = torch.rand(8, 2)
    target = x[:, :1] - 0.4 * x[:, 1:]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    losses = []
    for _ in range(35):
        optimizer.zero_grad()
        loss = (model(x) - target).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.3
    assert layer.weights.grad is not None and torch.isfinite(layer.weights.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
