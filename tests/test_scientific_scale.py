"""Large structured expectation checks are distinct from general dense simulation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import openqml as oq


@pytest.fixture
def science(monkeypatch):
    directory = Path(__file__).parents[1] / "benchmarks"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "scientific_scale_fixture", directory / "scientific_scale.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("case,width", [("ghz", 6), ("bell", 6), ("cluster", 4)])
def test_restricted_families_match_dense_reference_and_small_cut_rank(science, case, width):
    circuit, observables, expected, _, cut_rank = science.workload(oq, case, width)
    plan = oq.PauliPlan(circuit, observables)
    for angle in (0.0, 0.37, -0.4):
        result = plan.evaluate({"theta": angle})
        reference = oq.expectations(circuit, observables, {"theta": angle})
        analytic = [np.cos(angle) if value == "cos" else value for value in expected]
        np.testing.assert_allclose(result, reference, atol=1e-12)
        np.testing.assert_allclose(result, analytic, atol=1e-12)
        state = oq.statevector(circuit, {"theta": angle})
        rank = np.linalg.matrix_rank(state.reshape(2 ** (width // 2), -1), tol=1e-10)
        assert rank == 2**cut_rank


@pytest.mark.parametrize("case,width", [("ghz", 1024), ("bell", 1024), ("cluster", 256)])
@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_large_exact_readouts_and_native_derivatives_never_allocate_dense_states(
    science, monkeypatch, case, width, backend
):
    import openqml.simulator as simulator

    monkeypatch.setattr(
        simulator, "_initial_state", lambda *a, **k: pytest.fail("Allocated a dense state")
    )
    circuit, observables, expected, _, _ = science.workload(oq, case, width)
    plan = oq.PauliPlan(circuit, observables)
    angle = 0.37
    values = [np.cos(angle) if value == "cos" else value for value in expected]
    if backend == "numpy":
        np.testing.assert_allclose(plan.evaluate({"theta": angle}), values, atol=1e-12)
    elif backend == "torch":
        torch = pytest.importorskip("torch")
        value = torch.tensor(angle, dtype=torch.float64, requires_grad=True)
        output = plan.evaluate({"theta": value}, backend="torch", like=value)
        first = torch.autograd.grad(output[0], value, create_graph=True)[0]
        second = torch.autograd.grad(first, value)[0]
        np.testing.assert_allclose(output.detach().numpy(), values, atol=1e-12)
        assert first.item() == pytest.approx(-np.sin(angle), abs=1e-12)
        assert second.item() == pytest.approx(-np.cos(angle), abs=1e-12)
    else:
        jax = pytest.importorskip("jax")
        with oq.jax.enable_x64():

            def function(x):
                return plan.evaluate({"theta": x}, backend="jax", like=x)[0]

            value = oq.jax.numpy.array(angle)
            assert float(jax.jit(jax.grad(function))(value)) == pytest.approx(
                -np.sin(angle), abs=1e-12
            )
            assert float(jax.jit(jax.hessian(function))(value)) == pytest.approx(
                -np.cos(angle), abs=1e-12
            )
    profile = oq.estimate_resources(circuit)
    assert not profile.within_statevector_policy and plan.peak_terms <= 2


def test_structured_hybrid_model_updates_classical_and_quantum_weights(science, monkeypatch):
    torch = pytest.importorskip("torch")
    import openqml.simulator as simulator

    monkeypatch.setattr(
        simulator, "_initial_state", lambda *a, **k: pytest.fail("Allocated dense state")
    )
    circuit, observables, _, _, _ = science.workload(oq, "ghz", 1024)
    angle = oq.Parameter("x") * oq.Parameter("w")
    circuit = circuit.replace_operation(
        (len(circuit.operations) - 1,), oq.Operation("rz", (0,), (angle,))
    )
    layer = oq.torch.QuantumLayer(
        circuit,
        ("x",),
        ("w",),
        observables,
        max_qubits=1024,
        execution_method="pauli",
        dtype=torch.float64,
    )
    encoder = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    head = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        encoder.weight.fill_(0.8)
        layer.weights.fill_(0.6)
        head.weight.copy_(torch.tensor([[0.5, 0.2]], dtype=torch.float64))
    model = torch.nn.Sequential(encoder, layer, head)
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model(torch.tensor([[0.2], [0.4]], dtype=torch.float64)).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert all(
        not torch.equal(old, parameter)
        for old, parameter in zip(before, model.parameters(), strict=True)
    )


def test_pauli_growth_guard_is_a_representation_limit_not_a_universal_capacity_claim(science):
    from types import SimpleNamespace

    result = science.worker(SimpleNamespace(source=Path(oq.__file__).parents[1], case="guard"))
    assert result["status"] == "rejected_by_explicit_budget"
    assert result["codes"] == ["PAULI_TERM_LIMIT"]
