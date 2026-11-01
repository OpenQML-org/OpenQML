"""Large structured expectation checks are distinct from general dense simulation."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import openqml as oq


def _workload(case, width):
    theta = oq.Parameter("theta")
    builder = oq.CircuitBuilder(width)
    if case == "ghz":
        builder.h(0)
        for q in range(1, width):
            builder.cx(q - 1, q)
        labels, expected = ["X" * width, "Z" + "I" * (width - 2) + "Z"], ["cos", 1]
        structure, cut_rank = "GHZ_chain", 1
    elif case == "bell":
        if width % 2:
            raise ValueError("Crossed Bell pairs require an even width.")
        half = width // 2
        for q in range(half):
            builder.h(q).cx(q, q + half)
        labels = ["X" * width, "Z" * width]
        expected, structure, cut_rank = ["cos", 1], "product_of_crossed_Bell_pairs", half
    else:
        side = math.isqrt(width)
        if side**2 != width:
            raise ValueError("Cluster width must be a square.")
        for q in range(width):
            builder.h(q)
        for row in range(side):
            for col in range(side):
                q = row * side + col
                if col + 1 < side:
                    builder.cz(q, q + 1)
                if row + 1 < side:
                    builder.cz(q, q + side)
        labels, expected = [], []
        for q in sorted({0, width // 2, width - 1}):
            label, row, col = ["I"] * width, q // side, q % side
            label[q] = "X"
            for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= r < side and 0 <= c < side:
                    label[r * side + c] = "Z"
            labels.append("".join(label))
            expected.append("cos" if q == 0 else 1)
        # Products of one checkerboard's stabilizers have no internal graph
        # edges. They retain a positive sign and probe the full grid collectively.
        for color in (0, 1):
            label = []
            for q in range(width):
                row, col = divmod(q, side)
                if (row + col) % 2 == color:
                    label.append("X")
                else:
                    degree = sum(
                        0 <= r < side and 0 <= c < side
                        for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
                    )
                    label.append("Z" if degree % 2 else "I")
            labels.append("".join(label))
            expected.append("cos" if color == 0 else 1)
        structure, cut_rank = "connected_square_cluster_graph", side if side % 2 == 0 else None
    builder.rz(theta, 0)
    return builder.build(), tuple(oq.PauliString(p) for p in labels), expected, structure, cut_rank


@pytest.fixture
def science():
    return SimpleNamespace(workload=lambda _, case, width: _workload(case, width))


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


def test_pauli_growth_guard_is_a_representation_limit_not_a_universal_capacity_claim():
    x, builder = oq.Parameter("x"), oq.CircuitBuilder(12)
    for _ in range(3):
        for q in range(12):
            builder.ry((q + 1) * x, q)
        for q in range(11):
            builder.cz(q, q + 1)
    with pytest.raises(oq.ConversionError) as caught:
        oq.PauliPlan(builder.build(), (oq.PauliString("Z" * 12),), max_terms=8, max_work=5000)
    assert [d.code for d in caught.value.diagnostics] == ["PAULI_TERM_LIMIT"]
