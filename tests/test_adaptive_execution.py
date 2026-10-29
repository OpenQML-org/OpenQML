"""Representation selection, independent small references and admission limits."""

import math
import random

import numpy as np
import pytest

import openqml as oq


def nonclifford_chain(width, depth=4, seed=7):
    rng = random.Random(seed)
    builder = oq.CircuitBuilder(width)
    for layer in range(depth):
        for wire in range(width):
            builder.ry(rng.uniform(-1, 1), wire).rz(rng.uniform(-1, 1), wire)
        for wire in range(layer % 2, width - 1, 2):
            builder.cx(wire, wire + 1)
    return builder.build()


@pytest.mark.parametrize("seed", range(5))
def test_random_dense_mps_and_pauli_readouts_agree(seed):
    circuit = nonclifford_chain(6, 2, seed)
    observables = (oq.PauliString("XYZXYZ"), oq.PauliString("Z" * 6))
    reference = oq.expectations(circuit, observables)
    for method in ("mps", "statevector", "density", "pauli"):
        plan = oq.plan_execution(circuit, observables, method=method, max_memory_mb=64)
        assert plan.supported and plan.selected_method == method
        np.testing.assert_allclose(plan.evaluate(), reference, atol=1e-12)


def test_small_clifford_auto_queries_avoid_fixed_sdk_startup(monkeypatch):
    monkeypatch.setattr(
        oq.AerBackend, "_engine", lambda *a, **k: pytest.fail("Unnecessary SDK startup")
    )
    plan = oq.plan_execution(
        oq.Circuit(2).h(0).cx(0, 1), (oq.PauliString("XX"), oq.PauliString("ZZ")), max_memory_mb=16
    )
    assert plan.selected_method == "pauli"
    np.testing.assert_allclose(plan.evaluate(), (1, 1), atol=1e-12)


def test_nonclifford_wide_chain_uses_mps_after_expansion_rejection(monkeypatch):
    import openqml.simulator as simulator

    monkeypatch.setattr(
        simulator, "_initial_state", lambda *a, **k: pytest.fail("Dense allocation")
    )
    circuit = nonclifford_chain(96)
    observables = (oq.PauliString("Z" * 96), oq.PauliString("X" * 96))
    plan = oq.plan_execution(circuit, observables, max_memory_mb=64, max_pauli_terms=4)
    assert plan.selected_method == "mps"
    assert any(not c.eligible and c.method == "pauli" for c in plan.candidates)
    assert plan.mps_profile["max_bond"] <= 256
    assert isinstance(plan._runner, oq.MPSBackend)
    assert plan.mps_profile["truncation"] == "none_full_SVD_columns_retained"
    values = plan.evaluate()
    assert np.isfinite(values).all() and (np.abs(values) <= 1 + 1e-12).all()
    reference = oq.expectations(circuit, observables, backend="aer-mps")
    np.testing.assert_allclose(values, reference, atol=1e-12)


def test_stabilizer_matches_exact_pauli_reference_on_nonlocal_clifford_circuit(monkeypatch):
    import openqml.simulator as simulator

    monkeypatch.setattr(
        simulator, "_initial_state", lambda *a, **k: pytest.fail("Dense allocation")
    )
    rng = random.Random(31)
    builder = oq.CircuitBuilder(128)
    for _ in range(600):
        a, b = rng.sample(range(128), 2)
        builder.h(a).s(b).cx(a, b)
    circuit = builder.build()
    observables = tuple(
        oq.PauliString("".join(rng.choice("IXYZ") for _ in range(128))) for _ in range(7)
    )
    plan = oq.plan_execution(circuit, observables, max_memory_mb=16)
    assert plan.selected_method == "stabilizer"
    np.testing.assert_allclose(
        plan.evaluate(), oq.PauliPlan(circuit, observables).evaluate(), atol=1e-12
    )
    assert oq.get_backend("aer-stabilizer").method == "stabilizer"


def test_adaptive_backend_routes_existing_numeric_and_gradient_apis():
    theta = oq.Parameter("theta")
    circuit = oq.Circuit(2).ry(theta, 0).cx(0, 1)
    observable = oq.PauliString("ZI")
    assert oq.expectation(circuit, observable, {"theta": 0.3}, backend="auto") == pytest.approx(
        np.cos(0.3)
    )
    jacobian = oq.parameter_shift_jacobian(circuit, (observable,), [{"theta": 0.3}], backend="auto")
    np.testing.assert_allclose(jacobian, [[[-np.sin(0.3)]]], atol=1e-12)
    assert oq.get_backend("auto").capabilities.deterministic_expectations


def test_mps_admission_rejects_deep_and_nonlocal_inputs_without_execution(monkeypatch):
    def fail(*a, **k):
        pytest.fail("An inadmissible circuit reached a simulator")

    monkeypatch.setattr(oq.MPSBackend, "expectations_batch", fail)
    for circuit in (nonclifford_chain(80, 50), oq.Circuit(80).ry(0.3, 0).cx(0, 79)):
        plan = oq.plan_execution(
            circuit, (oq.PauliString("Z" * 80),), method="mps", max_memory_mb=16
        )
        assert not plan.supported
        with pytest.raises(oq.ResourceError, match="MPS|mps"):
            plan.evaluate()


def test_schmidt_bounds_cover_small_dense_ranks_at_every_cut():
    circuit = nonclifford_chain(8)
    plan = oq.plan_execution(circuit, (oq.PauliString("Z" * 8),), method="mps", max_memory_mb=16)
    state = oq.statevector(circuit)
    ranks = [np.linalg.matrix_rank(state.reshape(1 << cut, -1), tol=1e-12) for cut in range(1, 8)]
    assert max(ranks) <= plan.mps_profile["max_bond"]


def test_memory_derived_dense_capacity_has_no_fixed_twenty_wire_ceiling(monkeypatch):
    import openqml.simulator as simulator

    class ReachedAllocation(Exception):
        pass

    def allocation(*a, **k):
        raise ReachedAllocation

    monkeypatch.setattr(simulator, "_initial_state", allocation)
    with pytest.raises(ReachedAllocation):
        oq.statevector(oq.Circuit(24), max_memory_mb=1024)
    with pytest.raises(oq.ResourceError):
        oq.statevector(oq.Circuit(24), max_memory_mb=math.nextafter(1024, 0))
    assert oq.estimate_resources(oq.Circuit(22)).within_statevector_policy
    assert not oq.estimate_resources(oq.Circuit(23)).within_statevector_policy


def test_density_memory_boundary_uses_exact_integer_arithmetic():
    from openqml.density import _resource_check

    _resource_check(oq.Circuit(11), None, 256)
    with pytest.raises(oq.ResourceError):
        _resource_check(oq.Circuit(11), None, math.nextafter(256, 0))


def test_automatic_budget_observes_container_available_memory(monkeypatch):
    import openqml.adaptive as adaptive

    monkeypatch.setattr(adaptive, "_available_memory", lambda: 12 * 2**30)
    plan = oq.plan_execution(oq.Circuit(30), (oq.PauliString("Z" * 30),))
    assert plan.max_memory_mb == 3072
    monkeypatch.setattr(adaptive, "_available_memory", lambda: 100 * 2**30)
    assert oq.plan_execution(oq.Circuit(1), (oq.PauliString("Z"),)).max_memory_mb == 4096


def test_unavailable_gpu_is_visible_and_never_silently_required(monkeypatch):
    import openqml.adaptive as adaptive

    monkeypatch.setattr(adaptive, "_gpu_status", lambda: (False, 0))
    c, obs = oq.Circuit(3).ry(0.31, 0), (oq.PauliString("ZII"),)
    required = oq.plan_execution(c, obs, method="statevector", device="GPU", max_memory_mb=16)
    assert not required.supported
    automatic = oq.plan_execution(c, obs, method="statevector", device="auto", max_memory_mb=16)
    assert automatic.selected_method == "statevector"
    assert not automatic.candidates[-1].eligible


def test_cuda_dense_adaptive_execution_matches_cpu_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA device is required for GPU execution validation")
    c = nonclifford_chain(5, 3)
    obs = (oq.PauliString("XYZXY"),)
    plan = oq.plan_execution(c, obs, method="statevector", device="GPU", max_memory_mb=16)
    assert plan.selected_method == "statevector-gpu"
    np.testing.assert_allclose(plan.evaluate(), oq.expectations(c, obs), atol=1e-12)
    c = oq.Circuit(2).ry(oq.Parameter("theta"), 0).cx(0, 1)
    plan = oq.plan_execution(
        c, (oq.PauliString("ZI"),), method="statevector", device="GPU", max_memory_mb=16
    )
    value = torch.tensor(0.31, device="cuda", dtype=torch.float64, requires_grad=True)
    result = plan.evaluate({"theta": value}, backend="torch", like=value)[0]
    first = torch.autograd.grad(result, value, create_graph=True)[0]
    second = torch.autograd.grad(first, value)[0]
    assert second.item() == pytest.approx(-np.cos(0.31), abs=1e-12)


def test_noise_and_feedback_are_budgeted_as_density_not_assumed_clifford():
    c = (
        oq.Circuit(2)
        .h(0)
        .measure(0, 0)
        .if_else(oq.ClassicalCondition((0,), 1), oq.Circuit(2).x(1))
        .amplitude_damping(0.13, 1)
    )
    obs = (oq.PauliString("ZI"), oq.PauliString("IZ"))
    plan = oq.plan_execution(c, obs, max_memory_mb=16)
    assert plan.selected_method == "density"
    np.testing.assert_allclose(plan.evaluate(), oq.expectations(c, obs), atol=1e-12)
    with pytest.raises(oq.ValidationError):
        oq.plan_execution(c, obs, max_memory_mb=0)


def test_rejected_explicit_width_limit_is_inspectable():
    plan = oq.plan_execution(oq.Circuit(50), (oq.PauliString("Z" * 50),), max_qubits=40)
    assert not plan.supported and "max_qubits" in plan.to_dict()["candidates"][0]["reason"]


def test_mps_reversed_wire_order_and_opaque_two_site_matrix_match_dense():
    rng = np.random.default_rng(3)
    matrix = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    circuit = oq.Circuit(4).ry(0.31, 1).rx(-0.17, 2).cy(2, 1).crx(0.41, 3, 2).unitary(matrix, 1, 0)
    obs = (oq.PauliString("XYZI"), oq.PauliString("ZZXY"))
    np.testing.assert_allclose(
        oq.expectations(circuit, obs, backend="mps"), oq.expectations(circuit, obs), atol=1e-12
    )


def test_mps_hybrid_gradients_match_native_torch_and_share_one_optimizer():
    torch = pytest.importorskip("torch")
    spec = oq.data_reuploading(3, depth=1)
    native = oq.torch.QuantumLayer.from_spec(spec, dtype=torch.float64)
    numeric = oq.torch.QuantumLayer.from_spec(spec, dtype=torch.float64, backend="mps")
    with torch.no_grad():
        numeric.weights.copy_(native.weights)
    x = torch.tensor([[0.2, -0.1, 0.3], [-0.3, 0.4, 0.2]], dtype=torch.float64, requires_grad=True)
    expected = native(x).square().sum()
    expected_grads = torch.autograd.grad(expected, (x, native.weights))
    actual = numeric(x).square().sum()
    actual_grads = torch.autograd.grad(actual, (x, numeric.weights))
    for left, right in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 3, dtype=torch.float64),
        numeric,
        torch.nn.Linear(3, 1, dtype=torch.float64),
    )
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    model(x.detach()).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


@pytest.mark.parametrize("framework", ("torch", "jax"))
def test_adaptive_pauli_native_autodiff_retains_second_derivatives(framework):
    angle = oq.Parameter("theta")
    circuit = oq.Circuit(2).ry(angle, 0).cx(0, 1)
    plan = oq.plan_execution(circuit, (oq.PauliString("ZI"),), max_memory_mb=16)
    assert plan.selected_method == "pauli"
    if framework == "torch":
        value = oq.torch.tensor(0.31, requires_grad=True, dtype=oq.torch.float64)
        result = plan.evaluate({"theta": value}, backend="torch", like=value)[0]
        first = oq.torch.autograd.grad(result, value, create_graph=True)[0]
        second = oq.torch.autograd.grad(first, value)[0]
        assert second.item() == pytest.approx(-np.cos(0.31), abs=1e-12)
    else:
        with oq.jax.enable_x64():
            value = oq.jax.numpy.asarray(0.31)
            second = oq.jax.jit(
                oq.jax.hessian(lambda x: plan.evaluate({"theta": x}, backend="jax", like=x)[0])
            )(value)
            assert float(second) == pytest.approx(-np.cos(0.31), abs=1e-12)


def test_wide_mps_hybrid_layer_uses_numeric_gradients_without_dense_allocation(monkeypatch):
    import openqml.simulator as simulator

    monkeypatch.setattr(
        simulator, "_initial_state", lambda *a, **k: pytest.fail("Dense allocation")
    )
    builder = oq.CircuitBuilder(64).ry(oq.Parameter("x") * oq.Parameter("w"), 0)
    for wire in range(63):
        builder.cx(wire, wire + 1)
    layer = oq.torch.QuantumLayer(
        builder.build(),
        ("x",),
        ("w",),
        (oq.PauliString("Z" + "I" * 63),),
        backend="mps",
        dtype=oq.torch.float64,
    )
    with oq.torch.no_grad():
        layer.weights.fill_(0.7)
    value = oq.torch.tensor([[0.31]], dtype=oq.torch.float64)
    output = layer(value)[0, 0]
    output.backward()
    assert output.item() == pytest.approx(np.cos(0.31 * 0.7), abs=1e-12)
    assert layer.weights.grad[0].item() == pytest.approx(-0.31 * np.sin(0.31 * 0.7), abs=1e-12)


def test_layer_default_width_is_optional_and_compiled_growth_remains_bounded(monkeypatch):
    import openqml.simulator as simulator

    monkeypatch.setattr(
        simulator, "_initial_state", lambda *a, **k: pytest.fail("Dense allocation")
    )
    angle = oq.Parameter("x") * oq.Parameter("w")
    layer = oq.torch.QuantumLayer(
        oq.Circuit(64).ry(angle, 0),
        ("x",),
        ("w",),
        (oq.PauliString("Z" + "I" * 63),),
        execution_method="pauli",
        dtype=oq.torch.float64,
    )
    value = oq.torch.tensor([[0.31]], dtype=oq.torch.float64)
    assert layer.limits["max_qubits"] is None
    oq.torch.testing.assert_close(layer(value)[0, 0], oq.torch.cos(value[0, 0] * layer.weights[0]))
