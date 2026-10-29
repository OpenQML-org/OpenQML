"""Independent numerical and statistical checks of the research-backed additions."""

import numpy as np
import pytest

import openqml as oq


def finite_metric(circuit, bindings, names):
    state = oq.statevector(circuit.bind(bindings))
    derivatives = []
    for name in names:
        plus, minus = dict(bindings), dict(bindings)
        plus[name] += 1e-6
        minus[name] -= 1e-6
        derivatives.append(
            (oq.statevector(circuit.bind(plus)) - oq.statevector(circuit.bind(minus))) / 2e-6
        )
    jacobian = np.asarray(derivatives).T
    projected = jacobian - state[:, None] * (state.conj() @ jacobian)[None, :]
    return np.real(jacobian.conj().T @ projected)


@pytest.mark.parametrize("global_phase", [False, True])
def test_metric_products_match_independent_finite_state_derivatives(global_phase):
    x, y, z = (oq.Parameter(n) for n in ("x", "y", "z"))
    circuit = oq.Circuit(2).ry(x.sin() + y * x, 0).cx(0, 1).u(z, x / 2, y, 1).crx(x * z, 1, 0)
    if global_phase:
        circuit = circuit.with_global_phase(7 * x + y.sin())
    names, bindings = ("z", "x", "y"), {"x": 0.21, "y": -0.33, "z": 0.47}
    plan = oq.quantum_metric(circuit, parameters=names)
    expected = finite_metric(circuit, bindings, names)
    np.testing.assert_allclose(plan.matrix(bindings), expected, atol=1e-9)
    np.testing.assert_allclose(
        plan.matvec(bindings, [0.3, -0.4, 0.9]), expected @ [0.3, -0.4, 0.9], atol=1e-9
    )
    assert np.linalg.eigvalsh(plan.matrix(bindings)).min() > -1e-12


def test_damped_matrix_free_solve_matches_explicit_linear_system(monkeypatch):
    x, y = oq.Parameter("x"), oq.Parameter("y")
    circuit = oq.Circuit(1).ry(x, 0).rz(y, 0)
    plan = oq.quantum_metric(circuit)
    bindings, gradient = {"x": 0.7, "y": 0.3}, np.array([0.2, -0.4])
    expected = np.linalg.solve(plan.matrix(bindings) + 0.03 * np.eye(2), gradient)
    monkeypatch.setattr(
        oq.QuantumMetricPlan, "matrix", lambda *a, **k: pytest.fail("Dense metric constructed")
    )
    result = plan.natural_gradient(bindings, gradient, damping=0.03)
    assert result.converged and result.residual_norm < 1e-10
    np.testing.assert_allclose(result.direction, expected, atol=1e-10)


def test_metric_gauge_null_directions_and_layer_weight_ownership():
    x, y, phase = oq.Parameter("x"), oq.Parameter("y"), oq.Parameter("phase")
    circuit = oq.Circuit(1).ry(x, 0).rz(y, 0).with_global_phase(phase)
    plan = oq.quantum_metric(circuit)
    bindings = {"x": 0.0, "y": 0.2, "phase": 0.8}
    metric = plan.matrix(bindings)
    phase_index = plan.parameters.index("phase")
    assert np.max(np.abs(metric[phase_index])) < 1e-15
    spec = oq.LayerSpec(circuit, ("x",), ("y", "phase"))
    assert spec.metric_plan().parameters == ("y", "phase")
    assert oq.quantum_metric(oq.Circuit(1)).matrix({}).shape == (0, 0)


def test_metric_invalid_domains_and_resource_limits():
    x = oq.Parameter("x")
    with pytest.raises(oq.ValidationError):
        oq.quantum_metric("not a circuit")
    with pytest.raises(oq.ValidationError):
        oq.quantum_metric(oq.Circuit(1).reset(0))
    with pytest.raises(oq.ResourceError):
        oq.quantum_metric(oq.Circuit(30).ry(x, 0), max_memory_mb=1)
    plan = oq.quantum_metric(oq.Circuit(1).ry(x, 0))
    with pytest.raises(oq.ValidationError):
        plan.matvec({"x": [0.1, 0.2]}, [1.0])
    with pytest.raises(oq.ValidationError):
        plan.natural_gradient({"x": 0.1}, [1], damping=0)
    with pytest.raises(oq.ValidationError):
        plan.matvec({"x": 0.1}, [1, 2])


def test_adaptive_shot_budget_reproducibility_and_variance_allocation():
    observables = [oq.PauliString("Z"), oq.PauliString("X")]
    a = oq.estimate_adaptive([oq.Circuit(1)], observables, shots=2000, pilot_shots=32, seed=17)
    b = oq.estimate_adaptive([oq.Circuit(1)], observables, shots=2000, pilot_shots=32, seed=17)
    assert a.shots_per_circuit == 2000
    np.testing.assert_array_equal(a.values, b.values)
    assert a.production.shots_per_group == b.production.shots_per_group
    groups = {
        g.basis: n
        for g, n in zip(a.production.plan.groups, a.production.shots_per_group, strict=True)
    }
    assert groups["X"] > 5 * groups["Z"]
    assert a.values[0, 0] == 1 and abs(a.values[0, 1]) < 5 * a.standard_errors[0, 1]
    assert a.execution_count == 4


def test_adaptive_production_does_not_reuse_pilot_samples(monkeypatch):
    import openqml.measurement as measurement

    original, called = measurement.estimate, []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        called.append((kwargs, result))
        return result

    monkeypatch.setattr(measurement, "estimate", capture)
    result = oq.estimate_adaptive(
        [oq.Circuit(1).h(0)], [oq.PauliString("Z")], shots=100, pilot_shots=10, seed=2
    )
    assert len(called) == 2 and called[0][0]["seed"] != called[1][0]["seed"]
    assert called[0][1].shots_per_circuit == 10
    assert result.production is called[1][1] and result.production.shots_per_circuit == 90


def test_adaptive_identity_and_budget_validation():
    result = oq.estimate_adaptive([oq.Circuit(1)], [oq.PauliString("I", 2)], shots=100)
    assert result.shots_per_circuit == 0 and result.values[0, 0] == 2
    for kwargs in ({"shots": 10, "pilot_shots": 10}, {"variance_floor": 0}, {"shots": 2**65}):
        with pytest.raises(oq.ValidationError):
            oq.estimate_adaptive([oq.Circuit(1)], [oq.PauliString("Z")], **kwargs)


def random_local(seed, width=5):
    rng = np.random.default_rng(seed)
    builder = oq.CircuitBuilder(width)
    for _ in range(3):
        for q in range(width):
            builder.ry(float(rng.normal()), q).rz(float(rng.normal()), q)
        for q in rng.permutation(width - 1):
            builder.cx(int(q), int(q) + 1)
    return builder.build()


@pytest.mark.parametrize("seed", range(5))
def test_canonical_truncation_error_reports_bound_independent_dense_readouts(seed):
    circuit = random_local(seed)
    observables = [oq.PauliString(w) for w in ("XYZIZ", "ZZIII", "IIXXI", "IIIII")]
    backend = oq.TruncatedMPSBackend(max_bond_dimension=2, max_state_error=2)
    report = backend.evaluate(circuit, observables)
    expected = oq.expectations(circuit, observables)
    assert np.all(np.abs(report.values - expected) <= report.observable_error_bounds + 1e-12)
    assert report.max_bond_dimension <= 2 and not report.floating_point_error_included
    assert all(0 <= weight <= 1 for weight in report.discarded_weights)
    assert report.values[-1] == pytest.approx(1, abs=1e-12)


def test_no_truncation_matches_dense_including_reversed_two_wire_order():
    circuit = random_local(8).crx(0.7, 3, 2).iswap(1, 0)
    observables = [oq.PauliString("XYZIX"), oq.PauliString("ZZIII")]
    report = oq.TruncatedMPSBackend(max_bond_dimension=16, discarded_weight_tolerance=0).evaluate(
        circuit, observables
    )
    np.testing.assert_allclose(report.values, oq.expectations(circuit, observables), atol=2e-14)
    assert report.state_distance_bound == 0


def test_truncation_budget_and_nonlinear_gradient_rejection():
    circuit = oq.Circuit(2).h(0).cx(0, 1)
    observables = [oq.PauliString("XX")]
    with pytest.raises(oq.ResourceError, match="max_state_error"):
        oq.TruncatedMPSBackend(max_bond_dimension=1).evaluate(circuit, observables)
    with pytest.raises(oq.ResourceError, match="max_memory_mb"):
        oq.TruncatedMPSBackend().evaluate(
            circuit, observables, options=oq.ExecutionOptions(max_memory_mb=0.001)
        )
    with pytest.raises(oq.ValidationError, match="parameter-shift"):
        oq.TruncatedMPSBackend().prepare_gradient_circuit(circuit)
    with pytest.raises(oq.ValidationError, match="nearest-neighbor"):
        oq.TruncatedMPSBackend().evaluate(oq.Circuit(3).cx(0, 2), [oq.PauliString("XXX")])


def test_optimal_two_site_shadow_duals_are_unbiased_for_every_pauli_operator():
    from openqml.shadows import _dual, _frame, _pauli_basis

    reference = np.diag([0.7, 0.1, 0.1, 0.1]).astype(complex)
    dual, residual = _dual(reference)
    frame = _frame(2)
    np.testing.assert_allclose(frame.T @ dual, np.eye(16), atol=1e-14)
    assert residual < 1e-12
    # The reference is deliberately different from the measured entangled state.
    state = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
    rho = np.outer(state, state.conj())
    expected = np.einsum("pij,ji->p", _pauli_basis(2), rho).real
    probabilities = frame @ expected
    np.testing.assert_allclose(probabilities @ dual, expected, atol=1e-14)


def test_correlated_shadow_dual_reduces_exact_single_shot_variance_for_bell_case():
    from openqml.shadows import _dual, _frame, _pauli_basis

    state = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
    rho = np.outer(state, state.conj())
    frame = _frame(2)
    probabilities = frame @ np.einsum("pij,ji->p", _pauli_basis(2), rho).real
    canonical, _ = _dual(np.eye(4) / 4)
    optimized, _ = _dual(0.95 * rho + 0.05 * np.eye(4) / 4)
    index = 5  # XX in the I/X/Y/Z tensor basis
    old = probabilities @ np.square(canonical[:, index]) - 1
    new = probabilities @ np.square(optimized[:, index]) - 1
    assert 0 <= new < old / 10


def test_shadow_acquisition_seed_shot_accounting_and_posthoc_reuse(monkeypatch):
    import openqml.shadows as shadows

    circuit = oq.Circuit(2).h(0).cx(0, 1)
    a = oq.shadow_estimate(circuit, [oq.PauliString("XX")], shots=1200, pilot_shots=200, seed=7)
    b = oq.shadow_estimate(circuit, [oq.PauliString("XX")], shots=1200, pilot_shots=200, seed=7)
    assert a.shots == 1200 and a.pilot_shots == 200 and a.blocks == ((0, 1),)
    np.testing.assert_array_equal(a.values, b.values)
    assert abs(a.values[0] - 1) < 5 * a.standard_errors[0]
    monkeypatch.setattr(shadows, "_collect", lambda *a, **k: pytest.fail("Measurements repeated"))
    reread = a.reestimate([oq.PauliString("ZZ"), oq.PauliString("II", 2)])
    assert abs(reread.values[0] - 1) < 5 * reread.standard_errors[0]
    assert reread.values[1] == 2 and reread.standard_errors[1] == 0
    assert not a._production[0].flags.writeable


def test_shadow_canonical_and_adaptive_mixed_state_readouts():
    source = oq.Circuit(1).h(0).phase_damping(0.25, 0)
    a = oq.shadow_estimate(
        source, [oq.PauliString("X"), oq.PauliString("Z")], shots=2000, pilot_shots=0, seed=31
    )
    truth = oq.expectations(source, [oq.PauliString("X"), oq.PauliString("Z")])
    assert np.all(np.abs(a.values - truth) < 5 * a.standard_errors)
    assert a.blocks == ((0,),)


@pytest.mark.parametrize(
    "options",
    [
        {"block_size": 3},
        {"regularization": 0},
        {"pilot_shots": 1},
        {"shots": 100, "pilot_shots": 100},
        {"max_settings": 1},
        {"max_work": 10},
        {"max_memory_mb": 0.001},
        {"max_qubits": 1},
    ],
)
def test_shadow_admission_and_invalid_options(options):
    with pytest.raises((oq.ResourceError, oq.ValidationError)):
        oq.shadow_estimate(oq.Circuit(2), [oq.PauliString("XX")], seed=3, **options)


def test_verified_optimizer_checks_symbolic_and_feedback_optimizations():
    theta = oq.Parameter("theta")
    source = (
        oq.Circuit(2)
        .ry(theta, 0)
        .ry(-theta, 0)
        .measure(1, 0)
        .if_else(oq.ClassicalCondition((0,), 1), oq.Circuit(2).x(0).x(0))
    )
    result = oq.optimize_verified(source)
    assert result.proof.proven
    assert result.proof.check(source, result.circuit)
