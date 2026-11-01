"""Independent dense references, sparse-register scaling and error contracts."""

import itertools
import math

import numpy as np
import pytest

import openqml as oq


def word(width, sites, coefficient=1):
    return oq.PauliString("".join(sites.get(q, "I") for q in range(width)), coefficient)


def test_sparse_register_proofs_qasm_and_resource_reports_have_no_total_width_cap():
    width = 10**12
    q = width - 1
    left = oq.Circuit(width).h(q).h(q)
    right = oq.Circuit(width)
    proof = oq.prove_equivalence(left, right)
    assert proof.proven and proof.check(left, right)
    exported = oq.compile_verified(left, target="openqasm3")
    restored = oq.convert(exported.value, source="openqasm3", target="ir").value
    assert restored.num_qubits == width and restored.operations == left.operations
    estimate = oq.estimate_resources(oq.Circuit(10**1000))
    assert estimate.statevector_bytes is None and not estimate.within_statevector_policy
    assert estimate.statevector_log2_bytes == 10**1000 + 4
    assert oq.prove_equivalence(left, right, max_component_qubits=100).proven
    assert oq.Circuit(width).compose(left) == left
    assert isinstance(
        oq.estimate_resources(oq.Circuit(10**1000), batch_size=3).statevector_log2_bytes, str
    )
    with pytest.raises(oq.ResourceError, match="max_qubits"):
        oq.prove_equivalence(left, right, max_qubits=100)
    with pytest.raises(oq.ConversionError, match="REGISTER_LIMIT"):
        oq.convert(
            exported.value,
            source="openqasm3",
            target="ir",
            source_options={"max_register_bits": 100},
        )
    with pytest.raises(oq.ConversionError, match="OPERATION_LIMIT"):
        oq.convert(f"OPENQASM 3.0; qubit[{width}] q; h q;", source="openqasm3", target="ir")


def test_density_default_and_backend_routes_do_not_restore_a_twenty_wire_cap(monkeypatch):
    import openqml.density as density

    class ReachedAllocation(Exception):
        pass

    def allocate(*args, **kwargs):
        raise ReachedAllocation

    monkeypatch.setattr(density.np, "zeros", allocate)
    # Enormous explicit budget only probes admission; allocation is intercepted.
    with pytest.raises(ReachedAllocation):
        oq.density_matrix(oq.Circuit(21), max_memory_mb=2.0**28)

    seen = []
    monkeypatch.setattr(density, "density_matrix", lambda c, **kw: seen.append(kw["max_qubits"]))
    oq.get_backend("numpy").run_density(oq.Circuit(21), options=oq.ExecutionOptions())
    assert seen == [None]


def noisy_circuit(width, seed):
    rng = np.random.default_rng(seed)
    builder = oq.CircuitBuilder(width)
    for layer in range(3):
        for q in range(width):
            builder.ry(float(rng.normal()), q).rz(float(rng.normal()), q)
            getattr(builder, ("amplitude_damping", "phase_damping", "depolarizing")[layer])(0.13, q)
        for q in range(layer % 2, width - 1, 2):
            builder.cx(q, q + 1)
    return builder.build()


@pytest.mark.parametrize("seed", range(6))
def test_noise_aware_sparse_propagation_matches_full_density(seed):
    c = noisy_circuit(4, seed)
    observables = tuple(oq.PauliString("".join(p)) for p in itertools.product("IXYZ", repeat=4))
    exact = oq.expectations(c, observables)
    result = oq.SparsePauliPlan(c, observables, max_work=10_000_000).evaluate()
    np.testing.assert_allclose(result.values, exact, atol=2e-12)
    assert not result.error_bounds.any() and not any(result.discarded_terms)


@pytest.mark.parametrize(
    "gate", ("bit_flip", "phase_flip", "amplitude_damping", "phase_damping", "depolarizing")
)
@pytest.mark.parametrize("probability", (0.0, 0.37, 1.0))
def test_named_channels_match_independent_kraus_density(gate, probability):
    c = getattr(oq.Circuit(2).h(0).cx(0, 1).ry(0.21, 0), gate)(probability, 1)
    observables = tuple(oq.PauliString("".join(p)) for p in itertools.product("IXYZ", repeat=2))
    result = oq.SparsePauliPlan(c, observables).evaluate()
    np.testing.assert_allclose(result.values, oq.expectations(c, observables), atol=1e-12)


def test_custom_unitary_and_kraus_maps_keep_nonlocal_operand_order():
    rng = np.random.default_rng(53)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    unitary, _ = np.linalg.qr(a)
    operators = (np.sqrt(0.7) * unitary, np.sqrt(0.3) * np.eye(4))
    c = oq.Circuit(3).h(0).ry(0.31, 2).unitary(unitary, 2, 0).channel(operators, 0, 2)
    observables = tuple(oq.PauliString("".join(p)) for p in itertools.product("IXYZ", repeat=3))
    result = oq.SparsePauliPlan(c, observables, max_work=10_000_000).evaluate()
    np.testing.assert_allclose(result.values, oq.expectations(c, observables), atol=2e-12)


@pytest.mark.parametrize(
    "name,params,arity",
    (
        ("u", (0.3, -0.8, 0.7), 1),
        ("crx", (0.3,), 2),
        ("cry", (-0.2,), 2),
        ("crz", (0.8,), 2),
        ("cp", (0.8,), 2),
        ("ch", (), 2),
        ("ccx", (), 3),
        ("cswap", (), 3),
        ("rxx", (0.4,), 2),
        ("ryy", (0.4,), 2),
        ("rzz", (0.4,), 2),
    ),
)
def test_generic_and_rotation_gate_conjugation(name, params, arity):
    c = oq.Circuit(arity)
    for q in range(arity):
        c = c.ry(0.27 * (q + 1), q).rz(0.13, q)
    c = c.gate(name, *reversed(range(arity)), params=params)
    observables = tuple(oq.PauliString("".join(p)) for p in itertools.product("IXYZ", repeat=arity))
    result = oq.SparsePauliPlan(c, observables).evaluate()
    np.testing.assert_allclose(result.values, oq.expectations(c, observables), atol=2e-12)


@pytest.mark.parametrize("seed", range(8))
def test_discarded_l1_bounds_cover_noisy_hamiltonian_errors(seed):
    c = noisy_circuit(5, seed)
    observable = oq.Hamiltonian((oq.PauliString("XYZYX", 0.7), oq.PauliString("ZZZZZ", -0.3)))
    plan = oq.SparsePauliPlan(
        c, (observable,), max_terms=1024, coefficient_cutoff=0.0002, max_error=0.2
    )
    result = plan.evaluate()
    reference = oq.expectation(c, observable)
    assert result.discarded_terms[0] > 0
    assert abs(result.values[0] - reference) <= result.error_bounds[0] + 2e-12
    assert result.error_bounds[0] <= 0.2
    assert not result.values.flags.writeable


def test_sparse_words_address_far_wires_without_declared_width_storage():
    n = 10**12
    x, gamma = oq.Parameter("x"), oq.Parameter("gamma")
    c = oq.Circuit(n).ry(x, n - 1).amplitude_damping(gamma, n - 1)
    z = oq.SparsePauliString(n, {n - 1: "Z"})
    idle = oq.SparsePauliString(n, {3: "X"})
    h = oq.SparseHamiltonian((z, oq.SparsePauliString(n, {}, -0.25)))
    result = oq.SparsePauliPlan(c, (z, idle, h)).evaluate({"x": 0.4, "gamma": 0.15})
    expected = 0.85 * math.cos(0.4) + 0.15
    np.testing.assert_allclose(result.values, (expected, 0, expected - 0.25), atol=1e-13)
    assert result.active_qubits == 1 and result.num_qubits == n


def test_sparse_resource_failures_are_explicit_and_invalid_channels_are_not_skipped():
    c = oq.Circuit(2).ry(0.4, 0).cx(0, 1).ry(0.3, 1)
    with pytest.raises(oq.ResourceError, match="max_error"):
        oq.SparsePauliPlan(c, (oq.PauliString("ZZ"),), max_terms=1).evaluate()
    with pytest.raises(oq.ResourceError, match="max_work"):
        oq.SparsePauliPlan(c, (oq.PauliString("ZZ"),), max_work=1)
    with pytest.raises(oq.ResourceError, match="memory"):
        oq.SparsePauliPlan(c, (oq.PauliString("ZZ"),), max_memory_mb=1e-6)
    p = oq.Parameter("p")
    plan = oq.SparsePauliPlan(oq.Circuit(1).amplitude_damping(p, 0), (oq.PauliString("I"),))
    with pytest.raises(oq.ValidationError, match="probability"):
        plan.evaluate({"p": 1.1})


def test_mandatory_pruning_has_a_tight_worst_case_certificate():
    circuit = oq.Circuit(1).h(0).ry(0.1, 0)
    result = oq.SparsePauliPlan(
        circuit, (oq.PauliString("Z"),), max_terms=1, max_error=0.1
    ).evaluate()
    assert result.discarded_terms == (1,)
    error = abs(float(result.values[0]) + math.sin(0.1))
    assert error <= result.error_bounds[0]
    assert result.error_bounds[0] == pytest.approx(math.sin(0.1))


def test_nonselective_measurement_and_reset_match_density_evolution():
    circuit = oq.Circuit(3).h(0).cx(0, 2).measure(0, 1000).ry(0.4, 2).reset(1).rx(0.2, 1)
    observables = tuple(oq.PauliString("".join(p)) for p in itertools.product("IXYZ", repeat=3))
    result = oq.SparsePauliPlan(circuit, observables).evaluate()
    np.testing.assert_allclose(result.values, oq.expectations(circuit, observables), atol=1e-13)


def feedback_factor(width, q, target, bit):
    return (
        oq.Circuit(width)
        .ry(0.7, q)
        .measure(q, bit)
        .if_else(
            oq.ClassicalCondition((bit,), 1),
            oq.Circuit(width).x(target),
            oq.Circuit(width).ry(0.2, target),
        )
        .amplitude_damping(0.11, target)
    )


def test_classical_dependencies_join_quantum_factors_and_match_full_density():
    width = 6
    c = feedback_factor(width, 0, 5, 1000).compose(feedback_factor(width, 1, 4, 2000))
    obs = tuple(oq.PauliString(p) for p in ("ZIIIZZ", "XXIIYY", "IIIIII", "IZIIZI"))
    plan = oq.plan_components(c, obs)
    assert [len(w) for w, _, _ in plan.components] == [2, 2]
    assert plan.to_dict()["max_component_qubits"] == 2
    np.testing.assert_allclose(plan.evaluate(), oq.expectations(c, obs), atol=1e-13)
    # A shared storage bit makes the two apparent quantum components dependent.
    dependent = feedback_factor(width, 0, 5, 3).compose(feedback_factor(width, 1, 4, 3))
    joined = oq.plan_components(dependent, obs)
    assert [len(w) for w, _, _ in joined.components] == [4]
    np.testing.assert_allclose(joined.evaluate(), oq.expectations(dependent, obs), atol=1e-13)


def test_wide_feedback_auto_execution_allocates_only_local_density_states(monkeypatch):
    import openqml.density as density

    width = 256
    builder = oq.CircuitBuilder(width, num_clbits=width)
    for q in range(0, width, 2):
        builder.extend(feedback_factor(width, q, q + 1, q).operations)
    c = builder.build()
    original = density._simulate
    observed = []

    def guarded(circuit, *args, **kwargs):
        assert circuit.num_qubits <= 2
        observed.append(circuit.num_qubits)
        return original(circuit, *args, **kwargs)

    monkeypatch.setattr(density, "_simulate", guarded)
    obs = (word(width, {0: "Z", 1: "Z", 254: "Z", 255: "Z"}),)
    plan = oq.plan_execution(c, obs, max_memory_mb=32)
    assert plan.selected_method == "components"
    small = feedback_factor(2, 0, 1, 0)
    expected = oq.expectation(small, oq.PauliString("ZZ")) ** 2
    np.testing.assert_allclose(plan.evaluate(), (expected,), atol=1e-13)
    assert observed and set(observed) == {2}


def test_component_native_torch_second_derivatives_and_batched_parameters():
    torch = pytest.importorskip("torch")
    a = oq.Parameter("a")
    c = oq.Circuit(80).ry(a, 0).amplitude_damping(0.2, 0).ry(2 * a, 79)
    plan = oq.plan_components(c, (word(80, {0: "Z", 79: "Z"}),))
    x = torch.tensor([0.2, 0.4], dtype=torch.float64, requires_grad=True)
    actual = plan.evaluate({"a": x}, backend="torch")[..., 0]
    expected = (0.8 * torch.cos(x) + 0.2) * torch.cos(2 * x)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.sum(), x, create_graph=True)[0]
    ge = torch.autograd.grad(expected.sum(), x, create_graph=True)[0]
    torch.testing.assert_close(ga, ge)
    torch.testing.assert_close(
        torch.autograd.grad(ga.sum(), x)[0], torch.autograd.grad(ge.sum(), x)[0]
    )


def test_factor_metric_sums_shared_nonlinear_coordinates_and_cancels_phase():
    x, y, phase = oq.Parameter("x"), oq.Parameter("y"), oq.Parameter("phase")
    small = oq.Circuit(3).ry(x * y, 0).rz(x, 0).ry(x, 2).with_global_phase(phase)
    large = oq.Circuit(10**12).ry(x * y, 0).rz(x, 0).ry(x, 10**12 - 1).with_global_phase(phase)
    bindings = {"x": 0.4, "y": -0.3, "phase": 0.8}
    plan = oq.quantum_metric(large)
    assert plan.to_dict()["differentiated_components"] == 2
    # Independent full-state central differences, retaining all phase terms.
    states, h = [], 1e-6
    state = oq.statevector(small, bindings)
    for name in plan.parameters:
        plus, minus = dict(bindings), dict(bindings)
        plus[name] += h
        minus[name] -= h
        states.append((oq.statevector(small, plus) - oq.statevector(small, minus)) / (2 * h))
    jacobian = np.asarray(states).T
    projected = jacobian - state[:, None] * (state.conj() @ jacobian)[None, :]
    reference = np.real(jacobian.conj().T @ projected)
    np.testing.assert_allclose(plan.matrix(bindings), reference, atol=2e-10)
    gradient = np.arange(len(plan.parameters), dtype=float) + 1
    result = plan.natural_gradient(bindings, gradient)
    np.testing.assert_allclose(
        result.direction, np.linalg.solve(reference + 0.001 * np.eye(3), gradient), rtol=2e-8
    )


@pytest.mark.parametrize("sites", ({0: "A"}, ((0, "X"), (0, "Z")), {3: "X"}, {True: "Y"}))
def test_sparse_observable_validation(sites):
    with pytest.raises(oq.ValidationError):
        oq.SparsePauliString(3, sites)
