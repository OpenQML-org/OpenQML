"""Independent operator-basis checks and adversarial hybrid equivalence cases."""

import cmath
import json
import random
from fractions import Fraction

import numpy as np
import pytest

import openqml as oq
from openqml._exact_instrument import _Algebra, _components, _kernel, _matrix
from openqml.simulator import _gate_matrix, _TensorOps


def scalar_value(poly, bindings):
    def atom_value(atom):
        kind, payload = atom
        if kind == "parameter":
            return bindings[payload]
        expression = oq.Expression.from_dict(json.loads(payload))
        value = expression.evaluate(bindings)
        return np.sqrt(1 - value) if kind == "sqrt_one_minus" else value

    value = 0j
    for (phase, powers), coefficients in poly.items():
        coefficient = sum(
            float(c) * cmath.exp(1j * np.pi * i / 4) for i, c in enumerate(coefficients)
        )
        angle = sum(
            float(c)
            * (np.pi if atom == ("pi",) else 1 if atom == ("constant",) else atom_value(atom))
            for atom, c in phase
        )
        term = coefficient * cmath.exp(1j * angle)
        for atom, power in powers:
            term *= atom_value(atom) ** power
        value += term
    return value


@pytest.mark.parametrize("name", sorted(oq.ir.UNITARY_GATES - {"unitary"}))
def test_every_exact_gate_rule_against_all_independent_numeric_columns(name):
    from openqml.ir import GATES

    width, count = GATES[name]
    params = tuple(oq.Parameter(f"theta{i}") * (-0.5 if i == 1 else 1) for i in range(count))
    bindings = {f"theta{i}": 0.319 + 0.1 * i for i in range(count)}
    op = oq.Operation(name, tuple(range(width)), params)
    columns = _matrix(op, _Algebra(100_000, 2_000_000))
    actual = np.zeros((1 << width, 1 << width), complex)
    for c, column in enumerate(columns):
        for r, value in column:
            actual[r, c] = scalar_value(value, bindings)
    expected = _gate_matrix(name, [p.evaluate(bindings) for p in op.params], _TensorOps())
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)


def numeric_kernel(circuit, bindings):
    from openqml.density import _evolve
    from openqml.planning import _record_plan

    bound = circuit.bind(bindings) if circuit.parameters else circuit
    size = 1 << circuit.num_qubits
    basis = np.eye(size * size, dtype=complex).reshape(size * size, size * size)
    records = _record_plan(bound, retain_classical=True, max_branches=256)
    states = _evolve(bound, {frozenset(): basis}, {}, _TensorOps(), records)
    expected = {}
    for bits, values in states.items():
        record = sum(1 << bit for bit in bits)
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                if abs(value) > 1e-14:
                    expected[i, j // size, j % size, record] = value
    return expected


@pytest.mark.parametrize("seed", range(8))
def test_random_hybrid_kernels_against_all_matrix_units_including_coherences(seed):
    rng = random.Random(seed)
    builder = oq.CircuitBuilder(2, num_clbits=2)
    builder.h(0).cx(0, 1)
    for _ in range(5):
        name = rng.choice(
            (
                "rx",
                "ry",
                "rz",
                "t",
                "sx",
                "bit_flip",
                "phase_damping",
                "amplitude_damping",
                "depolarizing",
            )
        )
        q = rng.randrange(2)
        if name in {"rx", "ry", "rz"}:
            getattr(builder, name)(oq.Parameter("theta"), q)
        elif name in {"bit_flip", "phase_damping", "amplitude_damping", "depolarizing"}:
            getattr(builder, name)(oq.Parameter("p"), q)
        else:
            getattr(builder, name)(q)
    builder.measure(0, 0).if_else(
        oq.ClassicalCondition((0,), 1), oq.Circuit(2).y(1), oq.Circuit(2).h(1)
    )
    builder.reset(0).measure(0, 0).measure(1, 1)
    circuit = builder.build()
    bindings = {name: 0.25 if name == "p" else 0.431 for name in circuit.parameters}
    alg = _Algebra(100_000, 5_000_000)
    (component,) = _components(circuit, circuit, alg)
    actual = {
        key: scalar_value(value, bindings)
        for key, value in _kernel(circuit, component, alg).items()
    }
    expected = numeric_kernel(circuit, bindings)
    for key in actual.keys() | expected.keys():
        assert actual.get(key, 0) == pytest.approx(expected.get(key, 0), abs=1e-12)


def test_reset_equals_measurement_feedback_even_when_histories_are_overwritten():
    condition = oq.ClassicalCondition((0,), 1)
    left = oq.Circuit(1).reset(0).measure(0, 0)
    right = oq.Circuit(1).measure(0, 0).if_else(condition, oq.Circuit(1).x(0)).measure(0, 0)
    proof = oq.prove_equivalence(left, right)
    assert proof.proven and proof.method == "factored_density_kernel"
    assert proof.check(left, right)
    assert not proof.check(left, right.x(0))


def test_unobserved_internal_histories_merge_but_final_classical_records_do_not():
    left = oq.Circuit(1).reset(0).measure(0, 0)
    right = oq.Circuit(1).measure(0, 0).reset(0)
    assert not oq.prove_equivalence(left, right).proven
    # |0> input alone would also fail to distinguish identity from dephasing.
    assert not oq.prove_equivalence(oq.Circuit(1), oq.Circuit(1).phase_flip(0.5, 0)).proven
    assert not oq.prove_equivalence(oq.Circuit(1), oq.Circuit(1).reset(0)).proven


def test_symbolic_feedback_decomposition_and_conditional_global_phase():
    theta = oq.Parameter("theta")
    prefix = oq.Circuit(2).h(0).measure(0, 0)
    condition = oq.ClassicalCondition((0,), 1)
    left = prefix.if_else(condition, oq.Circuit(2).rx(theta, 1))
    right = prefix.if_else(condition, oq.Circuit(2).h(1).rz(theta, 1).h(1).with_global_phase(theta))
    proof = oq.prove_equivalence(left, right)
    assert proof.proven and proof.parameters == ("theta",)
    assert proof.to_dict()["global_phase"] == "unobservable_for_quantum_instruments"
    wrong = prefix.if_else(condition, oq.Circuit(2).rx(-theta, 1))
    assert not oq.prove_equivalence(left, wrong).proven


def test_symbolic_stochastic_channel_composition_and_channel_extremes():
    p, q = oq.Parameter("p"), oq.Parameter("q")
    left = oq.Circuit(1).bit_flip(p, 0).bit_flip(q, 0)
    right = oq.Circuit(1).bit_flip(p + q - 2 * p * q, 0)
    assert oq.prove_equivalence(left, right).proven
    assert oq.prove_equivalence(
        oq.Circuit(1).amplitude_damping(1, 0), oq.Circuit(1).reset(0)
    ).proven
    assert oq.prove_equivalence(
        oq.Circuit(1).phase_damping(1, 0), oq.Circuit(1).phase_flip(0.5, 0)
    ).proven
    assert oq.prove_equivalence(oq.Circuit(1).amplitude_damping(0, 0), oq.Circuit(1)).proven


def test_exact_kraus_representation_and_wire_order():
    kraus = (((1, 0), (0, 0)), ((0, 1), (0, 0)))
    assert oq.prove_equivalence(oq.Circuit(2).channel(kraus, 1), oq.Circuit(2).reset(1)).proven
    assert not oq.prove_equivalence(oq.Circuit(2).channel(kraus, 1), oq.Circuit(2).reset(0)).proven


def test_branch_is_selected_once_when_selected_body_writes_its_guard():
    prefix = oq.Circuit(2).h(0).measure(0, 0)
    condition = oq.ClassicalCondition((0,), 1)
    body = oq.Circuit(2).reset(0).measure(0, 0).x(1)
    left = prefix.if_else(condition, body)
    # A gate-by-gate guard would skip x(1) after overwriting the guard.
    wrong = prefix.if_else(condition, oq.Circuit(2).reset(0).measure(0, 0)).if_else(
        condition, oq.Circuit(2).x(1)
    )
    assert not oq.prove_equivalence(left, wrong).proven
    nested = body.if_else(oq.ClassicalCondition((0,), 0), oq.Circuit(2).h(1).h(1))
    assert oq.prove_equivalence(left, prefix.if_else(condition, nested)).proven


def test_cross_program_classical_dependency_union_prevents_unsound_factorization():
    left = (
        oq.Circuit(2).h(0).measure(0, 0).if_else(oq.ClassicalCondition((0,), 1), oq.Circuit(2).x(1))
    )
    right = oq.Circuit(2).h(0).measure(0, 0).bit_flip(0.5, 1)
    assert not oq.prove_equivalence(left, right).proven


def test_thousand_idle_wires_and_independent_active_components_need_no_dense_state(monkeypatch):
    import openqml.simulator as simulator

    monkeypatch.setattr(
        simulator, "_initial_state", lambda *a, **k: pytest.fail("Dense state allocated")
    )
    left, right = oq.CircuitBuilder(1024), oq.CircuitBuilder(1024)
    for q in range(32):
        left.reset(q).x(q)
        right.x(q).reset(q).x(q)
    proof = oq.prove_equivalence(left.build(), right.build(), max_component_qubits=1)
    assert proof.proven and proof.work < 100_000


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_component_qubits": 1}, "max_component_qubits"),
        ({"max_terms": 2}, "max_terms"),
        ({"max_work": 100}, "max_work"),
    ],
)
def test_resource_admission_is_explicit(kwargs, message):
    left = oq.Circuit(2).h(0).cx(0, 1).reset(0)
    with pytest.raises(oq.ResourceError, match=message):
        oq.prove_equivalence(left, left.x(1), **kwargs)


def test_compilation_certifies_equivalent_but_structurally_different_hybrid_export():
    source = oq.Circuit(1).reset(0).measure(0, 0)
    target = (
        oq.Circuit(1)
        .measure(0, 0)
        .if_else(oq.ClassicalCondition((0,), 1), oq.Circuit(1).x(0))
        .measure(0, 0)
    )
    adapter = oq.FrameworkAdapter("reset-as-feedback", lambda value: value, lambda value: target)
    oq.register_adapter(adapter)
    try:
        result = oq.compile_verified(source, target=adapter.name, max_component_qubits=1)
        assert result.proof.proven and result.proof.method == "factored_density_kernel"
        assert oq.plan_conversion(source, target=adapter.name).certify().proof.proven
    finally:
        oq.unregister_adapter(adapter.name)


def test_binary_pi_approximation_is_not_promoted_to_exact_named_gate():
    left = oq.Circuit(1).h(0).t(0).reset(0).h(0)
    right = oq.Circuit(1).h(0).p(np.pi / 4, 0).reset(0).h(0)
    # Reset really erases this difference, which the kernel may prove.
    assert oq.prove_equivalence(left, right).proven
    left = oq.Circuit(1).h(0).t(0).phase_flip(0.25, 0)
    right = oq.Circuit(1).h(0).p(np.pi / 4, 0).phase_flip(0.25, 0)
    assert not oq.prove_equivalence(left, right).proven


def test_coefficient_ring_is_exact_and_opaque_real_atoms_remain_conservative():
    alg = _Algebra(1000, 10000)
    value = alg.scale(alg.constant(), Fraction(1, 3))
    assert scalar_value(alg.add(value, value, value), {}) == 1
    expression = oq.Parameter("p").sin()
    left = oq.Circuit(1).bit_flip(expression, 0).bit_flip(expression, 0)
    right = oq.Circuit(1).bit_flip(2 * expression - 2 * expression * expression, 0)
    assert oq.prove_equivalence(left, right).proven


def test_universal_gate_kernel_proves_nonword_identity_without_losing_global_phase():
    left = oq.Circuit(1).h(0).z(0).h(0)
    right = oq.Circuit(1).x(0)
    proof = oq.prove_equivalence(left, right)
    assert proof.proven and proof.method == "factored_operator_kernel"
    assert proof.to_dict()["global_phase"] == "compared"
    assert not oq.prove_equivalence(left, right.with_global_phase(0.125)).proven
    # XZ = -ZX: projective equality must never become strict equality.
    a, b = oq.Circuit(1).x(0).z(0), oq.Circuit(1).z(0).x(0)
    assert not oq.prove_equivalence(a, b).proven
    assert oq.prove_equivalence(a, b, phase_sensitive=False).proven


def test_symbolic_operator_kernel_handles_interleaved_hadamard_and_rotation_identities():
    theta = oq.Parameter("theta")
    left = oq.Circuit(1).h(0).rz(theta, 0).h(0)
    right = oq.Circuit(1).rx(theta, 0)
    assert oq.prove_equivalence(left, right).proven
    assert not oq.prove_equivalence(left, oq.Circuit(1).rx(-theta, 0)).proven


def test_gate_kernel_never_rounds_generated_parameter_sums():
    theta = oq.Parameter("theta")
    # Adding these stored coefficients in binary floating arithmetic would
    # erase theta. The exact U rule must multiply the two exponentials.
    left = oq.Circuit(1).u(0.5, 1e30 * theta, theta, 0).phase_flip(0.25, 0)
    right = oq.Circuit(1).u(0.5, 1e30 * theta, 0, 0).phase_flip(0.25, 0)
    assert not oq.prove_equivalence(left, right).proven


@pytest.mark.parametrize("target", ["qiskit", "cirq", "openqasm3"])
def test_symbolic_hybrid_certification_across_real_framework_adapters(target):
    if target in {"qiskit", "cirq"}:
        pytest.importorskip(target)
    theta = oq.Parameter("theta")
    source = (
        oq.Circuit(2)
        .h(0)
        .measure(0, 0)
        .if_else(oq.ClassicalCondition((0,), 1), oq.Circuit(2).rx(theta, 1), oq.Circuit(2).z(1))
        .reset(0)
    )
    compiled = oq.compile_verified(source, target=target)
    assert compiled.proof.proven and compiled.proof.semantics == "quantum_instrument"
    assert compiled.proof.parameters == ("theta",)
