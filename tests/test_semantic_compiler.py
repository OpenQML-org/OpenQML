"""Algebraic certificates must remain distinct from binding-specific checks."""

import json
import random
from dataclasses import replace

import numpy as np
import pytest

import openqml as oq


def random_circuit(seed, width=3, gates=18):
    rng = random.Random(seed)
    builder = oq.CircuitBuilder(width)
    theta = oq.Parameter("theta")
    for _ in range(gates):
        name = rng.choice(("h", "x", "ry", "rz", "rx", "cx", "cz", "swap", "cp"))
        a, b = rng.sample(range(width), 2)
        if name in {"ry", "rz", "rx"}:
            getattr(builder, name)(theta * rng.choice((-0.5, 0.5, 1.0)), a)
        elif name == "cp":
            builder.cp(theta / 2, a, b)
        elif name in {"cx", "cz", "swap"}:
            getattr(builder, name)(a, b)
        else:
            getattr(builder, name)(a)
    return builder.build()


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("target", ("qiskit", "cirq", "pennylane", "openqasm2", "openqasm3"))
def test_random_circuit_roundtrips_against_independent_native_runtimes(seed, target):
    pytest.importorskip("qiskit" if target.startswith("openqasm") else target)
    if target == "openqasm3":
        pytest.importorskip("qiskit_qasm3_import")
    source = random_circuit(seed)
    bindings = {"theta": 0.319}
    plan = oq.plan_conversion(source, target=target)
    if target in {"pennylane", "openqasm2"}:
        plan = oq.plan_conversion(source, target=target, bindings=bindings)
        bindings = None
    check = plan.verify(bindings, mode="native", max_qubits=3)
    assert check.matched and check.checked_inputs == 8
    assert check.max_absolute_error < 1e-12


@pytest.mark.parametrize("target", ("ir", "json", "qiskit", "cirq", "openqasm3"))
def test_certified_symbolic_export_includes_all_bindings_and_phase(target):
    if target in {"qiskit", "cirq"}:
        pytest.importorskip(target)
    angle = oq.Parameter("theta")
    source = oq.Circuit(3).h(2).rz(angle, 0).cp(angle / 2, 0, 1).swap(1, 2)
    certified = oq.compile_verified(source, target=target)
    assert certified.proof.proven and certified.proof.semantics == "unitary"
    assert certified.proof.parameters == ("theta",)
    data = json.loads(certified.proof.to_json())
    assert data["parameter_domain"] == "all_real_bindings_where_expressions_are_defined"
    assert data["target_execution_checked"] is False
    assert data["numeric_constants"] == "exact_stored_binary_rationals"


def test_bad_symbolic_adapter_can_pass_one_binding_but_never_get_a_certificate():
    source = oq.Circuit(1).ry(oq.Parameter("theta"), 0)
    adapter = oq.FrameworkAdapter(
        "wrong-sign-proof-fixture",
        lambda value: value,
        lambda circuit: oq.Circuit(1).ry(-oq.Parameter("theta"), 0),
    )
    oq.register_adapter(adapter)
    try:
        # The exported circuit is deliberately wrong away from theta=0.
        plan = oq.plan_conversion(source, target=adapter.name)
        assert not oq.prove_equivalence(source, plan.export().value).proven
        with pytest.raises(oq.ConversionError, match="SEMANTIC_PROOF_INCONCLUSIVE"):
            plan.certify()
        a, b = source.bind({"theta": 0}), plan.export().value.bind({"theta": 0})
        np.testing.assert_allclose(oq.statevector(a), oq.statevector(b))
    finally:
        oq.unregister_adapter(adapter.name)


def test_large_phase_polynomial_proves_nonlocal_symbolic_rewrite_without_state(monkeypatch):
    import openqml.simulator as simulator

    monkeypatch.setattr(
        simulator, "_initial_state", lambda *a, **k: pytest.fail("Dense state allocated")
    )
    theta = oq.Parameter("theta")
    left, right = oq.CircuitBuilder(1024), oq.CircuitBuilder(1024)
    for q in range(128):
        left.rzz(theta * 0.5, q, 1023 - q)
        right.cx(q, 1023 - q).rz(theta * 0.5, 1023 - q).cx(q, 1023 - q)
    source, target = left.build(), right.build()
    proof = oq.prove_equivalence(source, target)
    assert proof.proven and proof.method == "phase_polynomial"
    assert proof.check(source, target)
    assert not proof.check(source, target.x(0))
    assert proof.work < 20_000


def test_phase_and_wire_mutations_are_not_proved_and_pi_is_not_rounded():
    theta = oq.Parameter("theta")
    source = oq.Circuit(2).rz(theta, 0).cx(0, 1)
    assert not oq.prove_equivalence(source, oq.Circuit(2).rz(theta, 1).cx(1, 0)).proven
    assert not oq.prove_equivalence(source, source.with_global_phase(0.1)).proven
    assert oq.prove_equivalence(source, source.with_global_phase(0.1), phase_sensitive=False).proven
    # A binary approximation of pi/2 is not promoted to the named exact S gate.
    assert not oq.prove_equivalence(oq.Circuit(1).s(0), oq.Circuit(1).p(np.pi / 2, 0)).proven


def test_clifford_projective_proof_does_not_hide_phase_in_strict_mode():
    left = oq.Circuit(2).h(0).cx(0, 1).h(0).h(1)
    right = oq.Circuit(2).h(1).cx(1, 0)
    proof = oq.prove_equivalence(left, right, phase_sensitive=False)
    assert proof.proven and proof.method == "clifford_conjugation"
    assert proof.to_dict()["global_phase"] == "ignored"
    assert not oq.prove_equivalence(left, right.with_global_phase(0.125)).proven


@pytest.mark.parametrize(
    "name",
    (
        "rx",
        "ry",
        "rxx",
        "ryy",
        "rzz",
        "u",
        "crx",
        "cry",
        "crz",
        "cp",
        "sx",
        "sxdg",
        "iswap",
        "cy",
        "cz",
        "swap",
    ),
)
def test_exact_word_rewrites_agree_with_complete_independent_operator_columns(name):
    from openqml.semantics import _word

    theta = oq.Parameter("theta")
    params = (
        (theta, theta / 2, -theta)
        if name == "u"
        else (theta,)
        if name in {"rx", "ry", "rxx", "ryy", "rzz", "crx", "cry", "crz", "cp"}
        else ()
    )
    arity = 1 if name in {"rx", "ry", "u", "sx", "sxdg"} else 2
    source = oq.Circuit(arity, (oq.Operation(name, tuple(range(arity)), params),))
    form, _ = _word(source, 10_000, 100_000)
    operations = []
    for gate, wires, angle, matrix in form[0]:
        if gate in {"p", "rz"}:
            value = sum(
                float(coefficient) * (0.41 if atom == ("parameter", "theta") else 1)
                for atom, coefficient in angle
            )
            operations.append(oq.Operation(gate, wires, (value,)))
        else:
            operations.append(oq.Operation(gate, wires, matrix=matrix))
    phase = sum(
        float(coefficient)
        * (0.41 if atom == ("parameter", "theta") else np.pi if atom == ("pi",) else 1)
        for atom, coefficient in form[1]
    )
    normalized = oq.Circuit(arity, tuple(operations)).with_global_phase(phase)
    bound = source.bind({"theta": 0.41}) if source.parameters else source
    for index in range(1 << arity):
        prefix = oq.Circuit(arity)
        for q in range(arity):
            if index & (1 << (arity - q - 1)):
                prefix = prefix.x(q)
        np.testing.assert_allclose(
            oq.statevector(prefix.compose(bound)),
            oq.statevector(prefix.compose(normalized)),
            atol=1e-12,
        )


def test_proof_bounds_precede_serialization_and_opaque_identities_remain_inconclusive(monkeypatch):
    circuit = oq.Circuit(9000)
    monkeypatch.setattr(
        oq.Circuit, "to_json", lambda *a, **k: pytest.fail("Unbounded serialization")
    )
    with pytest.raises(oq.ResourceError, match="width"):
        oq.prove_equivalence(circuit, circuit)


def test_effectful_identity_is_scoped_to_instrument_and_different_channel_is_unproved():
    source = oq.Circuit(1).h(0).amplitude_damping(0.2, 0).measure_all()
    proof = oq.prove_equivalence(source, replace(source))
    assert proof.proven and proof.semantics == "quantum_instrument"
    assert not oq.prove_equivalence(
        source, oq.Circuit(1).h(0).amplitude_damping(0.3, 0).measure_all()
    ).proven


def test_coverage_is_a_diagnostic_matrix_without_sdk_execution():
    matrix = oq.semantic_coverage()
    assert matrix["native_execution_checked"] is False
    rows = {row["feature"]: row["targets"] for row in matrix["rows"]}
    assert rows["clifford_unitary"]["qiskit"]["supported"]
    assert not rows["symbolic_rotations"]["openqasm2"]["supported"]
    assert not rows["explicit_noise_channel"]["openqasm3"]["supported"]
    assert "CHANNEL_UNSUPPORTED" in rows["explicit_noise_channel"]["openqasm3"]["diagnostics"]


def test_symbolic_coefficient_growth_is_bounded_before_certificate_serialization():
    angle = oq.Parameter("theta").sin()
    for _ in range(5):
        angle = angle * 1e308
    with pytest.raises(oq.ResourceError, match="coefficient complexity"):
        oq.prove_equivalence(oq.Circuit(1).rz(angle, 0), oq.Circuit(1))


@pytest.mark.parametrize("seed", range(8))
def test_random_phase_circuit_and_inverse_have_exact_identity_certificate(seed):
    rng = random.Random(seed)
    builder = oq.CircuitBuilder(8)
    theta = oq.Parameter("theta")
    for _ in range(64):
        a, b = rng.sample(range(8), 2)
        name = rng.choice(("x", "y", "s", "t", "cx", "cp", "rz", "swap"))
        if name in {"cp", "rz"}:
            getattr(builder, name)(theta * 0.5, a, b) if name == "cp" else builder.rz(
                theta * 0.5, a
            )
        elif name in {"cx", "swap"}:
            getattr(builder, name)(a, b)
        else:
            getattr(builder, name)(a)
    circuit = builder.build()
    proof = oq.prove_equivalence(circuit.compose(oq.inverse(circuit)), oq.Circuit(8))
    assert proof.proven and proof.method == "phase_polynomial"


@pytest.mark.parametrize(
    "name", ("h", "x", "y", "z", "s", "sdg", "sx", "sxdg", "cx", "cy", "cz", "swap", "iswap")
)
def test_exact_integer_clifford_rules_agree_with_independent_native_operators(name):
    from qiskit.quantum_info import Operator

    from openqml.semantics import _conjugate

    width = 2 if name in {"cx", "cy", "cz", "swap", "iswap"} else 1
    wires = tuple(range(width))
    circuit = oq.Circuit(width, (oq.Operation(name, wires),))
    unitary = np.asarray(Operator(oq.convert(circuit, target="qiskit").value).data)

    def pauli(x, z, phase):
        matrix = np.zeros((1 << width, 1 << width), dtype=complex)
        for column in range(1 << width):
            matrix[column ^ x, column] = 1j**phase * (-1) ** ((column & z).bit_count())
        return matrix

    for x in range(1 << width):
        for z in range(1 << width):
            phase = (x & z).bit_count()
            result = _conjugate(x, z, phase, name, wires)
            np.testing.assert_allclose(
                pauli(*result), unitary @ pauli(x, z, phase) @ unitary.conj().T, atol=1e-12
            )
