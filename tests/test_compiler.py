import numpy as np
import pytest

from openqml import (
    Circuit,
    ConversionError,
    Parameter,
    ValidationError,
    circuit_stats,
    decompose,
    simplify,
    statevector,
)
from openqml.ir import GATES, UNITARY_GATES


def assert_same_unitary(left, right, bindings=None):
    bindings = {} if bindings is None else bindings
    for index in range(2**left.num_qubits):
        preparation = Circuit(left.num_qubits)
        for q, bit in enumerate(format(index, f"0{left.num_qubits}b")):
            if bit == "1":
                preparation = preparation.x(q)
        for_a = {k: v for k, v in bindings.items() if k in left.parameters}
        for_b = {k: v for k, v in bindings.items() if k in right.parameters}
        np.testing.assert_allclose(
            statevector(preparation.compose(left), for_a),
            statevector(preparation.compose(right), for_b),
            atol=2e-12,
        )


@pytest.mark.parametrize("gate", sorted(UNITARY_GATES - {"unitary"}))
@pytest.mark.parametrize("entangler", ["cx", "cz"])
def test_every_gate_lowers_to_restricted_basis_with_full_phase(gate, entangler):
    arity, nparams = GATES[gate]
    qubits = (2, 0, 1)[:arity]
    angles = (Parameter("theta"), 0.31, -0.72)[:nparams]
    circuit = Circuit(3).gate(gate, *qubits, params=angles).with_global_phase(0.29)
    basis = {"p", "ry", entangler}
    result = decompose(circuit, basis=basis)
    assert all(op.name in basis for op in result.operations)
    assert_same_unitary(circuit, result, {"theta": 0.51})


def test_symbolic_global_phase_lowering_and_inverse():
    t, phi = Parameter("theta"), Parameter("phi")
    circuit = (
        Circuit(3)
        .h(0)
        .u(2 * t, phi, -t, 2)
        .crx(t, 0, 2)
        .cp(phi, 1, 0)
        .cswap(2, 0, 1)
        .iswap(2, 1)
        .with_global_phase(phi + t / 3)
    )
    lowered = decompose(circuit, basis={"p", "ry", "cx"})
    assert lowered.global_phase.parameters
    for bindings in ({"theta": 0.17, "phi": -0.61}, {"theta": -0.43, "phi": 0.72}):
        assert_same_unitary(circuit, lowered, bindings)
        combined = circuit.compose(circuit.inverse())
        assert_same_unitary(combined, Circuit(3), bindings)


def test_transformations_diagnose_invalid_basis_limits_and_measurement_inverse():
    with pytest.raises(ConversionError, match="DECOMPOSITION_UNAVAILABLE"):
        decompose(Circuit(1).h(0), basis={"cx"})
    with pytest.raises(ConversionError, match="DECOMPOSITION_LIMIT"):
        decompose(Circuit(3).ccx(0, 1, 2), basis={"p", "ry", "cx"}, max_operations=3)
    with pytest.raises(ValidationError):
        decompose(Circuit(1), basis={"unknown"})
    with pytest.raises(ValidationError, match="measurement"):
        Circuit(1).measure_all().inverse()


def test_simplification_is_phase_exact_and_respects_barriers():
    t = Parameter("theta")
    circuit = Circuit(2).x(1).x(1).ry(t, 0).ry(-t, 0).s(1).sdg(1)
    assert simplify(circuit).operations == ()
    assert_same_unitary(circuit, simplify(circuit), {"theta": 0.7})
    full_rotation = Circuit(1).rx(np.pi, 0).rx(np.pi, 0)
    simplified = simplify(full_rotation)
    assert len(simplified.operations) == 1
    np.testing.assert_allclose(statevector(simplified), [-1, 0], atol=1e-14)
    blocked = Circuit(1).x(0).barrier().x(0)
    assert simplify(blocked).operations == blocked.operations


def test_compose_wire_and_classical_maps_preserve_shared_symbols():
    theta = Parameter("theta")
    subcircuit = Circuit(2).ry(theta, 0).cx(0, 1).measure(1, 0).with_global_phase(theta)
    result = Circuit(4).ry(theta, 0).compose(subcircuit, qubits=(3, 1), clbits=(2,))
    assert result.operations[1].qubits == (3,)
    assert result.operations[2].qubits == (3, 1)
    assert result.operations[3].clbits == (2,)
    assert result.num_clbits == 3 and result.parameters == ("theta",)
    for mapping in ((0,), (0, 0), (0, 5), (-1, 1)):
        with pytest.raises(ValidationError):
            Circuit(4).compose(subcircuit, qubits=mapping)


def test_statistics_depth_and_idle_register_scaling():
    circuit = Circuit(4).h(0).x(2).cx(0, 1).barrier(0, 1, 2).x(2).measure(1, 0)
    stats = circuit_stats(circuit)
    assert (stats.gates, stats.multi_qubit_gates, stats.measurements, stats.depth) == (4, 1, 1, 3)
    assert circuit_stats(Circuit(1_000_000_000).x(999_999_999)).depth == 1


def test_ir_v1_backwards_read_and_v2_gate_guard():
    old = Circuit(2).h(0).cx(0, 1)
    payload = old.to_dict()
    payload["version"] = 1
    for op in payload["operations"]:
        for field in ("matrix", "kraus", "condition", "blocks"):
            op.pop(field, None)
    assert Circuit.from_dict(payload) == old
    payload = Circuit(1).u(0.1, 0.2, 0.3, 0).to_dict()
    assert payload["version"] == 4
    payload["version"] = 1
    for op in payload["operations"]:
        for field in ("matrix", "kraus", "condition", "blocks"):
            op.pop(field, None)
    with pytest.raises(ValidationError, match="schema does not support this operation"):
        Circuit.from_dict(payload)
