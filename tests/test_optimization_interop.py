import json
import subprocess
import sys

import numpy as np
import pytest

import openqml as q


def test_gradient_matches_analytic_nonlinear_shared_parameter_loss():
    def function(p):
        return (np.cos(2 * p[0]) - 0.3) ** 2 + p[1] ** 3

    parameters = np.array([0.21, 0.7])
    expected = [
        -4 * np.sin(2 * parameters[0]) * (np.cos(2 * parameters[0]) - 0.3),
        3 * parameters[1] ** 2,
    ]
    np.testing.assert_allclose(q.gradient(function, parameters), expected, atol=1e-8)


def test_parameter_shift_matches_expectation_derivative():
    def function(p):
        return q.Circuit(2).rx(p[0], 0).ry(p[1], 1).cx(0, 1).run().expectation("ZZ")

    parameters = [0.3, 0.7]
    np.testing.assert_allclose(
        q.parameter_shift(function, parameters), q.gradient(function, parameters), atol=1e-8
    )


@pytest.mark.parametrize("method", ["adam", "gd"])
def test_optimizer_returns_consistent_best_parameters(method):
    def objective(p):
        return float(np.sum((p - np.array([0.3, -0.7])) ** 2))

    result = q.minimize(objective, [1, 0], steps=150, learning_rate=0.1, method=method)
    assert result.value < 1e-7
    assert result.value == pytest.approx(objective(result.parameters))
    assert result.value == pytest.approx(min(result.history))
    assert result.n_evaluations >= result.n_steps + 1
    assert q.minimize(objective, [0.3, -0.7]).converged


def test_nonfinite_objective_or_gradient_is_rejected():
    for action in (
        lambda: q.gradient(lambda p: np.nan, [0]),
        lambda: q.minimize(lambda p: np.inf, [0]),
        lambda: q.minimize(lambda p: p[0] ** 2, [1], gradient_fn=lambda p: [1, 2]),
        lambda: q.gradient(lambda p: p[0], [1], epsilon=0),
    ):
        with pytest.raises(ValueError):
            action()


def test_qasm_roundtrip_preserves_full_state_and_rotation_precision():
    circuit = q.Circuit(3).h(0).rx(0.13, 1).ry(-0.7, 2).rz(0.9, 0).cx(0, 1).cy(1, 2)
    circuit.cz(2, 0).cp(np.pi / 7, 0, 2).ccx(0, 1, 2).swap(2, 0).s(1).tdg(2)
    text = circuit.to_qasm()
    restored = q.from_qasm(text)
    np.testing.assert_allclose(
        restored.run().statevector(), circuit.run().statevector(), atol=1e-12
    )
    np.testing.assert_allclose(
        q.from_qasm(circuit.inverse().to_qasm()).run().statevector(),
        circuit.inverse().run().statevector(),
        atol=1e-12,
    )


def test_qasm_expressions_comments_and_measurements():
    text = """OPENQASM 2.0;
    include "qelib1.inc"; // external includes are not fetched
    qreg reg[2]; creg out[2];
    /* a block comment */ ry(-(pi/2)+0.1, reg[0]);
    """
    # Multiple comma-separated angles are deliberately outside this subset.
    with pytest.raises(ValueError):
        q.from_qasm(text)
    text = text.replace("ry(-(pi/2)+0.1, reg[0])", "ry(-(pi/2)+0.1) reg[0]")
    np.testing.assert_allclose(
        q.from_qasm(text).run().statevector(),
        q.Circuit(2).ry(-np.pi / 2 + 0.1, 0).run().statevector(),
    )
    circuit = q.bell().measure(0, "a").measure(1, "b")
    state = q.from_qasm(circuit.to_qasm()).run(seed=4)
    assert state.classical["c0"] == state.classical["c1"]


@pytest.mark.parametrize(
    "statement",
    [
        "foo q[0]",
        'include "remote.inc"',
        "reset q[0]",
        "rx(__import__('os').system('true')) q[0]",
        "rx(1/0) q[0]",
        "h q[3]",
        "rx(pi, 0.1) q[0]",
        "qreg extra[2]",
        "if(c==1) x q[0]",
    ],
)
def test_qasm_rejects_unsupported_and_unsafe_source(statement):
    with pytest.raises(ValueError):
        q.from_qasm(f"OPENQASM 2.0; qreg q[1]; {statement};")


@pytest.mark.parametrize(
    "circuit",
    [
        q.Circuit(1).depolarizing(0.1, 0),
        q.Circuit(1).measure(0, basis="X"),
        q.Circuit(1).global_phase(0.3),
        q.Circuit(1).unitary(np.eye(2), [0], label="h"),
        q.Circuit(1).measure(0, "a").x(0, condition=("a", 1)),
    ],
)
def test_qasm_export_never_silently_drops_an_operation(circuit):
    with pytest.raises(q.UnsupportedOperationError):
        circuit.to_qasm()


def test_optional_qiskit_adapter_matches_statevector_and_density():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import DensityMatrix, Statevector

    circuit = q.Circuit(3).h(0).rx(0.3, 2).cx(0, 2).qft([2, 1]).global_phase(0.13)
    np.testing.assert_allclose(
        Statevector.from_instruction(circuit.to_qiskit()).data,
        circuit.run().statevector(),
        atol=1e-12,
    )
    circuit.amplitude_damping(0.1, 1)
    np.testing.assert_allclose(
        DensityMatrix.from_instruction(circuit.to_qiskit()).data,
        circuit.run().reduced_density(),
        atol=1e-12,
    )
    feedback = q.bell().measure(0, "bit").x(1, condition=("bit", 1)).to_qiskit()
    assert feedback.num_clbits == 1


def test_cli_runs_as_installed_module():
    output = subprocess.check_output(
        [sys.executable, "-m", "openqml", "demo", "--seed", "3"], text=True
    )
    payload = json.loads(output)
    assert sum(payload["counts"].values()) == 1024
    assert set(payload["counts"]) == {"000", "111"}


def test_import_never_requires_optional_frameworks_or_network():
    code = """
import socket, sys
def blocked(*args, **kwargs):
    raise AssertionError("import must not access the network")
socket.socket = blocked
import openqml
assert "qiskit" not in sys.modules
assert "torch" not in sys.modules
assert "pennylane" not in sys.modules
assert openqml.Circuit(1).x(0).run().sample(1) == {"1": 1}
"""
    subprocess.run([sys.executable, "-c", code], check=True)
