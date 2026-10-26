import numpy as np
import pytest

from openqml import Circuit, Parameter, PauliString, expectation, parameter_shift


def test_shared_scaled_parameter_shift_matches_analytic():
    t = Parameter("t")
    circuit = Circuit(1).ry(t, 0).ry(2 * t + 0.1, 0).with_global_phase(Parameter("phase"))
    bindings = {"t": 0.27, "phase": 0.32}
    gradients = parameter_shift(circuit, PauliString("Z"), bindings)
    assert gradients["t"] == pytest.approx(-3 * np.sin(3 * bindings["t"] + 0.1), abs=1e-12)
    assert gradients["phase"] == 0


@pytest.mark.parametrize("gate", ["rx", "ry", "rz", "p", "rxx", "ryy", "rzz"])
def test_shift_matches_finite_difference(gate):
    t = Parameter("theta")
    circuit = Circuit(2).h(0).ry(0.43, 1).cx(0, 1)
    circuit = circuit.gate(gate, *((0, 1) if len(gate) == 3 else (1,)), params=(1.7 * t - 0.3,))
    ob, theta, eps = PauliString("XY"), 0.32, 1e-6
    finite = (
        expectation(circuit, ob, {"theta": theta + eps})
        - expectation(circuit, ob, {"theta": theta - eps})
    ) / (2 * eps)
    assert parameter_shift(circuit, ob, {"theta": theta})["theta"] == pytest.approx(
        finite, abs=1e-8
    )
