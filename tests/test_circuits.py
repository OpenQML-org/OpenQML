import math

import numpy as np
import pytest

from openqml.backends import expectations, simulate, z_expectations
from openqml.circuits import Circuit, angle_embedding, hardware_efficient_ansatz, w, x


def test_bell_state():
    state = simulate(Circuit(2).h(0).cnot(0, 1)).state
    assert np.allclose(state, [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])


def test_rotation_matches_analytic_expectation():
    for theta in (0.0, 0.3, 1.7, math.pi):
        value = simulate(Circuit(1).ry(theta, 0)).expval({0: "Z"})
        assert value == pytest.approx(math.cos(theta), abs=1e-9)


def test_gate_set_is_unitary():
    circuit = (Circuit(3).h(0).rx(0.4, 1).rz(0.2, 2).cnot(0, 1).cz(1, 2)
               .swap(0, 2).crx(0.5, 0, 1).rzz(0.3, 1, 2).rxx(0.1, 0, 2))
    assert np.linalg.norm(simulate(circuit).state) == pytest.approx(1.0)


def test_symbolic_parameters_are_resolved():
    circuit = Circuit(1).ry(x(0), 0).rz(w(0), 0)
    assert circuit.n_features == 1 and circuit.n_parameters == 1
    value = simulate(circuit, weights=[0.0], features=[math.pi / 3]).expval({0: "Z"})
    assert value == pytest.approx(math.cos(math.pi / 3))


def test_serialisation_round_trip():
    original = angle_embedding(3, 3).compose(hardware_efficient_ansatz(3, 2))
    restored = Circuit.from_json(original.to_json())
    weights = np.linspace(0.1, 1.0, original.n_parameters)
    features = [0.2, 0.4, 0.6]
    assert np.allclose(simulate(original, weights, features).state,
                       simulate(restored, weights, features).state)


def test_amplitude_embedding_normalises():
    circuit = Circuit(2).amplitude_embedding()
    state = simulate(circuit, features=[1.0, 1.0, 1.0, 1.0]).state
    assert np.allclose(np.abs(state) ** 2, 0.25)


def test_depth_and_drawing():
    circuit = Circuit(2).h(0).cnot(0, 1).ry(0.3, 1)
    assert circuit.depth == 3
    assert "q0:" in circuit.draw()


def test_shots_add_noise_but_stay_in_range():
    circuit = Circuit(1).ry(0.7, 0)
    noisy = simulate(circuit, shots=256, seed=0).expval({0: "Z"})
    assert -1.0 <= noisy <= 1.0
    assert noisy != pytest.approx(math.cos(0.7), abs=1e-12)


def test_z_expectations_shape():
    circuit = angle_embedding(3, 3)
    assert z_expectations(circuit, [0, 1, 2], features=[0.1, 0.2, 0.3]).shape == (3,)
