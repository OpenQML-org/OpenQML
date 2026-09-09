import math

import numpy as np
import pytest

from openqml.backends import expectations, simulate, z_expectations
from openqml.circuits import Circuit, angle_embedding, hardware_efficient_ansatz, w, x
from openqml.circuits.circuit import Ref


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


def test_a_pauli_word_cannot_name_a_wire_twice():
    """``'X0 Y0'`` is i*Z0, not Y0 -- overwriting silently changes the operator."""
    from openqml.hamiltonians import parse_terms

    with pytest.raises(ValueError, match="appears twice"):
        parse_terms([(1.0, "X0 Y0")])


def test_the_two_spellings_of_a_pauli_word_agree():
    from openqml.hamiltonians import parse_terms

    assert parse_terms([(1.0, "I0 Z1")]) == parse_terms([(1.0, {0: "I", 1: "Z"})])
    assert parse_terms([(1.0, "I0 Z1")]) == [(1.0, {1: "Z"})]


def test_a_hamiltonian_wider_than_the_register_is_refused():
    """kron only walks range(n_qubits), so a higher wire would be dropped --
    silently replacing the operator with its restriction."""
    from openqml.hamiltonians import exact_ground_state

    terms = [(1.0, {0: "Z"}), (0.5, {3: "X"})]
    assert exact_ground_state(terms)[0] == pytest.approx(-1.5)
    with pytest.raises(ValueError, match="does not fit"):
        exact_ground_state(terms, 2)


def test_data_reuploading_encodes_every_feature():
    """The loop runs over features, not qubits: with more features than qubits
    the other way round never reaches the trailing columns."""
    from openqml.circuits import data_reuploading

    circuit = data_reuploading(n_qubits=2, n_features=5, layers=2)
    assert circuit.n_features == 5
    referenced = {p.index for gate in circuit.gates for p in gate.params
                  if isinstance(p, Ref) and p.kind == "x"}
    assert referenced == {0, 1, 2, 3, 4}


def test_stable_hash_does_not_depend_on_set_iteration_order():
    """A set iterates in hash order, randomised per process for strings, and
    sort_keys does not reorder a list -- so run identity used to drift."""
    import subprocess
    import sys

    code = ("from openqml.utils import stable_hash;"
            "print(stable_hash({'tags': {'a', 'b', 'c', 'd', 'e', 'f'}}))")
    digests = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "random"}).stdout.strip()
        for _ in range(4)
    }
    assert len(digests) == 1, f"stable_hash drifted across processes: {digests}"
