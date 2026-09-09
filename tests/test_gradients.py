"""The adjoint sweep has to produce the parameter-shift gradient, exactly.

Both rules are exact for an expectation value on an exact simulator, so they
are not an approximation of one another -- they are two ways of computing the
same number, and they have to agree to machine precision. What differs is
cost: ``2P+1`` circuit runs against one backward sweep.
"""

import numpy as np
import pytest

import openqml
from openqml.backends import (batched_energy_jacobian, batched_z, batched_z_jacobian,
                              get_backend, prefers_adjoint)
from openqml.circuits import Circuit, hardware_efficient_ansatz, w, x, zz_feature_map
from openqml.circuits.templates import get_ansatz
from openqml.models import VQE, VariationalQuantumClassifier
from openqml.models.variational import parameter_shift_gradient, shift_sets

CIRCUIT = zz_feature_map(3, 3).compose(hardware_efficient_ansatz(3, 2))
TERMS = [(0.7, {0: "X", 1: "Y"}), (-1.3, {2: "Z"}), (0.4, {0: "Z", 2: "Z"}), (0.2, {})]


def _shift_jacobian(circuit, wires, weights, features):
    """d<Z>/dweights the slow way: two whole runs per parameter."""
    sets, scale = shift_sets(weights)
    values = batched_z(circuit, wires, sets, features)
    return scale * (values[1::2] - values[2::2]).T  # (W, P)


def test_adjoint_matches_the_shift_rule():
    rng = np.random.default_rng(0)
    weights = rng.normal(size=CIRCUIT.n_parameters)
    features = rng.normal(size=3)
    wires = [0, 2]

    outputs, jacobian = batched_z_jacobian(CIRCUIT, wires, weights, features)

    assert np.allclose(outputs, batched_z(CIRCUIT, wires, weights, features))
    assert np.allclose(jacobian[0], _shift_jacobian(CIRCUIT, wires, weights, features))


def test_adjoint_matches_the_shift_rule_over_a_batch():
    rng = np.random.default_rng(1)
    weights = rng.normal(size=(5, CIRCUIT.n_parameters))
    features = rng.normal(size=(5, 3))
    _, jacobian = batched_z_jacobian(CIRCUIT, [0], weights, features)
    for row in range(5):
        expected = _shift_jacobian(CIRCUIT, [0], weights[row], features[row])
        assert np.allclose(jacobian[row], expected)


def test_adjoint_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(2)
    weights = rng.normal(size=(17, CIRCUIT.n_parameters))
    features = rng.normal(size=3)
    whole = batched_z_jacobian(CIRCUIT, [0, 1], weights, features)
    split = batched_z_jacobian(CIRCUIT, [0, 1], weights, features, chunk=4)
    assert np.allclose(whole[0], split[0]) and np.allclose(whole[1], split[1])


def test_adjoint_handles_a_weight_shared_by_several_gates():
    """Where the two-term shift rule stops applying, the sweep still holds.

    A weight driving more than one gate breaks the two-term rule, which is why
    the shift path falls back to central differences there. The adjoint sweep
    just sums the contributions, so it stays exact.
    """
    circuit = Circuit(2, name="shared").h(0).h(1)
    circuit.ry(w(0), 0).rz(w(0), 1).cnot(0, 1).rx(w(0), 1).ry(w(1), 0)
    weights = np.array([0.3, -0.8])

    _, jacobian = batched_z_jacobian(circuit, [0, 1], weights)

    for column, wire in enumerate([0, 1]):
        def value(candidate, wire=wire):
            return float(batched_z(circuit, [wire], candidate)[0, 0])

        numerical = np.array([
            (value(weights + 1e-6 * unit) - value(weights - 1e-6 * unit)) / 2e-6
            for unit in np.eye(len(weights))
        ])
        assert np.allclose(jacobian[0, column], numerical, atol=1e-6)


def test_adjoint_respects_a_scaled_weight_reference():
    """``w(i, scale=s)`` means the chain rule owes a factor of ``s``."""
    circuit = Circuit(1, name="scaled").h(0).ry(w(0, scale=2.5, offset=0.3), 0)
    plain = Circuit(1, name="plain").h(0).ry(w(0), 0)
    weights = np.array([0.4])

    _, scaled = batched_z_jacobian(circuit, [0], weights)
    _, direct = batched_z_jacobian(plain, [0], 2.5 * weights + 0.3)
    assert np.allclose(scaled[0], 2.5 * direct[0])


def test_energy_jacobian_matches_the_shift_rule():
    rng = np.random.default_rng(3)
    circuit = get_ansatz("real_amplitudes", 3, 2)
    weights = rng.normal(size=circuit.n_parameters)

    energy, jacobian = batched_energy_jacobian(circuit, TERMS, np.atleast_2d(weights))

    def value(candidate):
        device = get_backend(n_qubits=3)
        return float(device.run(circuit, candidate).expval_hamiltonian(TERMS))

    assert energy[0] == pytest.approx(value(weights))
    assert np.allclose(jacobian[0], parameter_shift_gradient(value, weights), atol=1e-9)


def test_amplitude_embedding_does_not_break_the_sweep():
    """State preparation is not differentiable, and stops the backward walk."""
    rng = np.random.default_rng(4)
    circuit = Circuit(2).amplitude_embedding().compose(get_ansatz("real_amplitudes", 2, 1))
    vector = rng.normal(size=4)
    vector /= np.linalg.norm(vector)
    weights = rng.normal(size=circuit.n_parameters)

    _, jacobian = batched_z_jacobian(circuit, [0], weights, vector)
    assert np.allclose(jacobian[0], _shift_jacobian(circuit, [0], weights, vector))


def test_both_rules_train_the_same_model():
    task = openqml.get_task(1)
    X, y = task.get_X_and_y()
    fitted = [
        VariationalQuantumClassifier(n_qubits=2, layers=1, maxiter=12, seed=0,
                                     gradient=method).fit(X[:40], y[:40])
        for method in ("adjoint", "parameter_shift")
    ]
    assert fitted[0].gradient_method_ == "adjoint"
    assert fitted[1].gradient_method_ == "parameter_shift"
    assert np.allclose(fitted[0].weights_, fitted[1].weights_)
    assert np.allclose(fitted[0].loss_curve_, fitted[1].loss_curve_)


def test_both_rules_reach_the_same_ground_state():
    record = openqml.get_task(4).get_hamiltonians()[3]
    energies = [VQE(layers=2, maxiter=40, seed=1, gradient=method).estimate(record["terms"])
                for method in ("adjoint", "parameter_shift")]
    assert energies[0] == pytest.approx(energies[1], abs=1e-9)


def test_adjoint_is_refused_where_it_has_no_meaning():
    """It is an analytic method; shot noise is not something it can represent."""
    task = openqml.get_task(1)
    X, y = task.get_X_and_y()
    model = VariationalQuantumClassifier(n_qubits=2, layers=1, maxiter=2, shots=128,
                                         gradient="adjoint")
    with pytest.raises(ValueError, match="shots=None"):
        model.fit(X[:20], y[:20])

    with pytest.raises(ValueError, match="unknown gradient method"):
        VariationalQuantumClassifier(gradient="nonsense").fit(X[:20], y[:20])


def test_auto_falls_back_to_the_shift_rule_under_shot_noise():
    task = openqml.get_task(1)
    X, y = task.get_X_and_y()
    model = VariationalQuantumClassifier(n_qubits=2, layers=1, maxiter=2, shots=128,
                                         seed=0).fit(X[:20], y[:20])
    assert model.gradient_method_ == "parameter_shift"


def test_the_cost_model_prefers_the_sweep_only_when_it_is_worth_it():
    """Asymptotically cheaper is not the same as cheaper here (see prefers_adjoint)."""
    assert not prefers_adjoint(batch=1, n_qubits=4, n_parameters=12)
    assert prefers_adjoint(batch=1, n_qubits=12, n_parameters=36)
    assert prefers_adjoint(batch=160, n_qubits=2, n_parameters=15)


def test_the_shift_rule_declines_a_scaled_weight_reference():
    """The two-term rule assumes the gate angle *is* the weight.

    With ``w(0, scale=2)`` a pi/2 shift moves the angle by pi and the two terms
    cancel, so the rule reports a gradient of exactly zero. The circuit has to
    be routed to the fallback instead.
    """
    from openqml.models.variational import _weights_are_unique

    scaled = Circuit(1).h(0).ry(w(0, scale=2.0), 0)
    plain = Circuit(1).h(0).ry(w(0), 0)
    assert _weights_are_unique(plain)
    assert not _weights_are_unique(scaled)
    assert not _weights_are_unique(Circuit(1).h(0).ry(w(0, offset=0.5), 0))


def test_a_scaled_weight_still_gets_a_correct_gradient():
    scaled = Circuit(1).h(0).ry(w(0, scale=2.0), 0)
    weights = np.array([0.4])
    evaluate = lambda t: batched_z(scaled, [0], np.asarray(t).reshape(1, -1))[0, 0]

    _, jacobian = batched_z_jacobian(scaled, [0], weights.reshape(1, -1))
    eps = 1e-6
    finite = (evaluate(weights + eps) - evaluate(weights - eps)) / (2 * eps)

    assert jacobian[0, 0, 0] == pytest.approx(finite, abs=1e-6)
    assert abs(jacobian[0, 0, 0]) > 1.0  # not the zero the two-term rule returned
