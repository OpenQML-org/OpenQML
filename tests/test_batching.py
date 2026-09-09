"""The batched execution path has to agree with the naive one, exactly."""

import numpy as np
import pytest

from openqml.backends import (batched_energies, batched_states, batched_z,
                              batched_z_jacobian, simulate)
from openqml.backends.statevector import StatevectorSimulator
from openqml.circuits import angle_embedding, hardware_efficient_ansatz
from openqml.models.kernel import clear_state_cache, encode_states
from openqml.models.variational import (
    VariationalQuantumClassifier,
    parameter_shift_gradient,
    shift_sets,
)

CIRCUIT = angle_embedding(3, 3).compose(hardware_efficient_ansatz(3, 2))
TERMS = [(0.7, {0: "X", 1: "Y"}), (-1.3, {2: "Z"}), (0.4, {0: "Z", 2: "Z"}), (0.2, {})]


def _inputs(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, CIRCUIT.n_parameters)), rng.normal(size=(n, 3))


def test_batched_states_match_the_loop():
    weights, features = _inputs()
    batched = batched_states(CIRCUIT, features, weights)
    loop = np.stack([StatevectorSimulator(3).run(CIRCUIT, w, f).state
                     for w, f in zip(weights, features)])
    assert np.allclose(batched, loop)


def test_batched_expectations_match_the_loop():
    weights, features = _inputs()
    batched = batched_z(CIRCUIT, [0, 2], weights, features)
    loop = np.array([[StatevectorSimulator(3).run(CIRCUIT, w, f).expval({q: "Z"}) for q in (0, 2)]
                     for w, f in zip(weights, features)])
    assert np.allclose(batched, loop)

    energies = batched_energies(CIRCUIT, TERMS, weights, features)
    reference = [StatevectorSimulator(3).run(CIRCUIT, w, f).expval_hamiltonian(TERMS)
                 for w, f in zip(weights, features)]
    assert np.allclose(energies, reference)


def test_chunking_does_not_change_the_answer():
    weights, features = _inputs(n=17)
    whole = batched_states(CIRCUIT, features, weights)
    split = batched_states(CIRCUIT, features, weights, chunk=3)
    assert np.allclose(whole, split)


def test_shared_weights_broadcast_over_a_feature_batch():
    weights, features = _inputs()
    shared = batched_z(CIRCUIT, [0], weights[0], features)
    tiled = batched_z(CIRCUIT, [0], np.repeat(weights[:1], len(features), axis=0), features)
    assert np.allclose(shared, tiled)


def test_amplitude_embedding_survives_a_weight_batch():
    """One prepared state, many parameter rows -- the shape a gradient asks for."""
    from openqml.circuits import Circuit, hardware_efficient_ansatz

    circuit = Circuit(2).amplitude_embedding().compose(hardware_efficient_ansatz(2, 1))
    rng = np.random.default_rng(5)
    vector = rng.normal(size=4)
    vector /= np.linalg.norm(vector)
    weights = rng.normal(size=(6, circuit.n_parameters))

    batched = batched_z(circuit, [0, 1], weights, vector)
    loop = np.array([[StatevectorSimulator(2).run(circuit, row, vector).expval({q: "Z"})
                      for q in (0, 1)] for row in weights])
    assert batched.shape == (6, 2)
    assert np.allclose(batched, loop)


def test_training_gradient_matches_finite_differences():
    """The batched training gradient is the true gradient of the loss.

    Ground truth here is a central difference, not the shift rule: the shift
    rule is exact for an expectation value, and a squared loss is not one.
    """
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 2))
    y = (X[:, 0] > 0).astype(int)
    model = VariationalQuantumClassifier(n_qubits=2, layers=1, maxiter=1, seed=0).fit(X, y)
    targets = np.where(y == model.classes_[0], -1.0, 1.0).reshape(-1, 1)
    weights = rng.normal(size=len(model.weights_))
    encoded = model._prepare_X(model._transform_features(X))

    loss, gradient = model._loss_and_gradient(weights, encoded, targets, unique=True)

    def evaluate(candidate):
        return float(np.mean((model._outputs_prepared(candidate, encoded) - targets) ** 2))

    numerical = np.zeros_like(weights)
    for index in range(len(weights)):
        plus, minus = weights.copy(), weights.copy()
        plus[index] += 1e-6
        minus[index] -= 1e-6
        numerical[index] = (evaluate(plus) - evaluate(minus)) / 2e-6

    assert loss == pytest.approx(evaluate(weights))
    assert np.allclose(gradient, numerical, atol=1e-6)


def test_shift_rule_is_exact_for_an_expectation_value():
    """Where the shift rule does apply, it agrees with finite differences."""
    rng = np.random.default_rng(3)
    weights = rng.normal(size=CIRCUIT.n_parameters)
    features = rng.normal(size=3)

    def energy(candidate):
        return float(batched_energies(CIRCUIT, TERMS, np.atleast_2d(candidate), features)[0])

    exact = parameter_shift_gradient(energy, weights)
    numerical = np.array([
        (energy(weights + 1e-6 * np.eye(len(weights))[i])
         - energy(weights - 1e-6 * np.eye(len(weights))[i])) / 2e-6
        for i in range(len(weights))
    ])
    assert np.allclose(exact, numerical, atol=1e-6)


def test_shift_sets_layout():
    sets, scale = shift_sets(np.array([0.0, 1.0]))
    assert sets.shape == (5, 2) and scale == 0.5
    assert np.allclose(sets[0], [0.0, 1.0])
    assert sets[1][0] > sets[2][0] and np.allclose(sets[1][1], 1.0)


def test_a_backend_without_the_fast_path_still_works():
    """The batched helpers and the adjoint sweep are both optional.

    A backend offering only run/state/expval falls back to a loop and to the
    parameter-shift rule, and has to reach the same answer either way.
    """
    from openqml import backends, models
    from openqml.backends.statevector import StatevectorSimulator as Reference

    class SlowSimulator:
        name = "slow"
        supports_batch = False

        def __init__(self, n_qubits, seed=None, shots=None):
            self._inner = Reference(n_qubits, seed=seed, shots=shots)
            self.n_qubits = n_qubits

        def run(self, circuit, weights=None, features=None):
            self._inner.run(circuit, weights, features)
            return self

        @property
        def state(self):
            return self._inner.state

        def expval(self, word):
            return self._inner.expval(word)

        def expval_hamiltonian(self, terms):
            return self._inner.expval_hamiltonian(terms)

    backends.register_backend("test.slow", lambda n, **kw: SlowSimulator(n, **kw))
    assert not backends.supports_adjoint(SlowSimulator(2))

    rng = np.random.default_rng(6)
    X = rng.normal(size=(24, 2))
    y = (X[:, 0] > 0).astype(int)
    kwargs = dict(n_qubits=2, layers=1, maxiter=4, seed=0)
    slow = VariationalQuantumClassifier(backend="test.slow", **kwargs).fit(X, y)
    fast = VariationalQuantumClassifier(**kwargs).fit(X, y)

    assert slow.gradient_method_ == "parameter_shift"
    assert np.allclose(slow.weights_, fast.weights_)
    assert np.array_equal(slow.predict(X), fast.predict(X))

    kernel = models.QuantumKernelClassifier(feature_map="zz", n_qubits=2, backend="test.slow")
    reference = models.QuantumKernelClassifier(feature_map="zz", n_qubits=2)
    assert np.array_equal(kernel.fit(X, y).predict(X), reference.fit(X, y).predict(X))


def test_state_cache_is_transparent():
    X = np.random.default_rng(2).normal(size=(12, 3))
    clear_state_cache()
    uncached = encode_states(X, "zz", 3, use_cache=False)
    cached = encode_states(X, "zz", 3)
    again = encode_states(X, "zz", 3)
    assert np.allclose(uncached, cached) and np.allclose(cached, again)
    assert not np.allclose(encode_states(X + 1.0, "zz", 3), cached)


def test_chunking_handles_weights_shared_as_a_single_row():
    """``(1, P)`` means "shared by the batch", the same as a 1-D array.

    Slicing it like a batch hands every chunk after the first an empty array.
    """
    circuit = angle_embedding(3, 3).compose(hardware_efficient_ansatz(3, 2))
    rng = np.random.default_rng(0)
    shared = rng.normal(size=(1, circuit.n_parameters))
    features = rng.normal(size=(9, 3))

    whole = batched_z(circuit, [0], shared, features)
    pieces = batched_z(circuit, [0], shared, features, chunk=4)
    assert pieces.shape == whole.shape == (9, 1)
    assert np.allclose(whole, pieces)

    value, jacobian = batched_z_jacobian(circuit, [0], shared, features)
    chunked_value, chunked_jacobian = batched_z_jacobian(circuit, [0], shared, features, chunk=4)
    assert np.allclose(value, chunked_value)
    assert np.allclose(jacobian, chunked_jacobian)


def test_a_useless_chunk_size_is_refused():
    circuit = angle_embedding(2, 2)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(4, 2))
    for bad in (0, -1):
        with pytest.raises(ValueError, match="chunk must be at least 1"):
            batched_z(circuit, [0], None, features, chunk=bad)


def test_the_single_run_path_refuses_a_batch():
    """``run``/``simulate`` used to reshape(-1) a batch and answer for row 0."""
    circuit = angle_embedding(2, 2)
    rng = np.random.default_rng(0)
    features = rng.normal(size=(4, 2))
    with pytest.raises(ValueError, match="single-circuit path"):
        simulate(circuit, None, features)
    assert simulate(circuit, None, features[0]).batch_size == 1
