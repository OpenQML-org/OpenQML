import numpy as np
import pytest

import openqml
from openqml.hamiltonians import exact_ground_state
from openqml.models import (
    ClassicalKernelClassifier,
    ExactDiagonalisation,
    LinearRegressor,
    MajorityClassifier,
    QuantumKernelClassifier,
    VariationalQuantumClassifier,
    VQE,
    quantum_kernel,
)


def test_quantum_kernel_is_a_valid_gram_matrix():
    X = np.random.default_rng(0).uniform(-1, 1, (8, 2))
    gram = quantum_kernel(X, feature_map="zz", n_qubits=2)
    assert np.allclose(np.diag(gram), 1.0)
    assert np.allclose(gram, gram.T)
    assert np.min(np.linalg.eigvalsh(gram)) > -1e-8


def test_quantum_kernel_classifier_beats_chance_on_moons():
    task = openqml.get_task(1)
    X, y = task.get_X_and_y()
    train, test = task.get_train_test_split_indices(0)
    model = QuantumKernelClassifier(feature_map="zz", n_qubits=2).fit(X[train], y[train])
    assert model.score(X[test], y[test]) > 0.6


def test_quantum_kernel_separates_tfim_phases():
    task = openqml.get_task(3)
    X, y = task.get_X_and_y()
    train, test = task.get_train_test_split_indices(0)
    model = QuantumKernelClassifier(feature_map="amplitude").fit(X[train], y[train])
    assert model.score(X[test], y[test]) >= 0.85


def test_iqp_kernel_solves_parity_where_rbf_anti_learns():
    """The headline claim of the parity task, pinned down so it cannot rot."""
    task = openqml.get_task(2)
    X, y = task.get_X_and_y()
    quantum, classical = [], []
    for _, _, train, test in task.download_split():
        quantum.append(QuantumKernelClassifier(feature_map="iqp", n_qubits=4)
                       .fit(X[train], y[train]).score(X[test], y[test]))
        classical.append(ClassicalKernelClassifier().fit(X[train], y[train])
                         .score(X[test], y[test]))
    assert np.mean(quantum) == 1.0
    assert np.mean(classical) < 0.5


def test_variational_classifier_reduces_its_loss():
    task = openqml.get_task(1)
    X, y = task.get_X_and_y()
    train, _ = task.get_train_test_split_indices(0)
    model = VariationalQuantumClassifier(n_qubits=2, layers=1, maxiter=8).fit(
        X[train][:40], y[train][:40])
    assert model.loss_curve_[-1] < model.loss_curve_[0]
    assert len(model.predict(X[train][:5])) == 5
    assert model.predict_proba(X[train][:5]).shape == (5, 2)


def test_vqe_approaches_exact_ground_state():
    record = openqml.get_task(4).get_hamiltonians()[3]
    energy = VQE(layers=2, maxiter=60, seed=1).estimate(record["terms"])
    assert energy >= record["reference_energy"] - 1e-9  # variational lower bound holds
    assert abs(energy - record["reference_energy"]) < 5e-3


def test_exact_diagonalisation_matches_helper():
    record = openqml.get_task(4).get_hamiltonians()[0]
    assert ExactDiagonalisation(2).estimate(record["terms"]) == pytest.approx(
        exact_ground_state(record["terms"], 2)[0])


def test_baselines_are_sane():
    y = np.array([0, 0, 0, 1])
    model = MajorityClassifier().fit(np.zeros((4, 2)), y)
    assert set(model.predict(np.zeros((3, 2)))) == {0}

    X = np.linspace(-1, 1, 40).reshape(-1, 1)
    target = 2 * X[:, 0] + 1
    assert LinearRegressor().fit(X, target).score(X, target) < 1e-8


def test_get_params_round_trip():
    model = QuantumKernelClassifier(feature_map="iqp", alpha=0.5)
    restored = QuantumKernelClassifier(**model.get_params())
    assert restored.get_params() == model.get_params()


def test_auto_rescaling_leaves_in_range_data_untouched():
    """Bit strings already live in [-1, 1]; rescaling them would break parity."""
    X, y = openqml.get_task(2).get_X_and_y()
    model = QuantumKernelClassifier(feature_map="iqp", n_qubits=4).fit(X, y)
    assert model._scaler_ is None
    assert model.score(X, y) == 1.0


def test_auto_rescaling_rescues_out_of_range_features():
    """Without it, an angle encoding wraps and the model lands below chance."""
    task = openqml.get_task(7)  # blobs-4d, features well outside [-1, 1]
    X, y = task.get_X_and_y()
    train, test = task.get_train_test_split_indices(0)
    scaled = QuantumKernelClassifier(feature_map="iqp", n_qubits=4).fit(X[train], y[train])
    raw = QuantumKernelClassifier(feature_map="iqp", n_qubits=4, rescale=False).fit(
        X[train], y[train])
    assert scaled._scaler_ is not None and raw._scaler_ is None
    assert scaled.score(X[test], y[test]) > raw.score(X[test], y[test])


def test_variational_model_accepts_statevector_inputs():
    import warnings

    task = openqml.get_task(3)
    X, y = task.get_X_and_y()
    train, test = task.get_train_test_split_indices(0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = VariationalQuantumClassifier(maxiter=15, seed=1).fit(X[train], y[train])
    # a complex input must be loaded into the amplitudes, never cast to float
    assert not [w for w in caught if "Complex" in type(w.message).__name__]
    assert model.n_qubits_ == 6  # 64 amplitudes loaded directly, not 64 rotations
    assert model.score(X[test], y[test]) > 0.5
