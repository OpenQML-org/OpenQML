import json

import numpy as np
import pytest

import openqml as q


def test_batched_feature_states_match_independent_circuits():
    inputs = np.array([[0.1, 0.3], [-0.7, 1.2], [0.0, -0.2]])
    kernel = q.QuantumKernel(layers=2)
    actual = kernel.statevectors(inputs)
    for sample, values in enumerate(inputs):
        circuit = q.Circuit(2)
        for layer in range(2):
            for wire, value in enumerate(values):
                circuit.ry(value, wire)
            for wire in range(2):
                circuit.rz((layer + 1) * (wire + 1) * 0.37, wire)
            circuit.cx(0, 1)
        np.testing.assert_allclose(actual[sample], circuit.run().statevector(), atol=1e-12)


def test_kernel_is_psd_normalized_symmetric_and_cross_consistent():
    features = np.random.default_rng(3).normal(size=(20, 3))
    kernel = q.QuantumKernel(layers=3)
    gram = kernel(features)
    np.testing.assert_allclose(gram, gram.T, atol=1e-12)
    np.testing.assert_allclose(gram.diagonal(), 1, atol=1e-12)
    assert np.linalg.eigvalsh(gram).min() > -1e-10
    np.testing.assert_allclose(kernel(features[:4], features[4:]), gram[:4, 4:])


def test_batched_variational_expectations_and_parameter_derivatives():
    model = q.VariationalQuantumClassifier(n_qubits=3, layers=2, epochs=2)
    features = np.array([[0.21, -0.3], [-0.5, 1.2], [0.0, 0.0]])
    model._prepare(features)
    parameters = np.random.default_rng(12).normal(size=model.n_angles_)
    values = model._expectations(features, parameters)
    for sample, angles in enumerate(features):
        circuit = q.Circuit(3)
        for wire in range(3):
            circuit.h(wire)
        for layer in range(2):
            for wire in range(3):
                circuit.rz(parameters[2 * (layer * 3 + wire)], wire)
            for wire, angle in enumerate(angles):
                circuit.rz(angle, wire)
            for wire in range(3):
                i = 2 * (layer * 3 + wire)
                circuit.ry(parameters[i + 1], wire)
            circuit.cx(0, 1).cx(1, 2).cx(2, 0)
        assert values[sample] == pytest.approx(circuit.run().expectation("IIZ"), abs=1e-12)
    jacobian = model._jacobian(features, parameters)
    for sample in range(len(features)):
        expected = q.gradient(lambda p: model._expectations(features, p)[sample], parameters)
        np.testing.assert_allclose(jacobian[sample], expected, atol=1e-8)


@pytest.mark.parametrize(
    "model_class", [q.VariationalQuantumClassifier, q.VariationalQuantumRegressor]
)
def test_loss_chain_rule_matches_finite_differences(model_class, monkeypatch):
    features = np.array([[0.3, 0.7], [-0.7, 1.3], [0.9, -0.1], [-0.2, -0.7]])
    labels = (
        np.array([0, 1, 1, 0])
        if model_class is q.VariationalQuantumClassifier
        else np.array([0.3, -0.5, 0.8, 0.4])
    )

    def check(self, objective, derivative, initial):
        for parameters in (initial, initial + 0.33):
            np.testing.assert_allclose(
                derivative(parameters), q.gradient(objective, parameters), atol=1e-6, rtol=1e-5
            )
        self.parameters_ = initial
        self.is_fitted_ = True

    monkeypatch.setattr(model_class, "_fit_objective", check)
    model_class(layers=2).fit(features, labels)


def test_kernel_classifier_learns_xor_and_preserves_label_types():
    data = q.load_dataset("xor", n_samples=80, noise=0.03, seed=7)
    a, b, c, d = data.split(seed=9)
    model = q.QuantumKernelClassifier().fit(a, np.where(c == 1, "odd", "even"))
    assert model.score(b, np.where(d == 1, "odd", "even")) >= 0.9
    probabilities = model.predict_proba(b)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert (probabilities >= 0).all()
    expected_scale = np.max(np.abs(a), axis=0)
    np.testing.assert_allclose(model.feature_scale_, expected_scale)
    model.predict(np.array([[100, 100]]))
    np.testing.assert_array_equal(model.feature_scale_, expected_scale)


@pytest.mark.parametrize("kind", ["classification", "regression"])
def test_variational_fit_is_reproducible_and_reduces_loss(kind):
    name = "moons" if kind == "classification" else "sine"
    data = q.load_dataset(name, n_samples=32, noise=0, frequency=1)
    cls = (
        q.VariationalQuantumClassifier
        if kind == "classification"
        else q.VariationalQuantumRegressor
    )
    first = cls(epochs=20, seed=2).fit(data.X, data.y)
    second = cls(epochs=20, seed=2).fit(data.X, data.y)
    np.testing.assert_allclose(first.parameters_, second.parameters_)
    np.testing.assert_allclose(first.predict(data.X), second.predict(data.X))
    assert min(first.history_) < first.history_[0] - 0.01
    assert first.n_circuit_evaluations_ > 0


def test_multiclass_kernel_and_binary_rejection():
    features = np.array([[-0.8], [-0.7], [0], [0.1], [0.8], [0.9]])
    labels = np.array([0, 0, 1, 1, 2, 2])
    model = q.QuantumKernelClassifier().fit(features, labels)
    assert model.score(features, labels) == 1
    with pytest.raises(ValueError, match="two classes"):
        q.VariationalQuantumClassifier().fit(features, labels)


def test_regressor_learns_a_phase_shifted_held_out_target():
    # Protect the trainable encoding phase: a cosine-only model cannot fit sin(x).
    data = q.load_dataset("sine", n_samples=64, frequency=1, noise=0, seed=7)
    a, b, c, d = data.split(seed=7)
    model = q.VariationalQuantumRegressor(epochs=100, seed=7).fit(a, c)
    assert model.score(b, d) > 0.95


@pytest.mark.parametrize(
    "model",
    [
        q.QuantumKernelClassifier(),
        q.VariationalQuantumClassifier(),
        q.VariationalQuantumRegressor(),
        q.LogisticClassifier(),
        q.RidgeRegressor(),
        q.FourierRegressor(),
    ],
)
def test_models_fail_clearly_before_fit(model):
    with pytest.raises(q.NotFittedError):
        model.predict([[1.0]])


def test_classical_baselines_and_parameter_count():
    features = np.linspace(-np.pi, np.pi, 40)[:, None]
    targets = np.sin(3 * features[:, 0])
    model = q.FourierRegressor(harmonics=3, regularization=1e-8).fit(features, targets)
    assert model.n_parameters_ == 7
    assert model.score(features, targets) > 0.99999
    linear = q.RidgeRegressor(regularization=1e-8).fit(features, features[:, 0] * 0.3 + 0.7)
    assert linear.score(features, features[:, 0] * 0.3 + 0.7) > 0.99999
    classifier = q.LogisticClassifier().fit(features, features[:, 0] > 0)
    assert classifier.score(features, features[:, 0] > 0) > 0.95


@pytest.mark.parametrize("name", [d["id"] for d in q.list_datasets()])
def test_offline_datasets_reproducible_split_and_fingerprint(name):
    first = q.load_dataset(name, n_samples=32, seed=2)
    second = q.load_dataset(name, n_samples=32, seed=2)
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    assert first.fingerprint == second.fingerprint
    a, b, c, d = first.split(seed=3)
    assert len(a) + len(b) == 32
    assert len(c) == len(a) and len(d) == len(b)
    changed = q.load_dataset(name, n_samples=32, seed=3)
    assert changed.fingerprint != first.fingerprint
    if first.task_type == "classification":
        np.testing.assert_array_equal(np.unique(c), np.unique(first.y))
        np.testing.assert_array_equal(np.unique(d), np.unique(first.y))


def test_custom_task_and_benchmark_json(tmp_path):
    features = np.arange(20, dtype=float)[:, None]
    task = q.make_task(
        features, 0.3 * features[:, 0] + 1, name="linear-user", task_type="regression"
    )
    result = q.run_benchmark(q.RidgeRegressor(regularization=1e-7), task, seed=13)
    assert result.metrics["r2"] > 0.999
    payload = json.loads(result.save(tmp_path / "result.json").read_text())
    assert payload["provenance"]["split_seed"] == 13
    assert payload["provenance"]["dataset_sha256"] == task.dataset.fingerprint
    assert payload["resources"]["train_samples"] == 15
    assert payload["provenance"]["parameter_matched"] is False
    assert payload["provenance"]["simulation"] is None


def test_comparison_has_identical_splits_and_all_seeds(tmp_path):
    comparison = q.compare_models("moons", seeds=(2, 5), epochs=2)
    assert len(comparison.results) == 6
    assert len({r.provenance["dataset_sha256"] for r in comparison.results}) == 1
    assert set(comparison.summary()) == {"quantum-kernel", "vqc", "logistic"}
    assert comparison.summary()["vqc"]["seeds"] == [2, 5]
    payload = json.loads(comparison.save(tmp_path / "comparison.json").read_text())
    assert payload["summary"] == comparison.summary()
    for seed in (2, 5):
        items = [r for r in comparison.results if r.seed == seed]
        assert (
            len({(r.resources["train_samples"], r.resources["test_samples"]) for r in items}) == 1
        )


def test_catalog_suites_and_input_failures():
    assert tuple(q.get_suite("qml-cls-1")) == ("xor", "circles", "moons")
    assert len(q.get_suite("qml-reg-1")) == 3
    assert q.get_model("vqc", epochs=3).epochs == 3
    for action in (
        lambda: q.get_task("missing"),
        lambda: q.get_suite("missing"),
        lambda: q.load_dataset("xor", noise=-1),
        lambda: q.get_model("missing"),
        lambda: q.run_benchmark("vqr", "xor"),
        lambda: q.compare_models(seeds=[]),
        lambda: q.compare_models(seeds=[1, 1]),
        lambda: q.QuantumKernel().pairwise(np.zeros((2001, 1))),
        lambda: q.QuantumKernel().pairwise(np.ones((3, 9))),
        lambda: q.RidgeRegressor().fit([[1], [2]], [float("nan"), 1]),
    ):
        with pytest.raises((ValueError, q.ResourceLimitError)):
            action()
    estimator = q.LogisticClassifier()
    with pytest.raises(ValueError, match="fresh"):
        q.compare_models("moons", models={"same": lambda: estimator}, seeds=(1, 2))
