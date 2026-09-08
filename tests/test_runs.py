import numpy as np
import pytest

import openqml
from openqml.exceptions import OpenQMLError
from openqml.models import ClassicalKernelClassifier, MajorityClassifier, VQE


def test_run_produces_one_prediction_per_row():
    task = openqml.get_task(1)
    run = openqml.run_model_on_task(ClassicalKernelClassifier(), task)
    assert len(run.predictions) == len(task.get_X_and_y()[0])
    assert run.get_metric_fn("accuracy").shape == (task.n_folds,)
    assert run.mean("accuracy") > 0.5


def test_run_publishes_and_appears_on_the_leaderboard():
    task = openqml.get_task(1)
    run = openqml.run_model_on_task(ClassicalKernelClassifier(), task).publish()
    board = openqml.leaderboard(1, output_format="records")
    assert board[0]["run_id"] == run.id
    assert openqml.runs.get_run(run.id).mean("accuracy") == pytest.approx(run.mean("accuracy"))


def test_duplicate_runs_are_refused_unless_allowed():
    task = openqml.get_task(1)
    openqml.run_model_on_task(MajorityClassifier(), task).publish()
    with pytest.raises(OpenQMLError):
        openqml.run_model_on_task(MajorityClassifier(), task).publish()
    openqml.config.avoid_duplicate_runs = False
    try:
        openqml.run_model_on_task(MajorityClassifier(), task).publish()
    finally:
        openqml.config.avoid_duplicate_runs = True


def test_run_survives_a_filesystem_round_trip(tmp_path):
    run = openqml.run_model_on_task(ClassicalKernelClassifier(), openqml.get_task(6))
    run.to_filesystem(tmp_path)
    restored = openqml.OpenQMLRun.from_filesystem(tmp_path)
    assert restored.run_hash == run.run_hash
    assert np.allclose(restored.get_metric_fn("accuracy"), run.get_metric_fn("accuracy"))


def test_ground_state_task_scores_every_hamiltonian():
    task = openqml.get_task(4)
    run = openqml.run_model_on_task(VQE(layers=1, maxiter=15), task)
    assert len(run.predictions) == len(task.get_hamiltonians())
    assert run.evaluations["absolute_energy_error"] >= 0.0


def test_ground_state_task_rejects_a_plain_classifier():
    with pytest.raises(OpenQMLError):
        openqml.run_model_on_task(MajorityClassifier(), openqml.get_task(4))


def test_flow_round_trip_reproduces_the_model():
    model = ClassicalKernelClassifier(kernel="rbf", alpha=0.01)
    flow = openqml.model_to_flow(model)
    rebuilt = openqml.flow_to_model(flow)
    assert type(rebuilt) is type(model)
    assert rebuilt.get_params() == model.get_params()


def test_flow_refuses_to_import_arbitrary_modules():
    flow = openqml.OpenQMLFlow(name="evil", model_class="os.system")
    with pytest.raises(OpenQMLError):
        flow.to_model()


def test_compare_returns_a_row_per_model():
    table = openqml.compare(
        [ClassicalKernelClassifier(), MajorityClassifier()],
        [openqml.get_task(1)],
        output_format="records",
    )
    assert len(table) == 2
    assert all("moons-2d / binary classification" in row for row in table)


def test_compare_separates_two_configurations_of_one_class():
    from openqml.models import QuantumKernelClassifier

    table = openqml.compare(
        [QuantumKernelClassifier(feature_map="iqp"), QuantumKernelClassifier(feature_map="zz")],
        [openqml.get_task(2)],
        output_format="records",
    )
    assert len(table) == 2
    assert {row["model"] for row in table} == {
        "QuantumKernelClassifier[feature_map=iqp]",
        "QuantumKernelClassifier[feature_map=zz]",
    }


def test_suite_runs_every_task():
    suite = openqml.get_suite("qml-cls-1")
    runs = suite.run(ClassicalKernelClassifier())
    assert len(runs) == len(suite.task_ids)
    assert {r.task_id for r in runs} == set(suite.task_ids)


def test_a_classifier_is_refused_on_a_regression_task():
    with pytest.raises(OpenQMLError):
        openqml.run_model_on_task(ClassicalKernelClassifier(), openqml.get_task(5))
