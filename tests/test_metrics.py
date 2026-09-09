"""Measures, and the layer that applies them to a fold.

The recurring hazard here is that ``y_proba``'s columns follow the *estimator's*
class ordering, fixed when it was fit on the training fold, while the obvious
thing to hand a measure is the labels present in the test fold. Those are not
the same list.
"""

import numpy as np
import pytest

import openqml
from openqml import metrics
from openqml.exceptions import DuplicateRunError, OpenQMLError
from openqml.runs.functions import _score


def test_log_loss_uses_the_estimators_class_ordering():
    """A fold missing a class must not shift every column after it."""
    classes = [0, 1, 2]
    proba = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    y_true = np.array([0, 2])  # class 1 never appears in this fold

    correct = -np.mean(np.log([0.6, 0.6]))
    assert metrics.log_loss(y_true, None, proba, classes=classes) == pytest.approx(correct)

    # Without the ordering there is nothing to go on but the fold's own labels,
    # which would map class 2 onto column 1 and quietly return -mean(log([0.6,
    # 0.2])). That now raises instead of answering wrongly.
    with pytest.raises(ValueError, match="columns"):
        metrics.log_loss(y_true, None, proba)


def test_log_loss_refuses_a_label_the_estimator_never_saw():
    proba = np.array([[0.5, 0.5]])
    with pytest.raises(ValueError, match="not among the estimator"):
        metrics.log_loss([7], None, proba, classes=[0, 1])


def test_log_loss_refuses_a_mismatched_probability_matrix():
    with pytest.raises(ValueError, match="columns"):
        metrics.log_loss([0, 1], None, np.array([[0.5, 0.5], [0.5, 0.5]]), classes=[0, 1, 2])


def test_roc_auc_takes_the_positive_class_from_the_estimator():
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    y_true = np.array([0, 0, 0])  # a fold with only the negative class
    assert np.isnan(metrics.roc_auc(y_true, None, proba, classes=[0, 1]))
    with pytest.raises(ValueError, match="binary"):
        metrics.roc_auc(y_true, None, proba)  # one label present -> not binary


def test_scoring_a_fold_passes_the_class_ordering_through():
    proba = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    scored = _score(["log_loss"], np.array([0, 2]), np.array([0, 2]), proba, classes=[0, 1, 2])
    assert scored["log_loss"] == pytest.approx(-np.mean(np.log([0.6, 0.6])))


def test_a_measure_that_does_not_apply_is_skipped_but_a_bug_is_not():
    """The blanket except used to drop a broken measure as if it were N/A."""
    assert _score(["log_loss"], [0, 1], [0, 1], None) == {}  # no probabilities: N/A

    def exploding(y_true, y_pred, y_proba=None):
        raise ZeroDivisionError("a real bug, not a fold that does not apply")

    metrics._MEASURES["exploding"] = exploding
    try:
        with pytest.raises(ZeroDivisionError):
            _score(["exploding"], [0, 1], [0, 1], None)
    finally:
        del metrics._MEASURES["exploding"]


def test_state_fidelity_cannot_exceed_one():
    psi = np.array([1.0 + 0j, 0.0])
    assert metrics.state_fidelity(psi, psi * 3.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        metrics.state_fidelity(psi, np.zeros(2))


def test_an_unknown_measure_raises_the_libraries_own_error():
    with pytest.raises(ValueError, match="unknown evaluation measure"):
        metrics.higher_is_better("no-such-measure")


# -- publishing --------------------------------------------------------------

def test_a_duplicate_run_raises_its_own_error_type():
    run = openqml.run_model_on_task(openqml.MajorityClassifier(), openqml.get_task(1))
    run.publish()
    again = openqml.run_model_on_task(openqml.MajorityClassifier(), openqml.get_task(1),
                                      avoid_duplicate_runs=False)
    with pytest.raises(DuplicateRunError):
        again.publish()
    assert issubclass(DuplicateRunError, OpenQMLError)


def test_bulk_publishing_does_not_swallow_a_real_failure(monkeypatch):
    """compare(publish=True) skips duplicates -- and only duplicates."""
    from openqml.runs import run as run_module

    def broken_publish(self):
        raise OpenQMLError("server said no")

    monkeypatch.setattr(run_module.OpenQMLRun, "publish", broken_publish)
    with pytest.raises(OpenQMLError, match="server said no"):
        openqml.compare([openqml.MajorityClassifier()], [openqml.get_task(1)], publish=True)


def test_the_leaderboard_reports_folds_not_repeats_times_folds():
    task = openqml.get_task(1)
    openqml.run_model_on_task(openqml.MajorityClassifier(), task).publish()
    row = openqml.leaderboard(task.id, output_format="records")[0]
    assert row["folds"] == task.estimation_procedure["folds"]
    assert row["repeats"] == 1
