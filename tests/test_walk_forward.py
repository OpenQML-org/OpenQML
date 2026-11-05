"""Causal train/validation/test boundaries and aligned rolling predictions."""

import numpy as np
import pytest

import openqml as oq


class TrackingModel(oq.Model):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def fit(self, x, y, **options):
        self.log.append((x.copy(), y.copy(), options))
        self.trainer = object()
        self.mean = float(y.mean())
        return self

    def predict(self, x, **options):
        return np.full(len(x), self.mean)

    def evaluate(self, x, y, **options):
        return float(np.mean((y - self.mean) ** 2))


def run(log, **options):
    n = 20
    arguments = dict(
        model_factory=lambda: TrackingModel(log),
        inputs=np.arange(n)[:, None],
        targets=np.arange(n) * 0.01,
        times=np.arange(n),
        label_end=np.arange(n) + 2,
        splitter=oq.finance.PurgedWalkForward(8, 4),
    )
    arguments.update(options)
    return oq.finance.walk_forward(**arguments)


def test_nested_purge_respects_label_boundaries_and_alignment():
    log = []
    result = run(log, validation_size=2, fit_options={"patience": 2})
    assert len(result.folds) == len(log) == 3
    for i, ((train, _, options), fold) in enumerate(zip(log, result.folds, strict=True)):
        validation = options["validation_data"][0]
        assert np.all(train[:, 0] + 2 < validation[0, 0])
        assert np.all(validation[:, 0] + 2 < fold.test_start)
        assert fold.train_size == len(train) and fold.validation_size == 2
        assert fold.purged_size == 4 and options["patience"] == 2
        assert np.all(result.fold_id[fold.test_start : fold.test_stop] == i)
        assert np.all(result.predictions[fold.test_start : fold.test_stop] == log[i][1].mean())
    assert not result.mask[:8].any() and result.mask[8:].all()
    assert np.isnan(result.predictions[:8]).all() and (result.fold_id[:8] == -1).all()


def test_future_target_changes_leave_earlier_predictions_unchanged():
    reference = run([])
    y = np.arange(20) * 0.01
    y[16:] += 100
    changed = run([], targets=y)
    np.testing.assert_array_equal(reference.predictions[:16], changed.predictions[:16])


@pytest.mark.parametrize(
    "options",
    [
        {"splitter": oq.finance.PurgedWalkForward(8, 4, step_size=2)},
        {"validation_size": 7},
        {"validation_size": 4},
        {"fit_options": {"validation_data": ([], [])}},
        {"inputs": np.zeros((19, 1))},
        {"targets": np.full(20, np.nan)},
        {"max_memory_mb": 1e-8},
        {"max_memory_mb": -1},
        {"times": np.zeros(20)},
        {"label_end": np.arange(20) - 1},
        {"model_factory": lambda: object()},
    ],
)
def test_preflight_rejects_overlap_invalid_data_and_resource_overflow(options):
    with pytest.raises((oq.ValidationError, oq.ResourceError)):
        run([], **options)


def test_fitted_factory_instances_are_rejected():
    model = TrackingModel([])
    with pytest.raises(oq.ValidationError, match="fresh"):
        run([], model_factory=lambda: model)


@pytest.mark.parametrize("framework", ["torch", "jax", "flax"])
def test_walk_forward_uses_native_hybrid_fit_validation_and_prediction(framework):
    pytest.importorskip("torch" if framework == "torch" else "jax")
    if framework != "torch":
        pytest.importorskip("optax")
    if framework == "flax":
        pytest.importorskip("flax")
    result = run(
        [],
        validation_size=2,
        splitter=oq.finance.PurgedWalkForward(10, 5),
        model_factory=lambda: oq.Model(
            framework=framework,
            backend="torch" if framework == "torch" else "jax",
            hidden_features=4,
            depth=1,
        ),
        fit_options={"epochs": 2, "patience": 1, "min_delta": 1e6},
    )
    assert result.predictions.shape == (20,)
    assert np.isfinite(result.predictions[result.mask]).all()
    assert all(f.best_epoch == 1 and np.isfinite(f.test_loss) for f in result.folds)
    if framework != "torch":
        import jax

        jax.clear_caches()
