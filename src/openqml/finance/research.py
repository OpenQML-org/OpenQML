"""Purged rolling refits with aligned, out-of-sample hybrid model predictions."""

from dataclasses import dataclass

import numpy as np

from ..compute.kernel import memory_budget
from ..errors import ValidationError
from ..ir.core import integer
from ..qml.models import Model
from ._array import admission
from .validation import PurgedWalkForward


@dataclass(frozen=True)
class WalkForwardMetrics:
    train_size: int
    validation_size: int
    test_size: int
    purged_size: int
    test_start: int
    test_stop: int
    test_loss: float
    best_epoch: int | None


@dataclass(frozen=True)
class WalkForwardResult:
    predictions: np.ndarray
    mask: np.ndarray
    fold_id: np.ndarray
    folds: tuple[WalkForwardMetrics, ...]


def walk_forward(
    model_factory,
    inputs,
    targets,
    *,
    times,
    label_end,
    splitter,
    validation_size=0,
    fit_options=None,
    predict_options=None,
    max_memory_mb=256,
):
    """Refit a fresh Model on each purged training fold and predict its test rows.

    Inputs/targets are finite CPU arrays with a leading observation axis.
    Each model_factory() call must return an unfitted openqml.Model. Optional
    validation uses the chronological training tail with a second label purge.
    Feature fitting and scenario selection belong inside each training fold.
    The result aligns with the original observation axis; uncovered rows hold
    NaN, mask=False and fold_id=-1. Decisions remain separate from bar returns
    so SelfFinancingPlan can apply its execution lag exactly once.
    """
    if not callable(model_factory) or not isinstance(splitter, PurgedWalkForward):
        raise ValidationError("Provide a model factory and a PurgedWalkForward splitter.")
    if splitter.step_size < splitter.test_size:
        raise ValidationError("Out-of-sample test folds must be disjoint: step_size >= test_size.")
    validation_size = integer(validation_size, "validation_size", 0)
    maximum = memory_budget(max_memory_mb)
    if maximum <= 0:
        raise ValidationError("max_memory_mb must be positive.")
    try:
        options, prediction_options = dict(fit_options or {}), dict(predict_options or {})
        x, y = np.asarray(inputs), np.asarray(targets)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Provide numeric arrays and option mappings.") from exc
    if "validation_data" in options:
        raise ValidationError("Use validation_size to select validation data within each fold.")
    if (
        x.ndim < 2
        or y.ndim < 1
        or not len(x)
        or len(x) != len(y)
        or x.dtype.kind not in "fiu"
        or y.dtype.kind not in "fiu"
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.shape(times) != (len(x),)
        or np.shape(label_end) != (len(x),)
    ):
        raise ValidationError("Use finite real inputs, targets and matching observation times.")
    if any(size == 0 for size in (*x.shape, *y.shape)):
        raise ValidationError("Input and target axes must be nonempty.")
    # Validate timestamps and the first split before constructing a native model.
    iterator = splitter.split(times, label_end)
    ends, timestamps = np.asarray(label_end), np.asarray(times)
    count = len(x)
    admission(2 * count, maximum)
    mask, fold_id = np.zeros(count, dtype=bool), np.full(count, -1, dtype=np.int64)
    predictions, summaries = None, []
    sample_elements = int(np.prod(x.shape[1:])) + int(np.prod(y.shape[1:]))
    for number, fold in enumerate(iterator):
        train, validation = fold.train, np.empty(0, dtype=int)
        inner_purged = 0
        if validation_size:
            if validation_size >= len(train):
                raise ValidationError("validation_size must leave a nonempty training fold.")
            train, validation = train[:-validation_size], train[-validation_size:]
            keep = ends[train] < timestamps[validation[0]]
            inner_purged = int(np.count_nonzero(~keep))
            train = train[keep]
            if not len(train):
                raise ValidationError("Validation label purging removed the training fold.")
        output_elements = 0 if predictions is None else predictions.size
        workspace = (len(fold.train) + len(fold.test)) * sample_elements
        admission(workspace + output_elements + 2 * count, maximum)
        model = model_factory()
        if not isinstance(model, Model) or model.trainer is not None:
            raise ValidationError("model_factory must return a fresh, unfitted openqml.Model.")
        fitting = dict(options)
        if validation_size:
            fitting["validation_data"] = (x[validation], y[validation])
        model.fit(x[train], y[train], **fitting)
        predicted = np.asarray(model.predict(x[fold.test], **prediction_options))
        if (
            predicted.ndim < 1
            or predicted.shape[0] != len(fold.test)
            or predicted.dtype.kind not in "fiu"
            or not np.isfinite(predicted).all()
        ):
            raise ValidationError("Model predictions must be finite with one row per test sample.")
        shape = (count,) + predicted.shape[1:]
        admission(workspace + int(np.prod(shape)) + predicted.size + 2 * count, maximum)
        if predictions is None:
            predictions = np.full(shape, np.nan, dtype=np.float64)
        if predictions.shape != shape:
            raise ValidationError("Model prediction shape changed across walk-forward folds.")
        predictions[fold.test] = predicted
        mask[fold.test], fold_id[fold.test] = True, number
        test_loss = model.evaluate(x[fold.test], y[fold.test], **prediction_options)
        summaries.append(
            WalkForwardMetrics(
                len(train),
                len(validation),
                len(fold.test),
                len(fold.purged) + inner_purged,
                int(fold.test[0]),
                int(fold.test[-1]) + 1,
                test_loss,
                model.best_epoch,
            )
        )
        # Release each fold's network and compiled trainer before the next refit.
        del model
    return WalkForwardResult(predictions, mask, fold_id, tuple(summaries))
