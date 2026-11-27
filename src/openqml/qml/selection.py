"""Deployment selection using held-out, paired bounded losses."""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from ..errors import ValidationError
from ..ir.core import integer
from ..ir.parameters import real


@dataclass(frozen=True)
class PredictorSelection:
    """The selected callable and the evidence used for that selection."""

    selected: str
    predict: Callable = field(repr=False, compare=False)
    mean_losses: Mapping[str, float]
    log_e_values: Mapping[str, float]
    costs: Mapping[str, float]
    max_cost: float
    observations: int
    loss_bound: float
    min_gain: float
    confidence: float


def _length(data):
    if isinstance(data, Mapping):
        lengths = {_length(value) for value in data.values()}
        if len(lengths) != 1:
            raise ValidationError("All input fields must share one observation axis.")
        return lengths.pop()
    try:
        return len(data)
    except TypeError as exc:
        raise ValidationError("Provide data with a leading observation axis.") from exc


def _batch(data, rows):
    if isinstance(data, Mapping):
        return {name: _batch(value, rows) for name, value in data.items()}
    return data[rows]


def _host(value):
    if hasattr(value, "detach"):
        value = value.detach()
        if value.is_floating_point():
            value = value.double()
        value = value.cpu()
    return np.asarray(value)


def _prediction_loss(targets, predictions, task):
    actual, predicted = _host(targets), _host(predictions)
    if task != "classification":
        if actual.shape == predicted.shape + (1,):
            actual = actual[..., 0]
        elif predicted.shape == actual.shape + (1,):
            predicted = predicted[..., 0]
    if actual.ndim < 1 or predicted.shape != actual.shape:
        raise ValidationError("Point predictions and targets must have matching shapes.")
    if task == "classification":
        values = predicted != actual
    else:
        if actual.dtype.kind not in "fiu" or predicted.dtype.kind not in "fiu":
            raise ValidationError("Regression and forecast predictions must be real.")
        residual = predicted.astype(np.float64) - actual.astype(np.float64)
        values = np.abs(residual) if task == "forecast" else np.square(residual)
    return values.reshape(len(actual), -1).mean(axis=1)


def select_predictor(
    hybrid,
    baselines,
    inputs,
    targets,
    *,
    loss,
    loss_bound,
    costs,
    max_cost,
    min_gain=0,
    confidence=0.95,
    batch_size=256,
):
    """Select a fitted predictor on fresh calibration observations under a common cost cap.

    Candidates are prediction callables. The shared loss(targets, predictions)
    returns one value in [0, loss_bound] per independent observation. Register
    the candidates, bound, costs and margin before calibration. Costs include
    all relevant resources in one caller-selected unit.
    """
    if (
        not isinstance(baselines, Mapping)
        or not baselines
        or not callable(hybrid)
        or not callable(loss)
    ):
        raise ValidationError("Supply a hybrid predictor, named classical predictors and a loss.")
    if any(
        not isinstance(k, str) or not k or k == "hybrid" or not callable(v)
        for k, v in baselines.items()
    ):
        raise ValidationError("Baselines must use distinct nonempty names other than 'hybrid'.")
    candidates = {"hybrid": hybrid, **baselines}
    if not isinstance(costs, Mapping) or set(costs) != set(candidates):
        raise ValidationError("Provide a comparable cost for every candidate including 'hybrid'.")
    costs = {name: real(costs[name]) for name in candidates}
    cap, bound, gain, level = map(real, (max_cost, loss_bound, min_gain, confidence))
    if cap < 0 or min(costs.values()) < 0 or not 0 <= gain < bound or not 0 < level < 1:
        raise ValidationError("Invalid costs, loss_bound, min_gain or confidence.")
    count, size = _length(targets), integer(batch_size, "batch_size", 1)
    if not count or _length(inputs) != count:
        raise ValidationError("Use matching nonempty calibration observations.")
    classical = [name for name in baselines if costs[name] <= cap]
    if not classical:
        raise ValidationError("At least one classical baseline must fit max_cost.")
    eligible = [name for name in candidates if costs[name] <= cap]
    totals = dict.fromkeys(eligible, 0.0)
    # A mixture of fixed bets on bounded paired improvements. Under the null
    # conditional mean <= gain, each factor is nonnegative with expectation <= 1.
    bets = np.array([1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 15 / 16])
    logs = {name: np.zeros_like(bets) for name in classical} if "hybrid" in totals else {}
    margin = gain / bound
    for start in range(0, count, size):
        stop = min(start + size, count)
        x, y = _batch(inputs, slice(start, stop)), _batch(targets, slice(start, stop))
        hybrid_loss = None
        for name in eligible:
            values = _host(loss(y, candidates[name](x)))
            if (
                values.shape != (stop - start,)
                or values.dtype.kind not in "fibu"
                or not np.isfinite(values).all()
                or np.any(values < 0)
                or np.any(values > bound)
            ):
                raise ValidationError(
                    f"{name} needs one finite loss in [0, loss_bound] per observation."
                )
            scaled = values.astype(np.float64) / bound
            totals[name] += float(scaled.sum())
            if name == "hybrid":
                hybrid_loss = scaled
            elif hybrid_loss is not None:
                difference = (scaled - hybrid_loss - margin) / (1 + margin)
                logs[name] += np.log1p(difference[:, None] * bets).sum(axis=0)
    means = {name: (total / count) * bound for name, total in totals.items()}
    evidence = {
        name: float(np.logaddexp.reduce(values) - math.log(len(bets)))
        for name, values in logs.items()
    }
    winner = min(classical, key=lambda name: (means[name], costs[name], name))
    threshold = math.log(len(classical)) - math.log1p(-level)
    if evidence and all(
        evidence[name] >= threshold and means[name] - means["hybrid"] > gain for name in classical
    ):
        winner = "hybrid"
    return PredictorSelection(
        winner,
        candidates[winner],
        MappingProxyType(means),
        MappingProxyType(evidence),
        MappingProxyType(costs),
        cap,
        count,
        bound,
        gain,
        level,
    )
