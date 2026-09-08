"""Evaluation measures used by tasks and runs.

Every measure has the signature ``fn(y_true, y_pred, y_proba=None) -> float`` so
that the run layer can apply them uniformly. ``higher_is_better`` records the
direction, which the leaderboard needs in order to sort.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

__all__ = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "log_loss",
    "roc_auc",
    "mean_squared_error",
    "mean_absolute_error",
    "r2",
    "absolute_energy_error",
    "state_fidelity",
    "get_measure",
    "list_measures",
    "higher_is_better",
]


def _as_1d(a):
    return np.asarray(a).reshape(-1)


def accuracy(y_true, y_pred, y_proba=None) -> float:
    y_true, y_pred = _as_1d(y_true), _as_1d(y_pred)
    return float(np.mean(y_true == y_pred))


def balanced_accuracy(y_true, y_pred, y_proba=None) -> float:
    y_true, y_pred = _as_1d(y_true), _as_1d(y_pred)
    recalls = []
    for label in np.unique(y_true):
        mask = y_true == label
        recalls.append(np.mean(y_pred[mask] == label))
    return float(np.mean(recalls))


def f1_macro(y_true, y_pred, y_proba=None) -> float:
    y_true, y_pred = _as_1d(y_true), _as_1d(y_pred)
    scores = []
    for label in np.unique(y_true):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(scores))


def log_loss(y_true, y_pred, y_proba=None, eps: float = 1e-12) -> float:
    if y_proba is None:
        raise ValueError("log_loss requires probability estimates")
    y_true = _as_1d(y_true)
    proba = np.clip(np.asarray(y_proba, dtype=float), eps, 1.0)
    proba = proba / proba.sum(axis=1, keepdims=True)
    classes = np.unique(y_true)
    index = {c: i for i, c in enumerate(classes)}
    picked = np.array([proba[i, index[c]] for i, c in enumerate(y_true)])
    return float(-np.mean(np.log(picked)))


def roc_auc(y_true, y_pred, y_proba=None) -> float:
    """Binary ROC AUC via the rank statistic (ties averaged)."""
    y_true = _as_1d(y_true)
    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError("roc_auc is only defined for binary problems")
    if y_proba is None:
        scores = _as_1d(y_pred).astype(float)
    else:
        scores = np.asarray(y_proba, dtype=float)[:, 1]
    positive = y_true == classes[1]
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    start = 0
    for i in range(1, len(sorted_scores) + 1):
        if i == len(sorted_scores) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mean_squared_error(y_true, y_pred, y_proba=None) -> float:
    return float(np.mean((_as_1d(y_true).astype(float) - _as_1d(y_pred).astype(float)) ** 2))


def mean_absolute_error(y_true, y_pred, y_proba=None) -> float:
    return float(np.mean(np.abs(_as_1d(y_true).astype(float) - _as_1d(y_pred).astype(float))))


def r2(y_true, y_pred, y_proba=None) -> float:
    y_true = _as_1d(y_true).astype(float)
    residual = np.sum((y_true - _as_1d(y_pred).astype(float)) ** 2)
    total = np.sum((y_true - y_true.mean()) ** 2)
    return float("nan") if total == 0 else float(1.0 - residual / total)


def absolute_energy_error(y_true, y_pred, y_proba=None) -> float:
    """|E_estimated - E_reference| in Hartree-like units, for VQE style tasks."""
    return float(np.abs(float(np.asarray(y_pred).item()) - float(np.asarray(y_true).item())))


def state_fidelity(psi, phi, y_proba=None) -> float:
    """|<psi|phi>|^2 for two normalised statevectors."""
    psi = np.asarray(psi).reshape(-1)
    phi = np.asarray(phi).reshape(-1)
    return float(np.abs(np.vdot(psi, phi)) ** 2)


_MEASURES: Dict[str, Callable] = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "f1_macro": f1_macro,
    "log_loss": log_loss,
    "roc_auc": roc_auc,
    "mean_squared_error": mean_squared_error,
    "mean_absolute_error": mean_absolute_error,
    "r2": r2,
    "absolute_energy_error": absolute_energy_error,
    "state_fidelity": state_fidelity,
}

_HIGHER_IS_BETTER = {
    "accuracy": True,
    "balanced_accuracy": True,
    "f1_macro": True,
    "log_loss": False,
    "roc_auc": True,
    "mean_squared_error": False,
    "mean_absolute_error": False,
    "r2": True,
    "absolute_energy_error": False,
    "state_fidelity": True,
}


def get_measure(name: str) -> Callable:
    try:
        return _MEASURES[name]
    except KeyError:
        raise ValueError(f"unknown evaluation measure {name!r}; "
                         f"available: {sorted(_MEASURES)}") from None


def list_measures() -> Dict[str, bool]:
    """Map every measure name to whether a larger value is better."""
    return dict(_HIGHER_IS_BETTER)


def higher_is_better(name: str) -> bool:
    return _HIGHER_IS_BETTER[name]
