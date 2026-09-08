"""Deterministic train/test splits.

Splits are derived from the seed recorded on the task, so two people running
the same task compare like with like without exchanging index files.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

__all__ = ["Split", "make_split"]


class Split:
    """Container for ``(repeat, fold) -> (train indices, test indices)``."""

    def __init__(self, folds: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
                 procedure: dict):
        self._folds = folds
        self.procedure = dict(procedure)

    @property
    def n_repeats(self) -> int:
        return max((repeat for repeat, _ in self._folds), default=0) + 1

    @property
    def n_folds(self) -> int:
        return max((fold for _, fold in self._folds), default=0) + 1

    def get(self, repeat: int = 0, fold: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        try:
            return self._folds[(int(repeat), int(fold))]
        except KeyError:
            raise ValueError(f"no split for repeat={repeat}, fold={fold}") from None

    def __iter__(self) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray]]:
        for (repeat, fold), (train, test) in sorted(self._folds.items()):
            yield repeat, fold, train, test

    def __len__(self) -> int:
        return len(self._folds)

    def __repr__(self) -> str:
        return (f"<Split {self.procedure.get('type')} repeats={self.n_repeats} "
                f"folds={self.n_folds}>")


def _stratified_folds(y: np.ndarray, n_folds: int, rng) -> List[np.ndarray]:
    assignment = np.empty(len(y), dtype=int)
    for label in np.unique(y):
        index = np.flatnonzero(y == label)
        rng.shuffle(index)
        assignment[index] = np.arange(len(index)) % n_folds
    return [np.flatnonzero(assignment == f) for f in range(n_folds)]


def _plain_folds(n_samples: int, n_folds: int, rng) -> List[np.ndarray]:
    order = rng.permutation(n_samples)
    return [np.sort(chunk) for chunk in np.array_split(order, n_folds)]


def make_split(n_samples: int, y: Optional[np.ndarray], procedure: dict) -> Split:
    """Build the split described by a task's ``estimation_procedure``."""
    kind = procedure.get("type", "crossvalidation")
    seed = int(procedure.get("seed", 0))
    folds: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    if kind == "none":
        everything = np.arange(n_samples)
        return Split({(0, 0): (everything, everything)}, procedure)

    if kind == "holdout":
        test_size = float(procedure.get("test_size", 0.33))
        for repeat in range(int(procedure.get("repeats", 1))):
            rng = np.random.default_rng(seed + repeat)
            order = rng.permutation(n_samples)
            cut = int(round(n_samples * (1 - test_size)))
            folds[(repeat, 0)] = (np.sort(order[:cut]), np.sort(order[cut:]))
        return Split(folds, procedure)

    if kind == "crossvalidation":
        n_folds = int(procedure.get("folds", 5))
        stratified = bool(procedure.get("stratified", True)) and y is not None
        for repeat in range(int(procedure.get("repeats", 1))):
            rng = np.random.default_rng(seed + repeat)
            groups = (_stratified_folds(np.asarray(y), n_folds, rng) if stratified
                      else _plain_folds(n_samples, n_folds, rng))
            for fold in range(n_folds):
                test = np.sort(groups[fold])
                train = np.sort(np.setdiff1d(np.arange(n_samples), test))
                folds[(repeat, fold)] = (train, test)
        return Split(folds, procedure)

    raise ValueError(f"unknown estimation procedure {kind!r}")
