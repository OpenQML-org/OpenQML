"""Task entities: a dataset plus everything needed to compare two results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .._store import get_store
from ..entities import OpenQMLEntity
from ..metrics import get_measure, higher_is_better
from .split import Split, make_split

__all__ = [
    "OpenQMLTask", "SupervisedClassificationTask", "SupervisedRegressionTask",
    "StateClassificationTask", "GroundStateEstimationTask", "TASK_TYPES", "task_from_dict",
]

TASK_TYPES = (
    "supervised_classification",
    "supervised_regression",
    "state_classification",
    "ground_state_estimation",
)


class OpenQMLTask(OpenQMLEntity):
    """A dataset, a target, a split rule and a measure -- fixed, so runs compare."""

    _entity_name = "tasks"
    _repr_fields = ("id", "name", "task_type", "dataset_id", "target_name",
                    "evaluation_measure", "qubits", "tags")
    task_type = "supervised_classification"

    def __init__(self, name: str, dataset_id: int, target_name: str = "class",
                 evaluation_measure: str = "accuracy", estimation_procedure: Optional[dict] = None,
                 description: str = "", qubits: Optional[int] = None, tags=None,
                 id: Optional[int] = None, task_type: Optional[str] = None,
                 uploaded_at: Optional[str] = None):
        self.id = id
        self.name = name
        self.dataset_id = int(dataset_id)
        self.target_name = target_name
        self.evaluation_measure = evaluation_measure
        self.estimation_procedure = dict(
            estimation_procedure or {"type": "crossvalidation", "folds": 5, "repeats": 1,
                                     "stratified": True, "seed": 0}
        )
        self.description = description
        self.qubits = qubits
        self.tags = list(tags or [])
        self.uploaded_at = uploaded_at
        if task_type:
            self.task_type = task_type
        self._dataset = None
        self._split: Optional[Split] = None

    # -- data -------------------------------------------------------------
    def get_dataset(self):
        from ..datasets import get_dataset

        if self._dataset is None:
            self._dataset = get_dataset(self.dataset_id)
        return self._dataset

    def get_X_and_y(self) -> Tuple[np.ndarray, np.ndarray]:
        """Features and target as arrays."""
        X, y, _ = self.get_dataset().get_data(target=self.target_name)
        return np.asarray(X), np.asarray(y)

    def download_split(self) -> Split:
        """The task's split object (built deterministically from its seed)."""
        if self._split is None:
            X, y = self.get_X_and_y()
            stratify = y if self.task_type in ("supervised_classification", "state_classification") else None
            self._split = make_split(len(X), stratify, self.estimation_procedure)
        return self._split

    def get_train_test_split_indices(self, fold: int = 0, repeat: int = 0):
        return self.download_split().get(repeat, fold)

    @property
    def n_folds(self) -> int:
        return self.download_split().n_folds

    @property
    def n_repeats(self) -> int:
        return self.download_split().n_repeats

    # -- scoring ----------------------------------------------------------
    def score(self, y_true, y_pred, y_proba=None, measure: Optional[str] = None) -> float:
        """Apply the task's evaluation measure (or an alternative one)."""
        return get_measure(measure or self.evaluation_measure)(y_true, y_pred, y_proba)

    @property
    def higher_is_better(self) -> bool:
        return higher_is_better(self.evaluation_measure)

    @property
    def class_labels(self) -> Optional[List]:
        if "classification" not in self.task_type:
            return None
        return sorted(np.unique(self.get_X_and_y()[1]).tolist())

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "task_type": self.task_type,
            "dataset_id": self.dataset_id,
            "target_name": self.target_name,
            "evaluation_measure": self.evaluation_measure,
            "estimation_procedure": self.estimation_procedure,
            "description": self.description,
            "qubits": self.qubits,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OpenQMLTask":
        return task_from_dict(payload)


class SupervisedClassificationTask(OpenQMLTask):
    task_type = "supervised_classification"


class SupervisedRegressionTask(OpenQMLTask):
    task_type = "supervised_regression"


class StateClassificationTask(OpenQMLTask):
    """Classification where the inputs are already quantum states."""

    task_type = "state_classification"

    def get_X_and_y(self):
        dataset = self.get_dataset()
        states = dataset.get_statevectors()
        _, y, _ = dataset.get_data(target=self.target_name)
        return states, np.asarray(y)


class GroundStateEstimationTask(OpenQMLTask):
    """Estimate the ground-state energy of every Hamiltonian in a dataset."""

    task_type = "ground_state_estimation"

    def get_hamiltonians(self) -> List[dict]:
        return self.get_dataset().get_hamiltonians()

    def get_X_and_y(self):
        X, y, _ = self.get_dataset().get_data(target=self.target_name)
        return np.asarray(X), np.asarray(y)


_TASK_CLASSES = {
    "supervised_classification": SupervisedClassificationTask,
    "supervised_regression": SupervisedRegressionTask,
    "state_classification": StateClassificationTask,
    "ground_state_estimation": GroundStateEstimationTask,
}


def task_from_dict(payload: Dict[str, Any]) -> OpenQMLTask:
    task_type = payload.get("task_type", "supervised_classification")
    if task_type not in _TASK_CLASSES:
        raise ValueError(f"unknown task type {task_type!r}; expected one of {TASK_TYPES}")
    known = {"id", "name", "dataset_id", "target_name", "evaluation_measure",
             "estimation_procedure", "description", "qubits", "tags", "uploaded_at"}
    return _TASK_CLASSES[task_type](**{k: v for k, v in payload.items() if k in known})
