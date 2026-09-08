"""The run entity: predictions, per-fold scores and enough metadata to trust them."""

from __future__ import annotations

import json
import pathlib
import platform
from typing import Any, Dict, List, Optional

import numpy as np

from .._store import get_store
from ..entities import OpenQMLEntity
from ..exceptions import OpenQMLError
from ..utils import as_table, jsonify, stable_hash

__all__ = ["OpenQMLRun"]


class OpenQMLRun(OpenQMLEntity):
    """One model, one task, one protocol -- plus the numbers it produced."""

    _entity_name = "runs"
    _repr_fields = ("id", "task_id", "flow_name", "evaluations", "backend", "shots",
                    "runtime_seconds", "tags")

    def __init__(self, task_id: int, flow_name: str, flow_id: Optional[int] = None,
                 parameter_settings: Optional[dict] = None, predictions=None,
                 fold_evaluations: Optional[dict] = None, evaluations: Optional[dict] = None,
                 backend: str = "", shots: Optional[int] = None, seed: Optional[int] = None,
                 runtime_seconds: Optional[float] = None, description: str = "",
                 tags=None, id: Optional[int] = None, task_type: str = "",
                 evaluation_measure: str = "", environment: Optional[dict] = None,
                 uploaded_at: Optional[str] = None, model=None):
        self.id = id
        self.task_id = int(task_id)
        self.task_type = task_type
        self.evaluation_measure = evaluation_measure
        self.flow_name = flow_name
        self.flow_id = flow_id
        self.parameter_settings = dict(parameter_settings or {})
        self.predictions = list(predictions or [])
        self.fold_evaluations = fold_evaluations or {}
        self.evaluations = evaluations or {}
        self.backend = backend
        self.shots = shots
        self.seed = seed
        self.runtime_seconds = runtime_seconds
        self.description = description
        self.tags = list(tags or [])
        self.environment = environment or {
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        self.uploaded_at = uploaded_at
        self.model = model

    # -- results ----------------------------------------------------------
    def get_metric_fn(self, measure: str) -> np.ndarray:
        """Per-fold values of one measure, flattened across repeats."""
        try:
            per_repeat = self.fold_evaluations[measure]
        except KeyError:
            raise OpenQMLError(
                f"run has no measure {measure!r}; available: {sorted(self.fold_evaluations)}"
            ) from None
        values = [value for repeat in sorted(per_repeat)
                  for _, value in sorted(per_repeat[repeat].items())]
        return np.array(values, dtype=float)

    def mean(self, measure: Optional[str] = None) -> float:
        return float(np.mean(self.get_metric_fn(measure or self.evaluation_measure)))

    def std(self, measure: Optional[str] = None) -> float:
        return float(np.std(self.get_metric_fn(measure or self.evaluation_measure)))

    def get_predictions(self, output_format: str = "dataframe"):
        return as_table(self.predictions, output_format)

    def summary(self) -> str:
        lines = [f"run on task {self.task_id} with {self.flow_name}"]
        for measure in sorted(self.fold_evaluations):
            values = self.get_metric_fn(measure)
            lines.append(f"  {measure}: {values.mean():.4f} +/- {values.std():.4f} "
                         f"over {len(values)} fold(s)")
        if self.runtime_seconds is not None:
            lines.append(f"  runtime: {self.runtime_seconds:.2f}s on {self.backend}"
                         + (f" ({self.shots} shots)" if self.shots else " (analytic)"))
        return "\n".join(lines)

    # -- identity ---------------------------------------------------------
    @property
    def run_hash(self) -> str:
        """Identifies task + model + parameters + seed; used to spot duplicates."""
        return stable_hash({
            "task_id": self.task_id,
            "flow_name": self.flow_name,
            "parameters": self.parameter_settings,
            "seed": self.seed,
            "shots": self.shots,
            "backend": self.backend,
        })

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "evaluation_measure": self.evaluation_measure,
            "flow_id": self.flow_id,
            "flow_name": self.flow_name,
            "parameter_settings": jsonify(self.parameter_settings),
            "predictions": jsonify(self.predictions),
            "fold_evaluations": jsonify(self.fold_evaluations),
            "evaluations": jsonify(self.evaluations),
            "backend": self.backend,
            "shots": self.shots,
            "seed": self.seed,
            "runtime_seconds": self.runtime_seconds,
            "description": self.description,
            "tags": self.tags,
            "environment": self.environment,
            "run_hash": self.run_hash,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OpenQMLRun":
        known = {"id", "task_id", "task_type", "evaluation_measure", "flow_id", "flow_name",
                 "parameter_settings", "predictions", "fold_evaluations", "evaluations",
                 "backend", "shots", "seed", "runtime_seconds", "description", "tags",
                 "environment", "uploaded_at"}
        payload = {k: v for k, v in payload.items() if k in known}
        folds = payload.get("fold_evaluations") or {}
        payload["fold_evaluations"] = {
            measure: {int(repeat): {int(fold): float(value) for fold, value in per_fold.items()}
                      for repeat, per_fold in per_repeat.items()}
            for measure, per_repeat in folds.items()
        }
        return cls(**payload)

    def publish(self) -> "OpenQMLRun":
        from .. import config
        from .functions import run_exists

        if config.avoid_duplicate_runs:
            existing = run_exists(self)
            if existing is not None:
                raise OpenQMLError(
                    f"an identical run is already published (id={existing}). Set "
                    f"openqml.config.avoid_duplicate_runs = False to publish it anyway."
                )
        response = get_store().post(self._entity_name, self.to_dict())
        self.id = int(response["id"])
        return self

    def to_filesystem(self, directory) -> pathlib.Path:
        """Write the run as JSON so it can be shared without a server."""
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "run.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_filesystem(cls, directory) -> "OpenQMLRun":
        path = pathlib.Path(directory)
        if path.is_dir():
            path = path / "run.json"
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
