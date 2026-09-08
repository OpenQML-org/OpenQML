"""Executing models on tasks -- the part that turns code into a comparable number."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import config
from .._store import get_store
from ..backends import DEFAULT_BACKEND
from ..exceptions import OpenQMLError
from ..flows import model_to_flow, publish_flow
from ..metrics import get_measure
from ..tasks import OpenQMLTask, get_task
from ..utils import as_table, clone, get_params, jsonify
from .run import OpenQMLRun

__all__ = ["run_model_on_task", "run_flow_on_task", "get_run", "list_runs", "run_exists",
           "delete_run", "run_models_on_tasks"]


def _measures_for(task: OpenQMLTask, extra: Optional[Sequence[str]]) -> List[str]:
    measures = [task.evaluation_measure]
    defaults = {
        "supervised_classification": ["accuracy", "balanced_accuracy", "f1_macro"],
        "state_classification": ["accuracy", "balanced_accuracy", "f1_macro"],
        "supervised_regression": ["mean_squared_error", "mean_absolute_error", "r2"],
        "ground_state_estimation": ["absolute_energy_error"],
    }
    for name in defaults.get(task.task_type, []) + list(extra or []):
        if name not in measures:
            measures.append(name)
    return measures


_EXPECTED_ESTIMATOR = {
    "supervised_classification": "classifier",
    "state_classification": "classifier",
    "supervised_regression": "regressor",
    "ground_state_estimation": "ground_state",
}


def _check_model_fits_task(model, task) -> None:
    """Catch the obvious mismatch early rather than scoring nonsense."""
    declared = getattr(model, "_estimator_type", None)
    expected = _EXPECTED_ESTIMATOR.get(task.task_type)
    if declared and expected and declared != expected:
        raise OpenQMLError(
            f"{type(model).__name__} declares itself a {declared}, but task "
            f"{task.id} is a {task.task_type} task and needs a {expected}"
        )


def _seeded(estimator, seed):
    """Override the estimator's own seed when the caller pinned one."""
    if seed is not None and hasattr(estimator, "set_params") and "seed" in get_params(estimator):
        estimator.set_params(seed=seed)
    return estimator


def _score(measures, y_true, y_pred, y_proba) -> Dict[str, float]:
    scores = {}
    for name in measures:
        try:
            scores[name] = float(get_measure(name)(y_true, y_pred, y_proba))
        except Exception:
            continue  # a measure that does not apply to this fold is simply skipped
    return scores


def _run_supervised(model, task, measures, seed):
    X, y = task.get_X_and_y()
    split = task.download_split()
    fold_evaluations: Dict[str, Dict[int, Dict[int, float]]] = {}
    predictions = []
    for repeat, fold, train, test in split:
        estimator = _seeded(clone(model), seed)
        estimator.fit(X[train], y[train])
        y_pred = np.asarray(estimator.predict(X[test]))
        y_proba = None
        if hasattr(estimator, "predict_proba"):
            try:
                y_proba = np.asarray(estimator.predict_proba(X[test]))
            except Exception:
                y_proba = None
        for position, index in enumerate(test):
            record = {"repeat": repeat, "fold": fold, "row_id": int(index),
                      "prediction": jsonify(y_pred[position]), "truth": jsonify(y[index])}
            if y_proba is not None:
                record["confidence"] = float(np.max(y_proba[position]))
            predictions.append(record)
        for name, value in _score(measures, y[test], y_pred, y_proba).items():
            fold_evaluations.setdefault(name, {}).setdefault(repeat, {})[fold] = value
    return fold_evaluations, predictions


def _run_ground_state(model, task, measures, seed):
    records = task.get_hamiltonians()
    fold_evaluations: Dict[str, Dict[int, Dict[int, float]]] = {}
    predictions = []
    for index, record in enumerate(records):
        estimator = _seeded(clone(model), seed)
        if not hasattr(estimator, "estimate"):
            raise OpenQMLError(
                f"{type(model).__name__} cannot run a ground_state_estimation task: it needs an "
                f"estimate(hamiltonian_terms) -> energy method (see openqml.models.VQE)"
            )
        energy = float(estimator.estimate(record["terms"]))
        reference = float(record["reference_energy"])
        predictions.append({
            "repeat": 0, "fold": index, "row_id": index,
            "separation": record.get("separation"),
            "prediction": energy, "truth": reference,
            "error": abs(energy - reference),
        })
        for name, value in _score(measures, [reference], [energy], None).items():
            fold_evaluations.setdefault(name, {}).setdefault(0, {})[index] = value
    return fold_evaluations, predictions


def run_model_on_task(model, task, avoid_duplicate_runs: Optional[bool] = None,
                      seed: Optional[int] = None, extra_measures: Optional[Sequence[str]] = None,
                      upload_flow: bool = False, tags: Optional[Sequence[str]] = None,
                      description: str = "") -> OpenQMLRun:
    """Run ``model`` on ``task`` under the task's own protocol and score it.

    ``model`` can be any estimator with ``fit``/``predict`` (scikit-learn
    included); ground-state tasks instead need an ``estimate(terms)`` method.
    Nothing is uploaded -- call ``.publish()`` on the returned run for that.
    """
    if not isinstance(task, OpenQMLTask):
        task = get_task(task)
    _check_model_fits_task(model, task)
    measures = _measures_for(task, extra_measures)
    flow = model_to_flow(model)

    started = time.time()
    if task.task_type == "ground_state_estimation":
        fold_evaluations, predictions = _run_ground_state(model, task, measures, seed)
    else:
        fold_evaluations, predictions = _run_supervised(model, task, measures, seed)
    runtime = time.time() - started

    evaluations = {
        name: float(np.mean([v for per_fold in per_repeat.values() for v in per_fold.values()]))
        for name, per_repeat in fold_evaluations.items()
    }
    parameters = get_params(model)
    run = OpenQMLRun(
        task_id=task.require_id() if task.id is not None else -1,
        task_type=task.task_type,
        evaluation_measure=task.evaluation_measure,
        flow_name=flow.name,
        flow_id=flow.id,
        parameter_settings=parameters,
        predictions=predictions,
        fold_evaluations=fold_evaluations,
        evaluations=evaluations,
        backend=str(parameters.get("backend") or "classical"),
        shots=parameters.get("shots"),
        seed=seed if seed is not None else parameters.get("seed"),
        runtime_seconds=runtime,
        description=description,
        tags=list(tags or []),
        model=model,
    )
    if upload_flow:
        flow = publish_flow(flow)
        run.flow_id = flow.id
    should_check = config.avoid_duplicate_runs if avoid_duplicate_runs is None else avoid_duplicate_runs
    if should_check:
        existing = run_exists(run)
        if existing is not None:
            run.description = (run.description + f" [identical to published run {existing}]").strip()
    return run


def run_flow_on_task(flow, task, **kwargs) -> OpenQMLRun:
    """Reconstruct the model a flow describes, then run it."""
    from ..flows import flow_to_model

    return run_model_on_task(flow_to_model(flow), task, **kwargs)


def run_models_on_tasks(models, tasks, publish: bool = False, **kwargs) -> List[OpenQMLRun]:
    """Cross product of models and tasks; the raw material for a comparison table."""
    runs = []
    for task in tasks:
        resolved = task if isinstance(task, OpenQMLTask) else get_task(task)
        for model in models:
            run = run_model_on_task(model, resolved, **kwargs)
            if publish:
                try:
                    run.publish()
                except OpenQMLError:
                    pass  # already published; keep the in-memory result
            runs.append(run)
    return runs


def run_exists(run: OpenQMLRun) -> Optional[int]:
    """Return the id of a published run with the same hash, if there is one."""
    for payload in get_store().list("runs"):
        if payload.get("run_hash") == run.run_hash:
            return int(payload["id"])
    return None


def get_run(identifier) -> OpenQMLRun:
    return OpenQMLRun.from_dict(get_store().get("runs", identifier))


def list_runs(task_id: Optional[int] = None, flow_name: Optional[str] = None,
              tag: Optional[str] = None, output_format: str = "dataframe"):
    records = []
    for payload in get_store().list("runs"):
        if task_id is not None and payload.get("task_id") != task_id:
            continue
        if flow_name and payload.get("flow_name") != flow_name:
            continue
        if tag and tag not in payload.get("tags", []):
            continue
        measure = payload.get("evaluation_measure", "")
        records.append({
            "id": payload.get("id"),
            "task_id": payload.get("task_id"),
            "flow_name": payload.get("flow_name"),
            "measure": measure,
            "value": payload.get("evaluations", {}).get(measure),
            "backend": payload.get("backend"),
            "shots": payload.get("shots"),
            "runtime_seconds": payload.get("runtime_seconds"),
            "uploaded_at": payload.get("uploaded_at"),
        })
    return as_table(records, output_format, index="id")


def delete_run(identifier) -> None:
    get_store().delete("runs", identifier)
