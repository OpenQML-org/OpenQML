"""Leaderboards and model comparisons built from published runs."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .._store import get_store
from ..metrics import higher_is_better
from ..runs import run_model_on_task
from ..tasks import get_task
from ..utils import as_table, get_params, jsonify

__all__ = ["list_evaluations", "leaderboard", "compare"]


def list_evaluations(function: str = "accuracy", tasks: Optional[Sequence[int]] = None,
                     flows: Optional[Sequence[str]] = None, uploaders=None,
                     output_format: str = "dataframe", sort: bool = True):
    """Every published run that reports ``function``, newest schema first.

    Mirrors ``openml.evaluations.list_evaluations``; the extra columns
    (``backend``, ``shots``) are the ones that decide whether two quantum
    numbers are comparable at all.
    """
    records = []
    for payload in get_store().list("runs"):
        value = (payload.get("evaluations") or {}).get(function)
        if value is None:
            continue
        if tasks and payload.get("task_id") not in set(tasks):
            continue
        if flows and payload.get("flow_name") not in set(flows):
            continue
        per_fold = (payload.get("fold_evaluations") or {}).get(function, {})
        spread = [v for folds in per_fold.values() for v in folds.values()]
        records.append({
            "run_id": payload.get("id"),
            "task_id": payload.get("task_id"),
            "flow_name": payload.get("flow_name"),
            "function": function,
            "value": float(value),
            "std": float(np.std(spread)) if spread else 0.0,
            "folds": max((len(folds) for folds in per_fold.values()), default=0),
            "repeats": len(per_fold),
            "backend": payload.get("backend"),
            "shots": payload.get("shots"),
            "runtime_seconds": payload.get("runtime_seconds"),
            "uploaded_at": payload.get("uploaded_at"),
        })
    if sort and records:
        records.sort(key=lambda r: r["value"], reverse=higher_is_better(function))
    return as_table(records, output_format)


def leaderboard(task_id: int, measure: Optional[str] = None, top_k: Optional[int] = None,
                output_format: str = "dataframe"):
    """Ranked results for one task, using that task's own measure by default."""
    measure = measure or get_task(task_id).evaluation_measure
    table = list_evaluations(measure, tasks=[task_id], output_format="records")
    if top_k:
        table = table[:top_k]
    return as_table(table, output_format)


def _labels(models) -> List[str]:
    """Label each model by class, plus the parameters that actually differ.

    Two configurations of the same class have to end up on separate rows, or a
    comparison table silently reports one of them twice.
    """
    names = [type(model).__name__ for model in models]
    settings = [get_params(model) for model in models]
    labels = []
    for index, name in enumerate(names):
        siblings = [j for j, other in enumerate(names) if other == name]
        distinguishing = {}
        if len(siblings) > 1:
            for key in settings[index]:
                values = {jsonify(settings[j].get(key)) for j in siblings}
                if len(values) > 1:
                    distinguishing[key] = settings[index][key]
        suffix = ", ".join(f"{k}={v}" for k, v in sorted(distinguishing.items()))
        labels.append(f"{name}[{suffix}]" if suffix else name)
    return labels


def compare(models, tasks, publish: bool = False, measure: Optional[str] = None,
            output_format: str = "dataframe", **kwargs):
    """Run several models over several tasks and tabulate model x task.

    This is the function the rest of the library exists to make easy. Report
    the whole table, including the columns where the classical baseline wins --
    a benchmark that only shows its wins is advertising.
    """
    from ..exceptions import DuplicateRunError

    tasks = [t if hasattr(t, "task_type") else get_task(t) for t in tasks]
    labels = _labels(models)
    rows = [{"model": label} for label in labels]
    for task in tasks:
        for index, model in enumerate(models):
            run = run_model_on_task(model, task, **kwargs)
            if publish:
                try:
                    run.publish()
                except DuplicateRunError:
                    pass  # an identical run is already on the board; keep ours in memory
            values = run.get_metric_fn(measure or run.evaluation_measure)
            rows[index][task.name] = f"{values.mean():.4f} +/- {values.std():.4f}"
    return as_table(rows, output_format, index="model")
