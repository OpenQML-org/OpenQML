"""Task-level API."""

from __future__ import annotations

from typing import Optional, Sequence

from .._store import get_store
from ..utils import as_table
from .task import TASK_TYPES, OpenQMLTask, task_from_dict

__all__ = ["list_tasks", "get_task", "get_tasks", "create_task", "delete_task"]


def list_tasks(task_type: Optional[str] = None, dataset_id: Optional[int] = None,
               tag: Optional[str] = None, output_format: str = "dataframe"):
    """List tasks, optionally filtered by type, dataset or tag."""
    records = []
    for payload in get_store().list("tasks"):
        if task_type and payload.get("task_type") != task_type:
            continue
        if dataset_id and payload.get("dataset_id") != dataset_id:
            continue
        if tag and tag not in payload.get("tags", []):
            continue
        procedure = payload.get("estimation_procedure", {})
        records.append({
            "id": payload.get("id"),
            "name": payload.get("name"),
            "task_type": payload.get("task_type"),
            "dataset_id": payload.get("dataset_id"),
            "evaluation_measure": payload.get("evaluation_measure"),
            "procedure": procedure.get("type"),
            "folds": procedure.get("folds"),
            "qubits": payload.get("qubits"),
        })
    return as_table(records, output_format, index="id")


def get_task(identifier) -> OpenQMLTask:
    """Fetch one task by id or name."""
    return task_from_dict(get_store().get("tasks", identifier))


def get_tasks(identifiers: Sequence) -> list:
    return [get_task(i) for i in identifiers]


def create_task(task_type: str, dataset_id: int, name: Optional[str] = None,
                target_name: str = "class", evaluation_measure: str = "accuracy",
                estimation_procedure: Optional[dict] = None, description: str = "",
                qubits: Optional[int] = None, tags: Optional[Sequence[str]] = None) -> OpenQMLTask:
    """Define a new task. Call ``.publish()`` to register it."""
    if task_type not in TASK_TYPES:
        raise ValueError(f"task_type must be one of {TASK_TYPES}")
    return task_from_dict({
        "task_type": task_type,
        "dataset_id": int(dataset_id),
        "name": name or f"dataset {dataset_id} / {task_type}",
        "target_name": target_name,
        "evaluation_measure": evaluation_measure,
        "estimation_procedure": estimation_procedure,
        "description": description,
        "qubits": qubits,
        "tags": list(tags or []),
    })


def delete_task(identifier) -> None:
    get_store().delete("tasks", identifier)
