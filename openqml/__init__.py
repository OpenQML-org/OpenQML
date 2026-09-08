"""OpenQML -- an open exchange for quantum machine learning experiments.

The layout follows OpenML deliberately: datasets, tasks, flows and runs are the
four entities, and the workflow is the same one people already know::

    import openqml

    task = openqml.tasks.get_task(1)
    model = openqml.models.QuantumKernelClassifier(feature_map="zz", n_qubits=2)
    run = openqml.runs.run_model_on_task(model, task)
    print(run.summary())
    run.publish()

What is added on top is the quantum metadata that makes two results actually
comparable: which encoding, how many qubits, which backend, analytic or shot
based -- and a classical control arm for every built-in model, because a
quantum number without a baseline beside it is not evidence of anything.

Out of the box everything runs against a local on-disk store, so no account and
no network are required. Point ``openqml.config.server`` at a deployment to
share results.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import config  # noqa: F401  (must come first: everything else reads it)
from . import (  # noqa: F401
    backends,
    catalog,
    circuits,
    datasets,
    evaluations,
    exceptions,
    flows,
    hamiltonians,
    metrics,
    models,
    runs,
    study,
    tasks,
    utils,
)
from .circuits import Circuit, w, x  # noqa: F401
from .datasets import OpenQMLDataset, create_dataset, get_dataset, list_datasets  # noqa: F401
from .evaluations import compare, leaderboard, list_evaluations  # noqa: F401
from .exceptions import OpenQMLError  # noqa: F401
from .flows import OpenQMLFlow, flow_to_model, model_to_flow  # noqa: F401
from .runs import OpenQMLRun, run_model_on_task  # noqa: F401
from .study import OpenQMLSuite, get_suite, list_suites  # noqa: F401
from .tasks import OpenQMLTask, create_task, get_task, list_tasks  # noqa: F401

__all__ = [
    "__version__", "config",
    "datasets", "tasks", "flows", "runs", "evaluations", "study",
    "circuits", "backends", "models", "metrics", "hamiltonians", "utils", "exceptions",
    "OpenQMLDataset", "OpenQMLTask", "OpenQMLFlow", "OpenQMLRun", "OpenQMLSuite",
    "list_datasets", "get_dataset", "create_dataset",
    "list_tasks", "get_task", "create_task",
    "model_to_flow", "flow_to_model",
    "run_model_on_task", "list_evaluations", "leaderboard", "compare",
    "get_suite", "list_suites",
    "Circuit", "w", "x", "OpenQMLError",
]


def reset_local_store() -> None:
    """Delete everything published locally and re-seed the bundled entities."""
    from ._store import get_store

    store = get_store()
    if hasattr(store, "reset"):
        store.reset()
    else:
        raise OpenQMLError("reset_local_store() only applies to the local store")
