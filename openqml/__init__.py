"""OpenQML -- an open exchange for quantum machine learning experiments.

Datasets, tasks, flows and runs, laid out the way OpenML lays them out, plus
the quantum metadata that decides whether two results are comparable at all --
encoding, qubit count, backend, analytic or shot based -- and a classical
control arm for every built-in model.

Everything a normal session needs is on the top-level namespace::

    from openqml import QuantumKernelClassifier, get_task, run_model_on_task

    run = run_model_on_task(QuantumKernelClassifier(feature_map="iqp"), get_task(1))
    print(run.summary())

The submodules (``openqml.models``, ``openqml.circuits``, ``openqml.backends``
...) stay available for the rest. No account and no network are required: out
of the box everything runs against a local on-disk store. Point
``openqml.config.server`` at a deployment to share results.
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

# -- entities: the four nouns, their constructors and their lookups -----------
from .datasets import (  # noqa: F401
    OpenQMLDataset,
    create_dataset,
    delete_dataset,
    get_dataset,
    get_datasets,
    list_datasets,
)
from .tasks import (  # noqa: F401
    GroundStateEstimationTask,
    OpenQMLTask,
    Split,
    StateClassificationTask,
    SupervisedClassificationTask,
    SupervisedRegressionTask,
    create_task,
    delete_task,
    get_task,
    get_tasks,
    list_tasks,
    make_split,
)
from .flows import (  # noqa: F401
    OpenQMLFlow,
    flow_to_model,
    get_flow,
    list_flows,
    model_to_flow,
)
from .runs import (  # noqa: F401
    OpenQMLRun,
    delete_run,
    get_run,
    list_runs,
    run_flow_on_task,
    run_model_on_task,
    run_models_on_tasks,
)
from .evaluations import compare, leaderboard, list_evaluations  # noqa: F401
from .study import (  # noqa: F401
    OpenQMLSuite,
    create_suite,
    delete_suite,
    get_suite,
    list_suites,
)

# -- models: quantum estimators and the classical arms they are judged against
from .models import (  # noqa: F401
    VQE,
    ClassicalKernelClassifier,
    ClassicalKernelRegressor,
    ExactDiagonalisation,
    LinearRegressor,
    MajorityClassifier,
    QuantumKernelClassifier,
    QuantumKernelRegressor,
    QuantumModel,
    VariationalQuantumClassifier,
    VariationalQuantumRegressor,
    clear_state_cache,
    list_models,
)

# -- circuits: the IR, its angle references, and the usual templates ----------
from .circuits import (  # noqa: F401
    Circuit,
    Gate,
    Ref,
    amplitude_embedding,
    angle_embedding,
    data_reuploading,
    get_ansatz,
    get_feature_map,
    hardware_efficient_ansatz,
    iqp_feature_map,
    real_amplitudes_ansatz,
    strongly_entangling_ansatz,
    w,
    x,
    zz_feature_map,
)

# -- execution and scoring ----------------------------------------------------
from .backends import (  # noqa: F401
    get_backend,
    list_backends,
    register_backend,
    simulate,
)
from .metrics import get_measure, list_measures  # noqa: F401
from .exceptions import OpenQMLError  # noqa: F401

__all__ = [
    "__version__", "config", "reset_local_store",
    # submodules
    "datasets", "tasks", "flows", "runs", "evaluations", "study",
    "circuits", "backends", "models", "metrics", "hamiltonians", "utils", "exceptions",
    "catalog",
    # entities
    "OpenQMLDataset", "OpenQMLTask", "OpenQMLFlow", "OpenQMLRun", "OpenQMLSuite",
    "SupervisedClassificationTask", "SupervisedRegressionTask",
    "StateClassificationTask", "GroundStateEstimationTask",
    "list_datasets", "get_dataset", "get_datasets", "create_dataset", "delete_dataset",
    "list_tasks", "get_task", "get_tasks", "create_task", "delete_task",
    "Split", "make_split",
    "model_to_flow", "flow_to_model", "get_flow", "list_flows",
    "run_model_on_task", "run_flow_on_task", "run_models_on_tasks",
    "get_run", "list_runs", "delete_run",
    "list_evaluations", "leaderboard", "compare",
    "get_suite", "list_suites", "create_suite", "delete_suite",
    # models
    "QuantumModel", "list_models",
    "QuantumKernelClassifier", "QuantumKernelRegressor",
    "ClassicalKernelClassifier", "ClassicalKernelRegressor",
    "VariationalQuantumClassifier", "VariationalQuantumRegressor",
    "VQE", "ExactDiagonalisation", "MajorityClassifier", "LinearRegressor",
    "clear_state_cache",
    # circuits
    "Circuit", "Gate", "Ref", "w", "x",
    "angle_embedding", "amplitude_embedding", "zz_feature_map", "iqp_feature_map",
    "data_reuploading", "hardware_efficient_ansatz", "strongly_entangling_ansatz",
    "real_amplitudes_ansatz", "get_feature_map", "get_ansatz",
    # execution and scoring
    "get_backend", "list_backends", "register_backend", "simulate",
    "get_measure", "list_measures",
    "OpenQMLError",
]


def reset_local_store() -> None:
    """Delete everything published locally and re-seed the bundled entities."""
    from ._store import get_store

    store = get_store()
    if hasattr(store, "reset"):
        store.reset()
    else:
        raise OpenQMLError("reset_local_store() only applies to the local store")
