"""Local OpenML-style tasks/suites and traceable quantum/classical comparisons.

These are SDK protocols, not a reproduction of the website's JavaScript duel.
Splits, data hashes, configurations, versions, timings, and metrics travel with
every result. No advantage or parameter matching is assumed.
"""

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from ._validation import integer
from .datasets import Dataset, load_dataset
from .models import (
    FourierRegressor,
    LogisticClassifier,
    QuantumKernelClassifier,
    RidgeRegressor,
    VariationalQuantumClassifier,
    VariationalQuantumRegressor,
    _features,
    _labels,
    _targets,
)


@dataclass
class Task:
    id: str
    dataset: Dataset
    test_size: float = 0.25

    @property
    def task_type(self):
        return self.dataset.task_type

    def split(self, *, seed=1):
        return self.dataset.split(test_size=self.test_size, seed=seed)


def make_task(features, targets, *, task_type="classification", name="custom", test_size=0.25):
    """Register an in-memory local task for the user's own data."""
    if task_type not in ("classification", "regression"):
        raise ValueError("task_type must be 'classification' or 'regression'")
    features = _features(features)
    targets = (
        _labels(targets, len(features))
        if task_type == "classification"
        else _targets(targets, len(features))
    )
    return Task(
        name, Dataset(features, targets, name, task_type, "User-provided local dataset"), test_size
    )


def get_task(name="moons", *, n_samples=96, seed=1, test_size=0.25, **dataset_options):
    dataset = load_dataset(name, n_samples=n_samples, seed=seed, **dataset_options)
    return Task(f"qml-{dataset.task_type}-{name}-v1", dataset, test_size)


@dataclass(frozen=True)
class Suite:
    id: str
    task_ids: tuple[str, ...]
    description: str

    def __iter__(self):
        return iter(self.task_ids)

    def __len__(self):
        return len(self.task_ids)


def get_suite(name="qml-cls-1"):
    suites = {
        "qml-cls-1": Suite(
            "qml-cls-1",
            ("xor", "circles", "moons"),
            "Synthetic binary classification correctness suite",
        ),
        "qml-reg-1": Suite(
            "qml-reg-1",
            ("sine", "step", "linear"),
            "Periodic, discontinuous, and linear regression toy suite",
        ),
    }
    if name not in suites:
        raise ValueError(f"unknown suite {name!r}; choose from {', '.join(suites)}")
    return suites[name]


MODELS = {
    "quantum-kernel": (QuantumKernelClassifier, "classification", True),
    "vqc": (VariationalQuantumClassifier, "classification", True),
    "logistic": (LogisticClassifier, "classification", False),
    "vqr": (VariationalQuantumRegressor, "regression", True),
    "ridge": (RidgeRegressor, "regression", False),
    "fourier": (FourierRegressor, "regression", False),
}


def list_models():
    return [
        {"id": name, "class": cls.__name__, "task_type": kind, "quantum": quantum}
        for name, (cls, kind, quantum) in MODELS.items()
    ]


def get_model(name, **parameters):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODELS)}")
    return MODELS[name][0](**parameters)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


@dataclass
class BenchmarkResult:
    protocol: str
    task_id: str
    model: str
    seed: int
    metrics: dict
    model_parameters: dict
    resources: dict
    provenance: dict

    def to_dict(self):
        return json.loads(json.dumps(asdict(self), default=_json_default, allow_nan=False))

    def save(self, path):
        path = Path(path)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return path


def run_benchmark(model, task="moons", *, seed=1):
    """Fit the supplied estimator on one seeded split; report held-out metrics."""
    seed = integer(seed, "seed", 0)
    task = get_task(task) if isinstance(task, str) else task
    if not isinstance(task, Task):
        raise ValueError("task must be a Task or built-in dataset id")
    if isinstance(model, str):
        if model not in MODELS or MODELS[model][1] != task.task_type:
            raise ValueError("model id must be known and compatible with the task")
        model = get_model(model, **({"seed": seed} if model in ("vqc", "vqr", "logistic") else {}))
    if not callable(getattr(model, "fit", None)) or not callable(getattr(model, "predict", None)):
        raise ValueError("model must provide fit() and predict()")
    known = next((kind for cls, kind, _ in MODELS.values() if type(model) is cls), None)
    quantum = any(type(model) is cls and is_quantum for cls, _, is_quantum in MODELS.values())
    if known is not None and known != task.task_type:
        raise ValueError("model task type does not match the benchmark")
    X_train, X_test, y_train, y_test = task.split(seed=seed)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = np.asarray(model.predict(X_test))
    predict_seconds = time.perf_counter() - started
    if prediction.shape != y_test.shape:
        raise ValueError("model predictions must have one value per test item")
    if task.task_type == "classification":
        labels = np.unique(y_test)
        metrics = {
            "accuracy": float(np.mean(prediction == y_test)),
            "balanced_accuracy": float(
                np.mean([np.mean(prediction[y_test == label] == label) for label in labels])
            ),
        }
    else:
        prediction = _targets(prediction, len(y_test))
        residual = prediction - y_test
        denominator = np.sum((y_test - y_test.mean()) ** 2)
        metrics = {
            "mse": float(np.mean(residual**2)),
            "mae": float(np.mean(np.abs(residual))),
            "r2": float(1 - np.sum(residual**2) / denominator)
            if denominator > 0
            else (1.0 if np.allclose(prediction, y_test) else 0.0),
        }
    try:
        package_version = version("openqml")
    except PackageNotFoundError:
        from . import __version__

        package_version = __version__
    parameters = model.get_params() if hasattr(model, "get_params") else {}
    resources = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": X_train.shape[1],
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "parameter_count": getattr(model, "n_parameters_", None),
        "circuit_evaluations": getattr(model, "n_circuit_evaluations_", 0),
    }
    provenance = {
        "dataset": task.dataset.name,
        "dataset_sha256": task.dataset.fingerprint,
        "dataset_options": task.dataset.metadata,
        "test_size": task.test_size,
        "split_seed": seed,
        "openqml": package_version,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "simulation": "exact NumPy statevector" if quantum else None,
        "hardware_execution": False if known is not None else None,
        "parameter_matched": False,
    }
    return BenchmarkResult(
        "openqml-sdk-bench-v1",
        task.id,
        type(model).__name__,
        seed,
        metrics,
        parameters,
        resources,
        provenance,
    )


@dataclass
class ComparisonResult:
    task_id: str
    results: list[BenchmarkResult]

    def summary(self):
        groups = {}
        for result in self.results:
            groups.setdefault(result.model, []).append(result)
        return {
            name: {
                "runs": len(items),
                "seeds": [r.seed for r in items],
                "median_metrics": {
                    metric: float(np.median([r.metrics[metric] for r in items]))
                    for metric in items[0].metrics
                },
            }
            for name, items in groups.items()
        }

    def to_dict(self):
        return {
            "protocol": "openqml-sdk-bench-v1",
            "task_id": self.task_id,
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path):
        path = Path(path)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return path


def compare_models(task="moons", *, models=None, seeds=(1, 2, 3), epochs=40):
    """Compare freshly constructed models on identical splits for each seed.

    models may map display names to zero-argument factories. Factories must return
    independent unfitted estimators. Built-in defaults include classical baselines.
    """
    task = get_task(task) if isinstance(task, str) else task
    if not isinstance(task, Task):
        raise ValueError("task must be a Task or built-in dataset id")
    seeds = tuple(integer(seed, "seed", 0) for seed in seeds)
    epochs = integer(epochs, "epochs", 1)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and distinct")
    if models is not None and (
        not models or any(not callable(factory) for factory in models.values())
    ):
        raise ValueError("models must map names to zero-argument estimator factories")
    results, seen = [], []
    for seed in seeds:
        factories = models
        if factories is None:
            factories = (
                {
                    "quantum-kernel": lambda: QuantumKernelClassifier(),
                    "vqc": lambda: VariationalQuantumClassifier(epochs=epochs, seed=seed),
                    "logistic": lambda: LogisticClassifier(seed=seed),
                }
                if task.task_type == "classification"
                else {
                    "vqr": lambda: VariationalQuantumRegressor(epochs=epochs, seed=seed),
                    "ridge": lambda: RidgeRegressor(),
                    "fourier": lambda: FourierRegressor(),
                }
            )
        for name, factory in factories.items():
            estimator = factory()
            if any(estimator is previous for previous in seen):
                raise ValueError("each model factory must return a fresh estimator")
            seen.append(estimator)
            result = run_benchmark(estimator, task, seed=seed)
            # Keep factory names so differently configured models can be distinguished.
            result.model = str(name)
            results.append(result)
    return ComparisonResult(task.id, results)
