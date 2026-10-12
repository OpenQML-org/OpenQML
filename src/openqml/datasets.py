"""Offline, seeded toy data and train/test splits; no network request on import."""

import hashlib
from dataclasses import dataclass, field

import numpy as np

from ._validation import integer, real
from .models import _features, _labels, _targets


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    name: str
    task_type: str
    description: str
    metadata: dict = field(default_factory=dict)

    @property
    def fingerprint(self):
        digest = hashlib.sha256()
        for values in (self.X, self.y):
            digest.update(str((values.shape, values.dtype.str)).encode())
            digest.update(np.ascontiguousarray(values).tobytes())
        return digest.hexdigest()

    def split(self, *, test_size=0.25, seed=1):
        return train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            seed=seed,
            stratify=self.task_type == "classification",
        )


DATASETS = {
    "xor": ("classification", "Four noisy XOR clusters; a nonlinear toy correctness task."),
    "circles": ("classification", "Concentric circles; radial separation is classically easy."),
    "moons": ("classification", "Two noisy interleaving semicircles."),
    "parity": ("classification", "Uniform bitstrings labeled by global parity; a promise toy."),
    "sine": ("regression", "A seeded sin(frequency*x) target on [-pi, pi]."),
    "step": ("regression", "A discontinuous +/-1 target on [-pi, pi]."),
    "linear": ("regression", "A noisy linear target on [-pi, pi]."),
}


def list_datasets():
    return [
        {
            "id": name,
            "task_type": kind,
            "description": description,
            "source": "openqml synthetic v1",
            "offline": True,
        }
        for name, (kind, description) in DATASETS.items()
    ]


def load_dataset(name="moons", *, n_samples=192, seed=1, noise=0.08, frequency=3, n_features=4):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {', '.join(DATASETS)}")
    n_samples = integer(n_samples, "n_samples", 8, 100_000)
    noise = real(noise, "noise")
    frequency = real(frequency, "frequency")
    if noise < 0:
        raise ValueError("noise must be nonnegative")
    rng = np.random.default_rng(seed)
    if name in ("xor", "circles", "moons"):
        y = np.arange(n_samples) % 2
        if name == "xor":
            first = rng.integers(0, 2, size=n_samples)
            second = first ^ y
            features = np.column_stack([2 * first - 1, 2 * second - 1]).astype(float)
        else:
            angles = rng.uniform(0, 2 * np.pi if name == "circles" else np.pi, n_samples)
            if name == "circles":
                radius = np.where(y == 0, 1.0, 0.45)
                features = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
            else:
                features = np.c_[np.cos(angles), np.sin(angles)]
                features[y == 1] = np.c_[1 - np.cos(angles[y == 1]), 0.5 - np.sin(angles[y == 1])]
        features += rng.normal(0, noise, features.shape)
    elif name == "parity":
        n_features = integer(n_features, "n_features", 1, 8)
        # Guaranteed coverage of both classes before the final shuffle.
        bits = np.resize(np.arange(1 << n_features), n_samples)
        rng.shuffle(bits)
        features = ((bits[:, None] >> np.arange(n_features)) & 1).astype(float)
        y = (features.sum(axis=1) % 2).astype(int)
    else:
        features = rng.uniform(-np.pi, np.pi, (n_samples, 1))
        x = features[:, 0]
        y = (
            np.sin(frequency * x)
            if name == "sine"
            else (np.where(x >= 0, 1.0, -1.0) if name == "step" else x / np.pi)
        )
        y += rng.normal(0, noise, n_samples)
    order = rng.permutation(n_samples)
    kind, description = DATASETS[name]
    metadata = {
        "generator": "openqml-synthetic-v1",
        "seed": seed,
        "n_samples": n_samples,
        "noise": noise,
    }
    if name == "sine":
        metadata["frequency"] = frequency
    if name == "parity":
        metadata["n_features"] = n_features
    return Dataset(features[order].copy(), y[order].copy(), name, kind, description, metadata)


def train_test_split(features, targets, *, test_size=0.25, seed=1, stratify=False):
    """Return X_train, X_test, y_train, y_test; inputs are never mutated.

    Stratification allocates a rounded test fraction per class, keeping at least
    one item on each side. Thus the total may differ slightly from n*test_size.
    """
    features = _features(features)
    labels = (
        _labels(targets, len(features)) if stratify else _targets_or_labels(targets, len(features))
    )
    test_size = real(test_size, "test_size")
    if not 0 < test_size < 1 or len(features) < 2:
        raise ValueError("split needs at least two samples and 0 < test_size < 1")
    rng = np.random.default_rng(seed)
    if stratify:
        train, test = [], []
        for label in np.unique(labels):
            indices = rng.permutation(np.flatnonzero(labels == label))
            if indices.size < 2:
                raise ValueError("stratified splitting needs at least two samples per class")
            count = max(1, min(len(indices) - 1, int(round(len(indices) * test_size))))
            test.extend(indices[:count])
            train.extend(indices[count:])
        train, test = rng.permutation(train), rng.permutation(test)
    else:
        indices = rng.permutation(len(features))
        count = max(1, min(len(indices) - 1, int(round(len(indices) * test_size))))
        test, train = indices[:count], indices[count:]
    return features[train].copy(), features[test].copy(), labels[train].copy(), labels[test].copy()


def _targets_or_labels(values, count):
    values = np.asarray(values)
    return _labels(values, count) if values.dtype.kind in "US" else _targets(values, count)
