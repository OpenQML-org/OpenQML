"""Compare cold bound-circuit submission with a warm compiled parameter batch."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from openqml import AerBackend, __version__, expectations_batch, hardware_efficient, z_observables

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="benchmarks/results_aer.json")
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--repeats", type=int, default=5)
args = parser.parse_args()
spec = hardware_efficient(4, 2)
rng = np.random.default_rng(19)
rows = [
    dict(zip(spec.circuit.parameters, rng.normal(size=len(spec.circuit.parameters)), strict=True))
    for _ in range(args.batch)
]
observables = z_observables(4)
plain = AerBackend(name="aer-uncompiled", parameter_batching=False)
cached = AerBackend(name="aer-compiled")


def evaluate(backend):
    return expectations_batch(spec.circuit, observables, rows, backend=backend)


def timing(backend):
    evaluate(backend)
    durations = []
    for _ in range(args.repeats):
        start = perf_counter()
        evaluate(backend)
        durations.append(perf_counter() - start)
    return median(durations)


reference, actual = evaluate(plain), evaluate(cached)
np.testing.assert_allclose(actual, reference, atol=1e-12)
cold, warm = timing(plain), timing(cached)

result = {
    "openqml": __version__,
    "python": platform.python_version(),
    "qiskit": version("qiskit"),
    "aer": version("qiskit-aer"),
    "device": "CPU",
    "max_parallel_threads": 1,
    "batch": args.batch,
    "qubits": 4,
    "depth": 2,
    "repeats": args.repeats,
    "statistic": "median",
    "bound_circuit_batch_ms": cold * 1000,
    "warm_parameter_batch_ms": warm * 1000,
    "speedup": cold / warm,
    "max_output_error": float(np.max(np.abs(actual - reference))),
    "cache": cached.cache_info(),
    "warmup_excluded": True,
}
Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
