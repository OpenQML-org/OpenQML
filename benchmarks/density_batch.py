"""Compare noisy/adaptive NumPy batches with scalar calls to the same kernels."""

import argparse
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from openqml import (
    Circuit,
    ClassicalCondition,
    Parameter,
    __version__,
    expectations,
    expectations_batch,
    parameter_shift_jacobian,
    z_observables,
)

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="benchmarks/results_density.json")
parser.add_argument("--repeats", type=int, default=5)
args = parser.parse_args()


def timing(function):
    function()
    values = []
    for _ in range(args.repeats):
        start = perf_counter()
        function()
        values.append(perf_counter() - start)
    return median(values) * 1000


results = []
for qubits in (2, 4):
    x, p = Parameter("x"), Parameter("p")
    circuit = (
        Circuit(qubits)
        .ry(x, 0)
        .measure(0, 0)
        .if_else(
            ClassicalCondition((0,), 1),
            Circuit(qubits).ry(x * x, qubits - 1),
        )
    )
    for q in range(qubits):
        circuit = circuit.ry(x / (q + 1), q).amplitude_damping(p, q)
    observables = z_observables(qubits)
    for batch in (8, 32):
        rows = [{"x": float(xv), "p": 0.23} for xv in np.linspace(-0.7, 0.8, batch)]

        def scalar(circuit=circuit, observables=observables, rows=rows):
            return np.array([expectations(circuit, observables, row) for row in rows])

        def vector(circuit=circuit, observables=observables, rows=rows):
            return expectations_batch(circuit, observables, rows)

        reference, actual = scalar(), vector()
        np.testing.assert_allclose(actual, reference, atol=1e-12)
        # Noise probabilities and angles share the streamed gradient path.
        jacobian = parameter_shift_jacobian(circuit, observables, rows)
        assert np.isfinite(jacobian).all()
        slow, fast = timing(scalar), timing(vector)
        results.append(
            {
                "qubits": qubits,
                "batch": batch,
                "classical_branches": 2,
                "observables": len(observables),
                "scalar_forward_ms": slow,
                "batch_forward_ms": fast,
                "ratio": slow / fast,
                "max_output_error": float(np.max(np.abs(actual - reference))),
            }
        )
result = {
    "openqml": __version__,
    "python": platform.python_version(),
    "numpy": np.__version__,
    "device": "cpu",
    "dtype": "complex128",
    "repeats": args.repeats,
    "statistic": "median",
    "warmup_excluded": True,
    "baseline": "scalar calls to the same current density/feedback kernels",
    "cases": results,
}
Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
