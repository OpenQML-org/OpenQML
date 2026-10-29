"""Compare adjoint and parameter-shift VJPs on the same NumPy workload.

Reports warm CPU timings and equivalent state-vector gate applications. Run
with --output /path/result.json to save results; no network or hardware is used.
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

from openqml import Parameter, adjoint_vjp, data_reuploading, parameter_shift_vjp


def benchmark(qubits, depth, batch, repeats):
    import openqml.adjoint as adjoint
    import openqml.simulator as simulator

    spec = data_reuploading(qubits, depth)
    circuit = spec.circuit.with_global_phase(Parameter("phase"))
    rng = np.random.default_rng(47)
    rows = [
        dict(zip(circuit.parameters, row, strict=True))
        for row in rng.uniform(-0.4, 0.4, (batch, len(circuit.parameters)))
    ]
    cotangents = rng.normal(size=(batch, len(spec.observables)))

    def shifted():
        return parameter_shift_vjp(
            circuit, spec.observables, rows, cotangents, chunk_size=32, max_memory_mb=128
        )

    def reverse():
        return adjoint_vjp(circuit, spec.observables, rows, cotangents, max_memory_mb=128)

    reference, value = shifted(), reverse()
    np.testing.assert_allclose(value, reference, atol=2e-11, rtol=2e-11)
    results = {}
    apply = simulator._apply_matrix
    for name, run in (("parameter_shift", shifted), ("adjoint", reverse)):
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            run()
            times.append(time.perf_counter() - start)
        gate_applications = 0

        def counted(state, *args, **kwargs):
            nonlocal gate_applications
            gate_applications += math.prod(state.shape[:-1])
            return apply(state, *args, **kwargs)

        with (
            patch.object(simulator, "_apply_matrix", counted),
            patch.object(adjoint, "_apply_matrix", counted),
        ):
            run()
        results[name] = {
            "median_seconds": statistics.median(times),
            "statevector_gate_applications": gate_applications,
        }
    return {
        "qubits": qubits,
        "depth": depth,
        "batch": batch,
        "parameters": len(circuit.parameters),
        "observables": len(spec.observables),
        "repeats": repeats,
        "max_abs_gradient_error": float(np.max(np.abs(value - reference))),
        "results": results,
        "cpu_speedup": results["parameter_shift"]["median_seconds"]
        / results["adjoint"]["median_seconds"],
        "gate_application_reduction": results["parameter_shift"]["statevector_gate_applications"]
        / results["adjoint"]["statevector_gate_applications"],
        "note": "Local warm NumPy microbenchmark; timings depend on the workload and machine. "
        "This does not measure hardware gradients, GPU speed or quantum advantage. "
        "Symbolic metadata and result storage can grow; retained quantum states "
        "do not grow with gate count.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.qubits, args.depth, args.batch, args.repeats) < 1:
        parser.error("qubits, depth, batch and repeats must be positive.")
    result = benchmark(args.qubits, args.depth, args.batch, args.repeats)
    text = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="")
