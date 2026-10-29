"""Reproducible, restricted-family scientific scale checks in fresh CPU workers.

Large cases use exact Pauli expectation propagation, not dense state simulation.
Each family is cross-checked against dense evolution at a small size and against
analytic expectations at its measured size. An IR-counting case performs no
quantum simulation. A growth-limit case demonstrates explicit bounded rejection.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from _memory import measure_peak
from planned_execution import source_hash


def workload(oq, case, width):
    theta = oq.Parameter("theta")
    builder = oq.CircuitBuilder(width)
    if case == "ghz":
        builder.h(0)
        for q in range(1, width):
            builder.cx(q - 1, q)
        labels, expected = ["X" * width, "Z" + "I" * (width - 2) + "Z"], ["cos", 1]
        structure, cut_rank = "GHZ_chain", 1
    elif case == "bell":
        if width % 2:
            raise ValueError("Crossed Bell pairs require an even width.")
        half = width // 2
        for q in range(half):
            builder.h(q).cx(q, q + half)
        labels = ["X" * width, "Z" * width]
        expected, structure, cut_rank = ["cos", 1], "product_of_crossed_Bell_pairs", half
    else:
        side = math.isqrt(width)
        if side**2 != width:
            raise ValueError("Cluster width must be a square.")
        for q in range(width):
            builder.h(q)
        for row in range(side):
            for col in range(side):
                q = row * side + col
                if col + 1 < side:
                    builder.cz(q, q + 1)
                if row + 1 < side:
                    builder.cz(q, q + side)
        labels, expected = [], []
        for q in sorted({0, width // 2, width - 1}):
            label, row, col = ["I"] * width, q // side, q % side
            label[q] = "X"
            for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= r < side and 0 <= c < side:
                    label[r * side + c] = "Z"
            labels.append("".join(label))
            expected.append("cos" if q == 0 else 1)
        # Products of one checkerboard's stabilizers have no internal graph
        # edges. They retain a positive sign and probe the full grid collectively.
        for color in (0, 1):
            label = []
            for q in range(width):
                row, col = divmod(q, side)
                if (row + col) % 2 == color:
                    label.append("X")
                else:
                    degree = sum(
                        0 <= r < side and 0 <= c < side
                        for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
                    )
                    label.append("Z" if degree % 2 else "I")
            labels.append("".join(label))
            expected.append("cos" if color == 0 else 1)
        structure, cut_rank = "connected_square_cluster_graph", side if side % 2 == 0 else None
    builder.rz(theta, 0)
    return builder.build(), tuple(oq.PauliString(p) for p in labels), expected, structure, cut_rank


def worker(args):
    import numpy as np

    import openqml as oq

    assert Path(oq.__file__).resolve().is_relative_to(args.source.resolve())
    if args.case == "ir":
        builder = oq.CircuitBuilder(8)
        for i in range(args.operations):
            builder.h(i % 8)
        circuit = builder.build()
        started = time.perf_counter()
        report, memory = measure_peak(
            lambda: oq.estimate_resources(circuit, max_operations=args.operations)
        )
        return {
            "case": "ir",
            "work_performed": "IR_counting_and_storage_estimation_only",
            "quantum_simulation_performed": False,
            "seconds": time.perf_counter() - started,
            "resources": report.to_dict(),
            "memory": memory,
        }
    if args.case == "guard":
        x, builder = oq.Parameter("x"), oq.CircuitBuilder(12)
        for _ in range(3):
            for q in range(12):
                builder.ry((q + 1) * x, q)
            for q in range(11):
                builder.cz(q, q + 1)
        try:
            oq.PauliPlan(builder.build(), (oq.PauliString("Z" * 12),), max_terms=8, max_work=5000)
        except oq.ConversionError as error:
            return {
                "case": "guard",
                "representation": "exact_Pauli_propagation",
                "status": "rejected_by_explicit_budget",
                "codes": [d.code for d in error.diagnostics],
            }
        raise AssertionError("The growth guard was not exercised.")

    small_width = 4 if args.case == "cluster" else 6
    small, small_obs, _, _, _ = workload(oq, args.case, small_width)
    small_plan = oq.PauliPlan(small, small_obs)
    small_error = 0.0
    for angle in (0.0, 0.37, -0.6):
        actual = small_plan.evaluate({"theta": angle})
        expected = oq.expectations(small, small_obs, {"theta": angle})
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        small_error = max(small_error, float(np.max(np.abs(actual - expected))))
    circuit, observables, reference, structure, cut_rank = workload(oq, args.case, args.qubits)
    started = time.perf_counter()
    plan, memory = measure_peak(lambda: oq.PauliPlan(circuit, observables))
    prepared = time.perf_counter() - started
    angles = np.array([0.0, 0.37, -0.6, 1.1])
    expected = np.stack(
        [np.cos(angles) if v == "cos" else np.full_like(angles, v) for v in reference], axis=-1
    )
    actual = plan.evaluate({"theta": angles})
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    times = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        output = plan.evaluate({"theta": angles})
        times.append(time.perf_counter() - start)
        np.testing.assert_allclose(output, expected, atol=1e-12, rtol=1e-12)
    return {
        "case": args.case,
        "qubits": args.qubits,
        "structure": structure,
        "representation": "exact_declared_Pauli_expectations",
        "generic_dense_simulation": False,
        "analytic_log2_Schmidt_rank_at_middle_cut": cut_rank,
        "small_dense_reference_qubits": small_width,
        "small_dense_max_error": small_error,
        "analytic_binding_count": len(angles),
        "analytic_max_error": float(np.max(np.abs(actual - expected))),
        "prepare_seconds": prepared,
        "warm_median_seconds": statistics.median(times),
        "warm_samples_seconds": times,
        "prepare_memory": memory,
        "plan": plan.to_dict(),
        "dense_storage_estimate": oq.estimate_resources(circuit).to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qubits", type=int, default=1024)
    parser.add_argument("--cluster-side", type=int, default=16)
    parser.add_argument("--operations", type=int, default=100_000)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=("ghz", "bell", "cluster", "ir", "guard"))
    args = parser.parse_args()
    if (
        args.qubits < 2
        or args.qubits % 2
        or not 2 <= args.cluster_side <= 32
        or min(args.operations, args.repetitions) < 1
    ):
        parser.error(
            "Use an even qubit count, cluster side between two and thirty-two, and positive limits."
        )
    if args.worker:
        sys.path.insert(0, str(args.source.resolve()))
        print(json.dumps(worker(args), allow_nan=False))
        return
    runs = []
    for case in (args.case,) if args.case else ("ghz", "bell", "cluster", "ir", "guard"):
        width = args.cluster_side**2 if case == "cluster" else args.qubits
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--source",
            str(args.source.resolve()),
            "--case",
            case,
            "--qubits",
            str(width),
            "--operations",
            str(args.operations),
            "--repetitions",
            str(args.repetitions),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            env=os.environ | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
        runs.append(json.loads(result.stdout))
    payload = {
        "scope": "restricted_families_not_general_quantum_scale",
        "cpu_only": True,
        "source_sha256": source_hash(args.source),
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "numpy": importlib.metadata.version("numpy"),
            "worker_threads": {"OMP_NUM_THREADS": 1, "OPENBLAS_NUM_THREADS": 1},
        },
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runs": runs,
    }
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
