"""Fresh-process JVP vs Jacobian contraction; optional previous source baseline.

Run with .venv/bin/python benchmarks/directional.py --baseline-source /.../src
No network access, provider sessions or timing assertions are used.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path


def source_hash(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def worker(source, mode, workload):
    import numpy as np

    import openqml as oq

    actual = Path(oq.__file__).resolve()
    assert actual.is_relative_to(source.resolve()), actual
    c = oq.Circuit(3).h(0).h(1).h(2)
    for i in range(24):
        angle = oq.Parameter(f"weight_{i:02d}")
        if workload == "shared_relative":
            angle = angle - oq.Parameter(f"weight_{(i + 1) % 24:02d}")
        c = c.ry(angle, i % 3)
        if i % 3 == 2:
            c = c.cx(0, 1).cx(1, 2)
    rng = np.random.default_rng(91)
    rows = [dict(zip(c.parameters, row, strict=True)) for row in rng.uniform(-0.6, 0.6, (8, 24))]
    tangents = rng.normal(size=(8, 24))
    if workload == "one_active":
        tangents[:, 1:] = 0
    elif workload == "shared_relative":
        tangents[:] = 1
        tangents[:, -1] = 0
    indices = np.flatnonzero(np.any(tangents != 0, axis=0))
    obs = [oq.PauliString(p) for p in ("ZII", "IZI", "IIZ", "XXX", "ZZI", "IZZ")]

    class Counter:
        name = "directional-benchmark"
        capabilities = oq.BackendCapabilities(native_expectations=True)

        def __init__(self):
            self.jobs = 0

        def expectations_batch(self, circuits, observables, *, options):
            self.jobs += len(circuits)
            return oq.expectations_batch(circuits, observables, max_memory_mb=32)

    backend = Counter()

    def run():
        if mode == "jacobian":
            jac = oq.parameter_shift_jacobian(c, obs, rows, chunk_size=32, backend=backend)
            return np.einsum("bmp,bp->bm", jac, tangents)
        if mode == "selected_jacobian":
            names = tuple(c.parameters[int(i)] for i in indices)
            jac = oq.parameter_shift_jacobian(
                c, obs, rows, parameters=names, chunk_size=32, backend=backend
            )
            return np.einsum("bmp,bp->bm", jac, tangents[:, indices])
        return oq.parameter_shift_jvp(c, obs, rows, tangents, chunk_size=32, backend=backend)

    value = run()
    backend.jobs = 0
    run()
    jobs = backend.jobs
    times = []
    for _ in range(5):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    tracemalloc.start()
    run()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "version": oq.__version__,
        "import_path": str(actual),
        "source_sha256": source_hash(source / "openqml"),
        "mode": mode,
        "workload": workload,
        "median_seconds": statistics.median(times),
        "repeats": 5,
        "shifted_jobs": jobs,
        "tracemalloc_peak_bytes": peak,
        "values": value.tolist(),
        "result_array_bytes": value.nbytes,
        "jacobian_array_bytes": 8 * 6 * 8 * (24 if mode == "jacobian" else len(indices))
        if mode != "jvp"
        else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results_directional.json"))
    parser.add_argument("--worker", choices=("jacobian", "selected_jacobian", "jvp"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--workload", choices=("dense", "one_active", "shared_relative"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.source, args.worker, args.workload)))
        return
    import numpy as np

    current = Path(__file__).resolve().parents[1] / "src"
    cases = []
    for workload in ("dense", "one_active", "shared_relative"):
        configurations = [(current, "jacobian"), (current, "jvp")]
        if args.baseline_source:
            configurations.insert(0, (args.baseline_source.resolve(), "jacobian"))
            if workload != "dense":
                configurations.insert(1, (args.baseline_source.resolve(), "selected_jacobian"))
        results = []
        for source, mode in configurations:
            env = dict(
                os.environ,
                PYTHONPATH=str(source),
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    mode,
                    "--source",
                    str(source),
                    "--workload",
                    workload,
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            results.append(json.loads(process.stdout))
        for result in results[1:]:
            np.testing.assert_allclose(result["values"], results[0]["values"], atol=1e-12)
        cases.append(
            {
                "workload": workload,
                "results": results,
                "baseline_over_jvp": results[0]["median_seconds"] / results[-1]["median_seconds"],
                "selected_baseline_over_jvp": next(
                    (
                        r["median_seconds"] / results[-1]["median_seconds"]
                        for r in results
                        if r["mode"] == "selected_jacobian"
                    ),
                    None,
                ),
            }
        )
    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "batch": 8,
        "parameters": 24,
        "observables": 6,
        "qubits": 3,
        "chunk_size": 32,
        "note": "Warm CPU microbenchmark, not a task/GPU/quantum advantage claim. Includes prior"
        " manually selected columns to avoid treating a full Jacobian as the only baseline."
        " Tracemalloc is not RSS or autodiff tape memory; dense JVP need not be faster.",
        "cases": cases,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps({"output": str(args.output), "ratios": [c["baseline_over_jvp"] for c in cases]})
    )


if __name__ == "__main__":
    main()
