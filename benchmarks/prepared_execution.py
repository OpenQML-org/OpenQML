"""Compare prepared execution with an exact earlier source checkout in fresh processes.

Use --baseline-source /path/to/previous/src --output /path/result.json.
Reports CPU medians and matrix-kernel dispatches, without timing assertions.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np


def source_hash(source):
    digest = hashlib.sha256()
    for path in sorted((source / "openqml").rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def worker(args):
    import openqml as oq
    import openqml.adjoint as engine
    import openqml.simulator as simulator

    assert Path(oq.__file__).resolve().is_relative_to(args.source.resolve())
    rng = np.random.default_rng(71)
    if args.case == "torch_training":
        import torch

        from openqml.interfaces.torch import QuantumLayer

        torch.manual_seed(23)
        spec = oq.data_reuploading(4, 6)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4, dtype=torch.float64),
            QuantumLayer.from_spec(
                spec, dtype=torch.float64, backend="numpy", gradient_method="adjoint"
            ),
            torch.nn.Linear(4, 1, dtype=torch.float64),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        features = torch.tensor(rng.normal(size=(64, 4)), dtype=torch.float64)

        def run():
            optimizer.zero_grad()
            loss = model(features).square().mean()
            loss.backward()
            optimizer.step()
            return torch.cat(
                (
                    loss.detach().reshape(1),
                    *(p.grad.detach().reshape(-1) for p in model.parameters()),
                )
            ).numpy()

        value = run()
        metadata = {
            "qubits": 4,
            "batch": 64,
            "stage": "classical encoder, quantum layer, classical head, backward and Adam update",
        }
    elif args.case == "conversion":
        dimension = 1 << args.matrix_qubits
        matrix, _ = np.linalg.qr(
            rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
        )
        circuit = (
            oq.Circuit(args.matrix_qubits)
            .unitary(matrix, *reversed(range(args.matrix_qubits)))
            .ry(oq.Parameter("weight"), 0)
            .with_global_phase(oq.Parameter("weight") / 3)
        )
        rows = [{"weight": float(x)} for x in rng.uniform(-0.4, 0.4, 16)]

        def run():
            return [
                oq.plan_conversion(circuit, target="ir", bindings=row).export().value
                for row in rows
            ]

        outputs = run()
        value = np.stack(
            [np.stack((oq.statevector(c).real, oq.statevector(c).imag), axis=-1) for c in outputs]
        )
        metadata = {
            "qubits": args.matrix_qubits,
            "batch": len(rows),
            "stage": "IR import/binding/plan export; no native SDK synthesis",
        }
    else:
        spec = oq.data_reuploading(4, 6)
        circuit = spec.circuit.with_global_phase(oq.Parameter("phase"))
        labels = ("".join(p) for p in itertools.product("IXYZ", repeat=4))
        observables = tuple(oq.PauliString(label) for label in itertools.islice(labels, 1, 9))
        rows = [
            dict(zip(circuit.parameters, row, strict=True))
            for row in rng.uniform(-0.4, 0.4, (64, len(circuit.parameters)))
        ]
        cotangents = rng.normal(size=(len(rows), len(observables)))
        tangents = rng.normal(size=(len(rows), len(circuit.parameters)))
        if args.mode == "prepared":
            plan = oq.AdjointPlan(circuit, observables, max_memory_mb=128)
            columns = {name: np.array([row[name] for row in rows]) for name in circuit.parameters}

            def run():
                if args.case == "vjp":
                    return plan.vjp(columns, cotangents)
                if args.case == "jvp":
                    return plan.jvp(columns, tangents)
                return plan.expectations(columns)
        else:

            def run():
                if args.case == "vjp":
                    return oq.adjoint_vjp(circuit, observables, rows, cotangents, max_memory_mb=128)
                if args.case == "jvp":
                    return oq.adjoint_jvp(circuit, observables, rows, tangents, max_memory_mb=128)
                return oq.expectations_batch(circuit, observables, rows, max_memory_mb=128)

        value = run()
        metadata = {
            "qubits": 4,
            "batch": len(rows),
            "readouts": len(observables),
            "symbols": len(circuit.parameters),
            "gates": len(circuit.operations),
        }
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    calls = equivalents = 0
    apply = simulator._apply_matrix

    def counted(state, *inner_args, **kwargs):
        nonlocal calls, equivalents
        calls += 1
        equivalents += math.prod(state.shape[:-1])
        return apply(state, *inner_args, **kwargs)

    with (
        patch.object(simulator, "_apply_matrix", counted),
        patch.object(engine, "_apply_matrix", counted),
    ):
        run()
    return {
        "source_sha256": source_hash(args.source),
        "source": str(args.source),
        "mode": args.mode,
        "median_seconds": statistics.median(times),
        "repeats": args.repeats,
        "matrix_kernel_calls": calls,
        "statevector_gate_applications": equivalents,
        "workload": metadata,
        "values": value.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--matrix-qubits", type=int, default=7)
    parser.add_argument(
        "--torch",
        action="store_true",
        help="Include full PyTorch optimizer steps; requires the torch extra.",
    )
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--case", choices=("vjp", "jvp", "primal", "conversion", "torch_training"))
    parser.add_argument("--mode", choices=("standalone", "prepared"))
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.matrix_qubits <= 8:
        parser.error("Use positive repeats and matrix-qubits between 1 and 8.")
    if args.worker:
        print(json.dumps(worker(args)))
        return
    current = Path(__file__).resolve().parents[1] / "src"
    cases = []
    workloads = ("vjp", "jvp", "primal", "conversion") + (("torch_training",) if args.torch else ())
    for case in workloads:
        configurations = []
        if args.baseline_source is not None:
            configurations.append((args.baseline_source.resolve(), "standalone"))
        if case != "torch_training":
            configurations.append((current, "standalone"))
        if case != "conversion":
            configurations.append((current, "prepared"))
        results, reference = [], None
        for source, mode in configurations:
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--source",
                    str(source),
                    "--case",
                    case,
                    "--mode",
                    mode,
                    "--repeats",
                    str(args.repeats),
                    "--matrix-qubits",
                    str(args.matrix_qubits),
                ],
                env=dict(
                    os.environ,
                    PYTHONPATH=str(source),
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                ),
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(process.stdout)
            values = np.asarray(result.pop("values"))
            if reference is None:
                reference = values
            np.testing.assert_allclose(values, reference, atol=3e-11, rtol=3e-11)
            result["max_abs_difference"] = float(np.max(np.abs(values - reference)))
            results.append(result)
        cases.append(
            {
                "case": case,
                "results": results,
                "baseline_over_final": results[0]["median_seconds"] / results[-1]["median_seconds"],
            }
        )
    payload = {
        "note": "Fresh-process warm CPU microbenchmarks against an exact source baseline. "
        "Conversion measures IR preparation, excluding native SDK synthesis. "
        "State-vector work, kernel dispatches and timing are different metrics; "
        "speedups depend on workload and machine.",
        "cases": cases,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output) if args.output else None,
                "ratios": {c["case"]: c["baseline_over_final"] for c in cases},
            }
        )
    )


if __name__ == "__main__":
    main()
