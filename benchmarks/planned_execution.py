"""Benchmark exact portfolio execution against previous source in fresh processes.

The workload has opaque two-wire blocks and shared product/Hamiltonian readouts.
Preparation is timed separately. Every timed numerical result is checked.
Use --torch for complete encoder/quantum/head/backward/Adam training steps.
"""

import argparse
import hashlib
import json
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
    import openqml.simulator as simulator

    assert Path(oq.__file__).resolve().is_relative_to(args.source.resolve())
    width = 2 * args.blocks
    c, terms = oq.Circuit(width), []
    x = np.array([[0, 1], [1, 0]])
    matrix = np.cos(0.07 / 2) * np.eye(4) - 1j * np.sin(0.07 / 2) * np.kron(x, x)
    ry = np.array([[np.cos(0.05), -np.sin(0.05)], [np.sin(0.05), np.cos(0.05)]])
    matrix = matrix @ np.diag([1, 1, 1, -1]) @ np.kron(ry, np.eye(2))
    for q in range(0, width, 2):
        c = c.ry(oq.Parameter("feature") * oq.Parameter("weight"), q).unitary(matrix, q, q + 1)
        terms.append(oq.PauliString("I" * q + "Z" + "I" * (width - q - 1), 1 / args.blocks))
    obs = (oq.PauliString("ZI" * args.blocks), oq.Hamiltonian(tuple(terms)))
    if args.case == "training":
        import torch

        from openqml.interfaces.torch import QuantumLayer

        torch.set_num_threads(1)
        torch.manual_seed(71)
        options = (
            {"execution_method": "planned"}
            if args.mode == "planned"
            else {"reduce_readout": True}
            if args.mode == "readout"
            else {}
        )
        start = time.perf_counter()
        layer = QuantumLayer(
            c, ("feature",), ("weight",), obs, dtype=torch.float64, max_qubits=width, **options
        )
        preparation = time.perf_counter() - start
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 1, dtype=torch.float64),
            layer,
            torch.nn.Linear(2, 1, dtype=torch.float64),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        features = torch.randn(args.batch, 4, dtype=torch.float64)

        def run():
            optimizer.zero_grad()
            loss = model(features).square().mean()
            loss.backward()
            result = torch.cat(
                (
                    loss.detach().reshape(1),
                    *(p.grad.detach().reshape(-1) for p in model.parameters()),
                )
            ).numpy()
            optimizer.step()
            return result
    else:
        columns = dict(feature=np.linspace(-0.2, 0.3, args.batch), weight=0.7)
        start = time.perf_counter()
        if args.mode == "planned":
            plan = oq.ExpectationPlan(c, obs)

            def run():
                return plan.evaluate(columns).reshape(-1)
        else:
            reduced = oq.plan_readout(c, obs) if args.mode == "readout" else None
            target, readouts = (reduced.circuit, reduced.observables) if reduced else (c, obs)

            def run():
                return oq.expectations(target, readouts, columns, max_qubits=width).reshape(-1)

        preparation = time.perf_counter() - start
    checked, times = [run()], []
    for _ in range(args.repeats):
        start = time.perf_counter()
        value = run()
        times.append(time.perf_counter() - start)
        checked.append(value)
    calls = amplitudes = saved_complex = 0
    apply = simulator._apply_matrix

    def counted(state, *inner, **kwargs):
        nonlocal calls, amplitudes
        calls += 1
        amplitudes = max(amplitudes, state.shape[-1])
        return apply(state, *inner, **kwargs)

    def pack(tensor):
        nonlocal saved_complex
        if tensor.is_complex():
            saved_complex += tensor.numel() * tensor.element_size()
        return tensor

    with patch.object(simulator, "_apply_matrix", counted):
        if args.case == "training":
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
                run()
        else:
            run()
    print(
        json.dumps(
            {
                "source_sha256": source_hash(args.source),
                "mode": args.mode,
                "case": args.case,
                "source_qubits": width,
                "batch": args.batch,
                "source_operations": len(c.operations),
                "preparation_seconds": preparation,
                "median_seconds": statistics.median(times),
                "samples_seconds": times,
                "matrix_kernel_calls": calls,
                "largest_state_amplitudes": amplitudes,
                "saved_complex_tensor_bytes": saved_complex,
                "checked_values": np.stack(checked).tolist(),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--blocks", type=int, default=7)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode", choices=("full", "readout", "planned"), default="full", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--case",
        choices=("expectations", "training"),
        default="expectations",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if min(args.blocks, args.batch, args.repeats) < 1:
        parser.error("Use positive block/batch/repeat counts.")
    if args.worker:
        worker(args)
        return
    report = []
    for case in ("expectations", "training") if args.torch else ("expectations",):
        cases = [
            ("full", args.baseline_source or args.source),
            ("readout", args.source),
            ("planned", args.source),
        ]
        if args.baseline_source is not None:
            cases.insert(1, ("full", args.source))
        runs = []
        for mode, source in cases:
            source = source.resolve()
            env = {
                **os.environ,
                "PYTHONPATH": str(source),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--source",
                str(source),
                "--case",
                case,
                "--mode",
                mode,
                "--blocks",
                str(args.blocks),
                "--batch",
                str(args.batch),
                "--repeats",
                str(args.repeats),
            ]
            result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
            runs.append(json.loads(result.stdout))
        errors = []
        for run in runs[1:]:
            np.testing.assert_allclose(
                run["checked_values"], runs[0]["checked_values"], atol=3e-11, rtol=3e-11
            )
            difference = np.array(run["checked_values"]) - runs[0]["checked_values"]
            errors.append(float(np.max(np.abs(difference))))
        for run in runs:
            del run["checked_values"]
        report.append(
            {
                "case": case,
                "runs": runs,
                "speedup": runs[0]["median_seconds"] / runs[-1]["median_seconds"],
                "max_absolute_errors": errors,
            }
        )
    text = json.dumps({"results": report}, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
