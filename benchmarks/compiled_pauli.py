"""Measure reusable Pauli arithmetic against exact earlier source in fresh processes.

The connected GHZ workload has a small propagated operator frontier. Results
are not claims about general circuits. --torch includes full hybrid training.
Preparation and warm-up are separate from seven default timed repetitions.
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
    circuit = oq.Circuit(args.qubits).ry(oq.Parameter("feature") * oq.Parameter("weight"), 0)
    for q in range(args.qubits - 1):
        circuit = circuit.cx(q, q + 1)
    obs = (oq.PauliString("I" * (args.qubits - 1) + "Z"), oq.PauliString("X" * args.qubits))
    if args.case == "torch_training":
        import torch

        from openqml.interfaces.torch import QuantumLayer

        torch.set_num_threads(1)
        torch.manual_seed(71)
        options = (
            {"reduce_readout": True}
            if args.mode == "readout"
            else {"execution_method": "pauli"}
            if args.mode == "pauli"
            else {}
        )
        start = time.perf_counter()
        layer = QuantumLayer(
            circuit,
            ("feature",),
            ("weight",),
            obs,
            dtype=torch.float64,
            max_qubits=args.qubits,
            **options,
        )
        preparation = time.perf_counter() - start
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 1, dtype=torch.float64),
            layer,
            torch.nn.Linear(2, 1, dtype=torch.float64),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        data = torch.randn(args.batch, 4, dtype=torch.float64)

        def run():
            optimizer.zero_grad()
            loss = model(data).square().mean()
            loss.backward()
            value = torch.cat(
                (
                    loss.detach().reshape(1),
                    *(p.grad.detach().reshape(-1) for p in model.parameters()),
                )
            ).numpy()
            optimizer.step()
            return value
    else:
        columns = {"feature": np.linspace(-0.6, 0.8, args.batch), "weight": 0.7}
        start = time.perf_counter()
        if args.mode == "pauli":
            plan = oq.PauliPlan(circuit, obs)

            def run():
                return plan.evaluate(columns).reshape(-1)
        else:
            reduced = oq.plan_readout(circuit, obs) if args.mode == "readout" else None
            target, readouts = (reduced.circuit, reduced.observables) if reduced else (circuit, obs)

            def run():
                return oq.expectations(target, readouts, columns, max_qubits=args.qubits).reshape(
                    -1
                )

        preparation = time.perf_counter() - start

    checked = [run()]
    times = []
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
        if args.case == "torch_training":
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
                "qubits": args.qubits,
                "batch": args.batch,
                "source_gates": len(circuit.operations),
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
    parser.add_argument("--qubits", type=int, default=14)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode", choices=("full", "readout", "pauli"), default="full", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--case",
        choices=("numpy_expectations", "torch_training"),
        default="numpy_expectations",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.qubits < 2 or min(args.batch, args.repeats) < 1:
        parser.error("Use at least two wires and positive batch/repeats.")
    if args.worker:
        worker(args)
        return
    report = []
    for case in ("numpy_expectations", "torch_training") if args.torch else ("numpy_expectations",):
        cases = [
            ("full", args.baseline_source or args.source),
            ("readout", args.source),
            ("pauli", args.source),
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
                "--qubits",
                str(args.qubits),
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
                run["checked_values"], runs[0]["checked_values"], atol=2e-11, rtol=2e-11
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
