"""Compare planned execution before and after dense function reuse and packing.

Each source has opaque entangling two-wire blocks. Tied parameters permit exact
function reuse; independent parameters permit template batching. The baseline
already factors the circuit and uses local dense execution. Every timed output
or complete encoder/quantum/head/backward/Adam trajectory is checked. Preparation
is separate; workers use fresh processes with one CPU thread.
"""

import argparse
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
from planned_execution import source_hash


def worker(args):
    import openqml as oq
    import openqml.simulator as simulator

    assert Path(oq.__file__).resolve().is_relative_to(args.source.resolve())
    tied = args.family == "tied"
    width = 2 * args.blocks
    inputs = ("x",) if tied else tuple(f"x{i}" for i in range(args.blocks))
    weights = ("w",) if tied else tuple(f"w{i}" for i in range(args.blocks))
    x = np.array([[0, 1], [1, 0]])
    matrix = np.cos(0.035) * np.eye(4) - 1j * np.sin(0.035) * np.kron(x, x)
    ry = np.array([[np.cos(0.05), -np.sin(0.05)], [np.sin(0.05), np.cos(0.05)]])
    matrix = matrix @ np.diag([1, 1, 1, -1]) @ np.kron(ry, np.eye(2))
    circuit, terms = oq.Circuit(width), []
    for i in range(args.blocks):
        angle = oq.Parameter(inputs[0 if tied else i]) * oq.Parameter(weights[0 if tied else i])
        circuit = circuit.ry(angle, 2 * i).unitary(matrix, 2 * i, 2 * i + 1)
        terms.append(
            oq.PauliString("II" * i + "ZI" + "II" * (args.blocks - i - 1), 1 / args.blocks)
        )
    obs = (oq.PauliString("ZI" * args.blocks), oq.Hamiltonian(tuple(terms)))
    planning = {"max_dense_batch": args.dense_batch} if args.dense_batch else {}
    if args.case == "training":
        import torch

        from openqml.interfaces.torch import QuantumLayer

        torch.set_num_threads(1)
        torch.manual_seed(71)
        start = time.perf_counter()
        layer = QuantumLayer(
            circuit,
            inputs,
            weights,
            obs,
            execution_method="planned",
            planning_options=planning,
            dtype=torch.float64,
            max_qubits=width,
        )
        preparation = time.perf_counter() - start
        plan = layer.execution_plan
        model = torch.nn.Sequential(
            torch.nn.Linear(4, len(inputs), dtype=torch.float64),
            layer,
            torch.nn.Linear(2, 1, dtype=torch.float64),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        features = torch.randn(args.batch, 4, dtype=torch.float64)

        def run():
            optimizer.zero_grad()
            loss = model(features).square().mean()
            loss.backward()
            values = torch.cat(
                (
                    loss.detach().reshape(1),
                    *(p.grad.detach().reshape(-1) for p in model.parameters()),
                )
            )
            optimizer.step()
            return values.numpy()
    else:
        bindings = {
            **{n: np.linspace(-0.2, 0.3, args.batch) + i / 1000 for i, n in enumerate(inputs)},
            **{n: 0.3 + i / 100 for i, n in enumerate(weights)},
        }
        start = time.perf_counter()
        plan = oq.ExpectationPlan(circuit, obs, **planning)
        preparation = time.perf_counter() - start

        def run():
            return plan.evaluate(bindings).reshape(-1)

    checked, times = [run()], []
    for _ in range(args.repeats):
        start = time.perf_counter()
        value = run()
        times.append(time.perf_counter() - start)
        checked.append(value)
    calls = amplitudes = state_rows = saved_complex = 0
    original = simulator._apply_matrix

    def counted(state, *args, **kwargs):
        nonlocal calls, amplitudes, state_rows
        calls += 1
        amplitudes = max(amplitudes, state.shape[-1])
        state_rows = max(state_rows, math.prod(state.shape[:-1]))
        return original(state, *args, **kwargs)

    def pack(tensor):
        nonlocal saved_complex
        if tensor.is_complex():
            saved_complex += tensor.numel() * tensor.element_size()
        return tensor

    with patch.object(simulator, "_apply_matrix", counted):
        if args.case == "training":
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                run()
        else:
            run()
    print(
        json.dumps(
            {
                "source_sha256": source_hash(args.source),
                "case": args.case,
                "family": args.family,
                "source_qubits": width,
                "source_blocks": args.blocks,
                "batch": args.batch,
                "dense_batch_limit": args.dense_batch or None,
                "dense_unique_lanes": (
                    sum(len(b.parameter_slots) for b in plan.dense_batches)
                    if hasattr(plan, "dense_batches")
                    else args.blocks
                ),
                "contraction_nodes": len(plan._nodes),
                "preparation_seconds": preparation,
                "median_seconds": statistics.median(times),
                "samples_seconds": times,
                "matrix_kernel_calls": calls,
                "largest_state_amplitudes": amplitudes,
                "largest_state_rows": state_rows,
                "saved_complex_tensor_bytes": saved_complex,
                "checked_values": np.stack(checked).tolist(),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--family", choices=("tied", "independent"), default="tied", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--case",
        choices=("expectations", "training"),
        default="expectations",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dense-batch", type=int, default=32, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.blocks, args.batch, args.repeats) < 1:
        parser.error("Use positive block, batch and repeat counts.")
    if args.worker:
        worker(args)
        return
    report = []
    for family in ("tied", "independent"):
        for case in ("expectations", "training") if args.torch else ("expectations",):
            runs = []
            for source, cap in ((args.baseline_source, 0), (args.source, 1), (args.source, 32)):
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
                    "--baseline-source",
                    str(args.baseline_source.resolve()),
                    "--source",
                    str(source),
                    "--family",
                    family,
                    "--case",
                    case,
                    "--dense-batch",
                    str(cap),
                    "--blocks",
                    str(args.blocks),
                    "--batch",
                    str(args.batch),
                    "--repeats",
                    str(args.repeats),
                ]
                result = subprocess.run(
                    command, env=env, capture_output=True, text=True, check=True
                )
                runs.append(json.loads(result.stdout))
            errors = []
            for run in runs[1:]:
                np.testing.assert_allclose(
                    run["checked_values"], runs[0]["checked_values"], atol=3e-11, rtol=3e-11
                )
                errors.append(
                    float(
                        np.max(np.abs(np.array(run["checked_values"]) - runs[0]["checked_values"]))
                    )
                )
            for run in runs:
                del run["checked_values"]
            report.append(
                {
                    "family": family,
                    "case": case,
                    "runs": runs,
                    "speedup": runs[0]["median_seconds"] / runs[-1]["median_seconds"],
                    "max_absolute_errors": errors,
                }
            )
    result = json.dumps({"results": report}, indent=2)
    print(result)
    if args.output:
        args.output.write_text(result + "\n")


if __name__ == "__main__":
    main()
