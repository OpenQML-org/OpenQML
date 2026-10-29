"""Measure sparse readout routing and exact matrix keys against previous source.

Fresh single-thread CPU workers compare the previous packed planner with the
current planner. Every timed expectation and full training loss/gradient vector
is checked. The matrix case measures key construction plus exact equality;
tracemalloc reports key allocations, not process RAM or training memory.
"""

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import time
import tracemalloc
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
from planned_execution import source_hash


def worker(args):
    import openqml as oq
    import openqml.simulator as simulator

    assert Path(oq.__file__).resolve().is_relative_to(args.source.resolve())
    rng = np.random.default_rng(18)
    if args.case == "matrix_identity":
        from openqml._scheduling import _matrix_key

        matrix, _ = np.linalg.qr(rng.normal(size=(256, 256)) + 1j * rng.normal(size=(256, 256)))
        payload = tuple(tuple(complex(z) for z in row) for row in matrix)
        reference = _matrix_key(payload)
        changed = matrix.copy()
        changed[0, 0] += 1e-12
        assert reference != _matrix_key(tuple(tuple(complex(z) for z in row) for row in changed))

        def run():
            assert _matrix_key(payload) == reference
            return np.array([1.0])

        preparation = 0
    else:
        matrix, _ = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
        labels = ["".join(p) for p in product("IXYZ", repeat=3) if set(p) != {"I"}][: args.blocks]
        inputs, weights = (
            tuple(f"x{i}" for i in range(args.blocks)),
            tuple(f"w{i}" for i in range(args.blocks)),
        )
        circuit, terms = oq.Circuit(3 * args.blocks), []
        for i, label in enumerate(labels):
            circuit = (
                circuit.ry(oq.Parameter(inputs[i]) * oq.Parameter(weights[i]), 3 * i)
                .rx(0.23, 3 * i + 2)
                .unitary(matrix, 3 * i, 3 * i + 1, 3 * i + 2)
            )
            terms.append(
                oq.PauliString("III" * i + label + "III" * (args.blocks - i - 1), 1 / args.blocks)
            )
        obs = (oq.PauliString("".join(labels)), oq.Hamiltonian(tuple(terms)))
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
                dtype=torch.float64,
                max_qubits=circuit.num_qubits,
            )
            preparation = time.perf_counter() - start
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
                result = torch.cat(
                    (
                        loss.detach().reshape(1),
                        *(p.grad.detach().reshape(-1) for p in model.parameters()),
                    )
                ).numpy()
                optimizer.step()
                return result
        else:
            bindings = {
                **{n: np.linspace(-0.2, 0.3, args.batch) + i / 1000 for i, n in enumerate(inputs)},
                **{n: 0.3 + i / 100 for i, n in enumerate(weights)},
            }
            start = time.perf_counter()
            plan = oq.ExpectationPlan(circuit, obs)
            preparation = time.perf_counter() - start

            def run():
                return plan.evaluate(bindings).reshape(-1)

    checked, samples = [run()], []
    for _ in range(args.repeats):
        start = time.perf_counter()
        value = run()
        samples.append(time.perf_counter() - start)
        checked.append(value)
    calls = readout_rows = saved_complex = retained = peak = 0
    if args.case == "matrix_identity":
        gc.collect()
        tracemalloc.start()
        key = _matrix_key(payload)
        retained, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert key == reference
    else:
        apply, read = simulator._apply_matrix, simulator._expectation_from_state

        def gate(state, *args):
            nonlocal calls
            calls += 1
            return apply(state, *args)

        def readout(state, *args):
            nonlocal readout_rows
            readout_rows += math.prod(state.shape[:-1])
            return read(state, *args)

        def pack(tensor):
            nonlocal saved_complex
            if tensor.is_complex():
                saved_complex += tensor.numel() * tensor.element_size()
            return tensor

        with (
            patch.object(simulator, "_apply_matrix", gate),
            patch.object(simulator, "_expectation_from_state", readout),
        ):
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
                "source_qubits": 3 * args.blocks if args.case != "matrix_identity" else None,
                "batch": args.batch,
                "preparation_seconds": preparation,
                "median_seconds": statistics.median(samples),
                "samples_seconds": samples,
                "matrix_kernel_calls": calls,
                "readout_rows": readout_rows,
                "saved_complex_tensor_bytes": saved_complex,
                "key_traced_retained_bytes": retained,
                "key_traced_peak_bytes": peak,
                "checked_values": np.stack(checked).tolist(),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--case",
        choices=("expectations", "training", "matrix_identity"),
        default="expectations",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if not 1 <= args.blocks <= 63 or min(args.batch, args.repeats) < 1:
        parser.error("Use 1 to 63 blocks and positive batch/repeat counts.")
    if args.worker:
        worker(args)
        return
    report = []
    for case in (
        ("expectations", "training", "matrix_identity")
        if args.torch
        else ("expectations", "matrix_identity")
    ):
        runs = []
        for source in (args.baseline_source, args.source):
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
                "--case",
                case,
                "--blocks",
                str(args.blocks),
                "--batch",
                str(args.batch),
                "--repeats",
                str(args.repeats),
            ]
            run = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
            runs.append(json.loads(run.stdout))
        np.testing.assert_allclose(
            runs[1]["checked_values"], runs[0]["checked_values"], atol=3e-11, rtol=3e-11
        )
        error = float(
            np.max(np.abs(np.array(runs[1]["checked_values"]) - runs[0]["checked_values"]))
        )
        for run in runs:
            del run["checked_values"]
        report.append(
            {
                "case": case,
                "runs": runs,
                "speedup": runs[0]["median_seconds"] / runs[1]["median_seconds"],
                "max_absolute_error": error,
            }
        )
    result = json.dumps({"results": report}, indent=2)
    print(result)
    if args.output:
        args.output.write_text(result + "\n")


if __name__ == "__main__":
    main()
