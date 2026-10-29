"""Compare full and reduced hybrid training in isolated CPU processes.

Requires openqml[torch]. Use --baseline-source /path/to/earlier/src for a source comparison.
The workload has a disconnected two-wire readout cone. Reported reductions
are workload-specific; saved tensor volume is not process peak memory.
"""

import argparse
import hashlib
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
    import torch

    import openqml as oq
    import openqml.adjoint as adjoint
    import openqml.simulator as simulator
    from openqml.interfaces.torch import QuantumLayer

    assert Path(oq.__file__).resolve().is_relative_to(args.source.resolve())
    torch.set_num_threads(1)
    torch.manual_seed(41)
    width = args.qubits
    inputs = tuple(f"x_{q}" for q in range(width))
    weights = tuple(f"w_{q}" for q in range(width))
    circuit = oq.Circuit(width)
    for _ in range(args.depth):
        for q, (x, w) in enumerate(zip(inputs, weights, strict=True)):
            circuit = circuit.ry(oq.Parameter(x) * oq.Parameter(w), q).rx(oq.Parameter(w), q)
        for q in range(width - 3):
            circuit = circuit.cx(q, q + 1)
        circuit = circuit.cx(width - 2, width - 1)
    obs = (
        oq.PauliString("I" * (width - 2) + "IZ"),
        oq.PauliString("I" * (width - 2) + "XX"),
    )
    options = {} if args.mode == "full" else {"reduce_readout": True}
    if args.backend == "numpy":
        options.update(backend="numpy", gradient_method="adjoint")
    layer = QuantumLayer(circuit, inputs, weights, obs, dtype=torch.float64, **options)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, width, dtype=torch.float64),
        layer,
        torch.nn.Linear(2, 1, dtype=torch.float64),
    )
    features = torch.randn(args.batch, 4, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def run():
        optimizer.zero_grad()
        loss = model(features).square().mean()
        loss.backward()
        value = torch.cat(
            (loss.detach().reshape(1), *(p.grad.reshape(-1) for p in model.parameters()))
        )
        optimizer.step()
        return value.numpy()

    value = run()
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)

    calls = state_amplitudes = batch_states = saved_complex_bytes = 0
    apply = simulator._apply_matrix

    def counted(state, *inner_args, **kwargs):
        nonlocal calls, state_amplitudes, batch_states
        calls += 1
        state_amplitudes = max(state_amplitudes, state.shape[-1])
        batch_states = max(batch_states, math.prod(state.shape[:-1]))
        return apply(state, *inner_args, **kwargs)

    def pack(tensor):
        nonlocal saved_complex_bytes
        if tensor.is_complex():
            saved_complex_bytes += tensor.numel() * tensor.element_size()
        return tensor

    with (
        patch.object(simulator, "_apply_matrix", counted),
        patch.object(adjoint, "_apply_matrix", counted),
        torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t),
    ):
        run()
    print(
        json.dumps(
            {
                "source_sha256": source_hash(args.source),
                "mode": args.mode,
                "backend": args.backend,
                "source_qubits": width,
                "executed_qubits": layer.circuit.num_qubits,
                "source_operations": len(circuit.operations),
                "executed_operations": len(layer.circuit.operations),
                "batch": args.batch,
                "depth": args.depth,
                "readouts": len(obs),
                "median_seconds": statistics.median(times),
                "samples_seconds": times,
                "matrix_kernel_calls": calls,
                "largest_state_amplitudes": state_amplitudes,
                "largest_state_batch": batch_states,
                "saved_complex_bytes": saved_complex_bytes,
                "loss_and_gradients": value.tolist(),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qubits", type=int, default=12)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode", choices=("full", "reduced"), default="full", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--backend", choices=("torch", "numpy"), default="torch", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.qubits < 4 or min(args.batch, args.depth, args.repeats) < 1:
        parser.error("Use at least four qubits and positive batch/depth/repeats.")
    if args.worker:
        worker(args)
        return

    results = []
    for backend in ("torch", "numpy"):
        runs = []
        cases = [("full", args.baseline_source or args.source), ("reduced", args.source)]
        if args.baseline_source is not None:
            cases.insert(1, ("full", args.source))
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
                "--mode",
                mode,
                "--backend",
                backend,
                "--qubits",
                str(args.qubits),
                "--batch",
                str(args.batch),
                "--depth",
                str(args.depth),
                "--repeats",
                str(args.repeats),
            ]
            result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
            runs.append(json.loads(result.stdout))
        errors = []
        for run in runs[1:]:
            np.testing.assert_allclose(
                run["loss_and_gradients"], runs[0]["loss_and_gradients"], atol=2e-12, rtol=2e-12
            )
            errors.append(
                float(
                    np.max(
                        np.abs(np.array(run["loss_and_gradients"]) - runs[0]["loss_and_gradients"])
                    )
                )
            )
        for run in runs:
            del run["loss_and_gradients"]
        results.append(
            {
                "backend": backend,
                "runs": runs,
                "speedup": runs[0]["median_seconds"] / runs[-1]["median_seconds"],
                "max_absolute_errors": errors,
                "state_amplitude_reduction": runs[0]["largest_state_amplitudes"]
                / runs[-1]["largest_state_amplitudes"],
            }
        )
    report = {"stage": "encoder, quantum layer, head, backward and Adam update", "results": results}
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
