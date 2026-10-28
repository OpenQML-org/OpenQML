"""Audit construction, NumPy state working memory and cold/warm bridge latency.

Every source/case uses a fresh CPU worker. Linux RSS windows reset VmHWM after
imports and input setup; no fork/exec lifetime high-water mark is attributed to
the call. Kernel RSS counters remain approximate. Native autodiff/GPU memory is
outside this NumPy measurement. Bridge timings use a local analytic test backend,
not a provider, and report imports/initialization separately from warm calls.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from _memory import measure_peak
from _precision import scoped_x64
from planned_execution import source_hash


def worker(args):
    started = time.perf_counter()
    import numpy as np

    import openqml as oq

    imported = time.perf_counter() - started
    assert Path(oq.__file__).resolve().is_relative_to(args.source.resolve())
    if args.case == "build":

        def run():
            circuit = oq.CircuitBuilder(2) if hasattr(oq, "CircuitBuilder") else oq.Circuit(2)
            for index in range(args.gates):
                circuit = circuit.ry(0.001, index % 2)
            return circuit.build() if hasattr(circuit, "build") else circuit

        reference = run()
        samples = []
        for _ in range(args.repetitions):
            start = time.perf_counter()
            result = run()
            samples.append(time.perf_counter() - start)
            assert result == reference
        return dict(
            case="build",
            gates=args.gates,
            method="builder" if hasattr(oq, "CircuitBuilder") else "immutable_append",
            median_seconds=statistics.median(samples),
            samples_seconds=samples,
            num_operations=len(reference.operations),
        )
    if args.case == "memory":
        from openqml.simulator import _apply_matrix, _TensorOps

        width = args.qubits
        state = np.ones(2**width, dtype=complex) / np.sqrt(2**width)
        rng = np.random.default_rng(7)
        matrix, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))

        def run():
            return _apply_matrix(state, matrix, (width - 1, 2), width, _TensorOps())

        started = time.perf_counter()
        result, report = measure_peak(run)
        elapsed = time.perf_counter() - started
        np.testing.assert_allclose(np.vdot(result, result), 1, atol=1e-11)
        # Allocate the independent reference only after capturing the RSS window.
        wires = (width - 1, 2)
        order = wires + tuple(q for q in range(width) if q not in wires)
        inverse = tuple(order.index(q) for q in range(width))
        block = state.reshape((2,) * width).transpose(order).reshape(4, -1)
        expected = (matrix @ block).reshape((2,) * width).transpose(inverse).reshape(-1)
        np.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)
        times = []
        for _ in range(args.repetitions):
            start = time.perf_counter()
            actual = run()
            times.append(time.perf_counter() - start)
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
            del actual
        import hashlib

        report.update(
            case="memory",
            qubits=width,
            state_bytes=state.nbytes,
            memory_window_seconds=elapsed,
            warm_kernel_median_seconds=statistics.median(times),
            warm_kernel_samples_seconds=times,
            output_real_sum=float(result.real.sum()),
            output_imag_sum=float(result.imag.sum()),
            output_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
            max_absolute_error=float(np.max(np.abs(result - expected))),
        )
        return report

    before = time.perf_counter()
    __import__(args.interface)
    framework_import = time.perf_counter() - before
    return bridge(args, oq, np, imported, framework_import)


@scoped_x64
def bridge(args, oq, np, imported, framework_import):
    class AnalyticBackend:
        name = "audit-analytic"
        capabilities = oq.BackendCapabilities(
            native_expectations=True, deterministic_expectations=True
        )

        def expectations_batch(self, circuits, observables, *, options):
            return np.asarray([[np.cos(c.operations[0].params[0].constant)] for c in circuits])

    circuit = oq.Circuit(1).ry(oq.Parameter("x") * oq.Parameter("w"), 0)
    start = time.perf_counter()
    if args.interface == "torch":
        import torch

        start = time.perf_counter()
        layer = oq.torch.QuantumLayer(
            circuit, ["x"], ["w"], backend=AnalyticBackend(), dtype=torch.float64
        )
        weights = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
        inputs = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)
        preparation = time.perf_counter() - start

        def forward():
            return layer.functional(weights, inputs).sum()

        def backward(value):
            return tuple(g.detach().numpy() for g in torch.autograd.grad(value, (weights, inputs)))
    else:
        import jax
        import jax.numpy as jnp

        start = time.perf_counter()
        layer = oq.jax.QuantumLayer(circuit, ["x"], ["w"], backend=AnalyticBackend())
        weights, inputs = jnp.array([0.4]), jnp.array([[0.3]])
        preparation = time.perf_counter() - start
        forward_compiled = jax.jit(lambda w, x: layer(w, x).sum())
        gradients = jax.jit(jax.grad(lambda w, x: layer(w, x).sum(), argnums=(0, 1)))

        def forward():
            return forward_compiled(weights, inputs).block_until_ready()

        def backward(value):
            return tuple(np.asarray(g) for g in gradients(weights, inputs))

    start = time.perf_counter()
    value = forward()
    middle = time.perf_counter()
    derivative = backward(value)
    end = time.perf_counter()
    np.testing.assert_allclose(
        float(value.detach()) if args.interface == "torch" else float(value),
        np.cos(0.12),
        atol=1e-12,
    )
    np.testing.assert_allclose(derivative[0], [-0.3 * np.sin(0.12)], atol=1e-12)
    np.testing.assert_allclose(derivative[1], [[-0.4 * np.sin(0.12)]], atol=1e-12)
    forwards, backwards = [], []
    for _ in range(args.repetitions):
        before = time.perf_counter()
        output = forward()
        mid = time.perf_counter()
        backward(output)
        after = time.perf_counter()
        forwards.append(mid - before)
        backwards.append(after - mid)
    return dict(
        case="bridge",
        interface=args.interface,
        library_import_seconds=imported,
        framework_import_seconds=framework_import,
        layer_prepare_seconds=preparation,
        cold_forward_seconds=middle - start,
        cold_backward_seconds=end - middle,
        warm_forward_median_seconds=statistics.median(forwards),
        warm_backward_median_seconds=statistics.median(backwards),
        backend="local_analytic_no_sdk_or_provider",
        scope="cpu_framework_bridge",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--case", choices=("build", "memory", "bridge"), required=True)
    parser.add_argument("--interface", choices=("torch", "jax"))
    parser.add_argument("--gates", type=int, default=2000)
    parser.add_argument("--qubits", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.gates, args.repetitions) < 1 or args.qubits < 4:
        parser.error("Positive gates/repetitions and at least four qubits are required")
    if args.case == "bridge" and args.interface is None:
        args.interface = "torch"
    if args.worker:
        sys.path.insert(0, str(args.source.resolve()))
        print(json.dumps(worker(args), allow_nan=False))
        return
    runs = []
    sources = ([args.baseline_source] if args.baseline_source is not None else []) + [args.source]
    for source in sources:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--source",
            str(source.resolve()),
            "--case",
            args.case,
            "--gates",
            str(args.gates),
            "--qubits",
            str(args.qubits),
            "--repetitions",
            str(args.repetitions),
        ]
        if args.interface is not None:
            command += ["--interface", args.interface]
        completed = subprocess.run(
            command,
            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
            text=True,
            capture_output=True,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        result = json.loads(completed.stdout)
        result["source"] = str(source.resolve())
        result["source_sha256"] = source_hash(source)
        runs.append(result)
    if args.case == "memory" and len(runs) == 2:
        import numpy as np

        for field in ("output_real_sum", "output_imag_sum"):
            np.testing.assert_allclose(runs[0][field], runs[1][field], atol=1e-9, rtol=1e-12)
    report = json.dumps({"runs": runs}, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
