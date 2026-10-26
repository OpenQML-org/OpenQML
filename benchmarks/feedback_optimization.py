"""Prior-source native hybrid training versus opt-in logical optimization.

Warm CPU medians exclude compile time, which is reported separately. State
and gradient probes use an independent analytic feedback formula. This does
not test injected physical-gate noise, GPU performance or quantum advantage.
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
from dataclasses import asdict
from pathlib import Path


def worker(args):
    import numpy as np
    import torch

    import openqml as oq
    from openqml.interfaces.torch import QuantumLayer

    torch.set_num_threads(1)
    source = args.source.resolve()
    assert Path(oq.__file__).resolve().is_relative_to(source)
    digest = hashlib.sha256()
    for p in sorted((source / "openqml").rglob("*.py")):
        digest.update(
            p.relative_to(source / "openqml").as_posix().encode() + b"\0" + p.read_bytes()
        )
    feature, weight = oq.Parameter("feature"), oq.Parameter("weight")
    c = oq.Circuit(2).ry(feature, 0).measure(0, 0)
    for _ in range(args.length):
        c = c.if_else(
            oq.ClassicalCondition((0,), 1),
            oq.Circuit(2).ry(weight * weight, 1).with_global_phase(weight / 17),
            oq.Circuit(2).rx(weight / 8, 1),
        )
    before = oq.circuit_stats(c)
    start = time.perf_counter()
    if args.mode == "simplify":
        c = oq.simplify(c)
    elif args.mode == "optimize":
        c = oq.optimize(c).circuit
    compile_seconds = time.perf_counter() - start
    layer = QuantumLayer(c, ["feature"], ["weight"], dtype=torch.float64)
    with torch.no_grad():
        layer.weights.fill_(0.23)
    x = torch.linspace(-0.7, 0.7, 16, dtype=torch.float64).reshape(16, 1).requires_grad_()

    def run():
        layer.zero_grad(set_to_none=True)
        x.grad = None
        output = layer(x)
        loss = output.square().mean()
        loss.backward()
        return output.detach().numpy(), layer.weights.grad.detach().numpy(), x.grad.detach().numpy()

    output, dw, dx = run()
    xv, wv, n = x.detach().numpy()[:, 0], 0.23, args.length
    prob = np.sin(xv / 2) ** 2
    a, b = n * wv * wv, n * wv / 8
    z = prob * np.cos(a) + (1 - prob) * np.cos(b)
    expected = np.stack((np.cos(xv), z), axis=-1)
    expected_dx = (np.cos(xv) * -np.sin(xv) + z * (np.cos(a) - np.cos(b)) * np.sin(xv) / 2) / len(
        xv
    )
    expected_dw = np.sum(
        z * (-prob * np.sin(a) * 2 * n * wv - (1 - prob) * np.sin(b) * n / 8)
    ) / len(xv)
    np.testing.assert_allclose(output, expected, atol=3e-12)
    np.testing.assert_allclose(dw, [expected_dw], atol=2e-10)
    np.testing.assert_allclose(dx[:, 0], expected_dx, atol=3e-12)
    run()
    times = []
    for _ in range(5):
        start = time.perf_counter()
        run()
        times.append(time.perf_counter() - start)
    return {
        "version": oq.__version__,
        "import_path": oq.__file__,
        "source_sha256": digest.hexdigest(),
        "mode": args.mode,
        "length": n,
        "batch": 16,
        "dtype": "float64",
        "device": "cpu",
        "torch_threads": 1,
        "compile_seconds": compile_seconds,
        "median_training_seconds": statistics.median(times),
        "repeats": 5,
        "before": asdict(before),
        "after": asdict(oq.circuit_stats(c)),
        "max_output_error": float(np.max(np.abs(output - expected))),
        "max_input_gradient_error": float(np.max(np.abs(dx[:, 0] - expected_dx))),
        "weight_gradient_error": float(abs(dw[0] - expected_dw)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results_optimization.json"))
    parser.add_argument("--mode", choices=("original", "simplify", "optimize"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--length", type=int)
    args = parser.parse_args()
    if args.mode:
        print(json.dumps(worker(args)))
        return
    current = Path(__file__).resolve().parents[1] / "src"
    results = []
    for length in (8, 64, 256):
        cases = [(current, "original"), (current, "optimize")]
        if args.baseline_source:
            cases = [
                (args.baseline_source.resolve(), "original"),
                (args.baseline_source.resolve(), "simplify"),
            ] + cases
        rows = []
        for source, mode in cases:
            env = dict(
                os.environ,
                PYTHONPATH=str(source),
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--source",
                str(source),
                "--mode",
                mode,
                "--length",
                str(length),
            ]
            result = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
            rows.append(json.loads(result.stdout))
        results.append(
            {
                "length": length,
                "results": rows,
                "baseline_over_optimized": rows[0]["median_training_seconds"]
                / rows[-1]["median_training_seconds"],
            }
        )
    payload = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scope": "warm forward/backward CPU microbenchmark with separate compile cost; "
        "no RSS/GPU or whole-task claim",
        "cases": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "ratios": [r["baseline_over_optimized"] for r in results]}
        )
    )


if __name__ == "__main__":
    main()
