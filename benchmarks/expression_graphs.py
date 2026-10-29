"""Compare shared expression workloads and ordinary controls in isolated CPU workers.

Timings include complete requested operations. Hybrid cases run a classical
encoder, native quantum layer and classical head through loss and gradients.
JIT compilation is reported separately; parameters are fixed between calls.
Every timed result is checked, including against the previous source tree.
These synthetic shared graphs do not predict a universal training speedup.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from planned_execution import source_hash

CASES = {
    "evaluate": 14,
    "bind": 8,
    "substitute": 9,
    "differentiate": 7,
    "torch-hybrid": 8,
    "jax-hybrid": 8,
    "chain-evaluate": 24,
    "affine-evaluate": 0,
    "affine-bind": 0,
    "torch-control": 0,
    "jax-control": 0,
}


def worker(args):
    import openqml as oq

    assert Path(oq.__file__).resolve().is_relative_to(args.source.resolve())
    begin = time.perf_counter()
    depth = CASES[args.case]
    affine = "affine" in args.case or "control" in args.case
    x, w = oq.Parameter("x"), oq.Parameter("w")
    angle = x + w + 0.1 if affine else x * w + 0.1
    for _ in range(depth):
        angle = angle.sin() if args.case == "chain-evaluate" else (angle + angle).sin()
    counts = {}

    if args.case.startswith(("torch-", "jax-")):
        rng = np.random.default_rng(71)
        initial = {
            "encoder": rng.normal(size=(2, 1)) * 0.2,
            "bias": np.array([0.1]),
            "quantum": np.array([0.4]),
            "head": rng.normal(size=(2, 1)) * 0.2,
            "offset": np.array([0.1]),
        }
        raw = rng.normal(size=(32, 2))
        circuit = oq.Circuit(2).ry(angle, 0).rx(angle, 1).cx(0, 1).ry(angle, 0).rz(angle, 1)
        if args.case.startswith("torch-"):
            import torch

            torch.set_num_threads(1)
            p = {
                k: torch.tensor(v, dtype=torch.float64, requires_grad=True)
                for k, v in initial.items()
            }
            inputs = torch.tensor(raw, dtype=torch.float64)
            layer = oq.torch.QuantumLayer(circuit, ["x"], ["w"], dtype=torch.float64)

            def run():
                encoded = torch.tanh(inputs @ p["encoder"] + p["bias"])
                output = layer.functional(p["quantum"], encoded) @ p["head"] + p["offset"]
                loss = output.square().mean()
                grads = torch.autograd.grad(loss, tuple(p.values()))
                return np.concatenate(
                    [
                        loss.detach().numpy().reshape(-1),
                        *(g.detach().numpy().ravel() for g in grads),
                    ]
                )

            module = torch
        else:
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            p = {k: jnp.asarray(v) for k, v in initial.items()}
            inputs = jnp.asarray(raw)
            layer = oq.jax.QuantumLayer(circuit, ["x"], ["w"])

            def loss(p):
                encoded = jnp.tanh(inputs @ p["encoder"] + p["bias"])
                output = layer(p["quantum"], encoded) @ p["head"] + p["offset"]
                return jnp.mean(output**2)

            compiled = jax.jit(jax.value_and_grad(loss))

            def run():
                value, grads = compiled(p)
                return np.concatenate(
                    [
                        np.asarray(value).reshape(-1),
                        *(np.asarray(grads[k]).ravel() for k in initial),
                    ]
                )

            module = jnp
        sine, calls = module.sin, []

        def counted(value):
            calls.append(None)
            return sine(value)

        with patch.object(module, "sin", counted):
            reference = run()
        counts["warmup_sin_calls"] = len(calls)
    else:
        values = {"x": np.linspace(-0.3, 0.4, 64), "w": 0.4}
        scalars = {"x": 0.3, "w": 0.4}
        if args.case in {"bind", "affine-bind"}:
            circuit = oq.Circuit(
                2, tuple(oq.Operation("ry", (i % 2,), params=(angle,)) for i in range(32))
            ).with_global_phase(angle)

            def run():
                bound = circuit.bind(scalars)
                return np.asarray(
                    [
                        bound.global_phase.constant,
                        *(op.params[0].constant for op in bound.operations),
                    ]
                )
        elif args.case == "substitute":

            def run():
                target = angle.substitute(
                    {"x": oq.Parameter("feature"), "w": oq.Parameter("weight")}
                )
                return np.asarray(target.evaluate({"feature": 0.3, "weight": 0.4}))
        elif args.case == "differentiate":

            def run():
                return np.asarray(angle.derivative("w").evaluate(scalars))
        else:

            def run():
                return np.asarray(angle.evaluate(values))

        reference = run()
    preparation = time.perf_counter() - begin
    samples = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        actual = run()
        elapsed = time.perf_counter() - start
        np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-10)
        samples.append(elapsed)

    if not args.case.startswith(("torch-", "jax-")):
        sine, constructor = np.sin, oq.Expression.__post_init__
        counts = dict(sin_calls=0, constructed_expressions=0)

        def counted(value):
            counts["sin_calls"] += 1
            return sine(value)

        def constructed(value):
            counts["constructed_expressions"] += 1
            return constructor(value)

        with (
            patch.object(np, "sin", counted),
            patch.object(oq.Expression, "__post_init__", constructed),
        ):
            np.testing.assert_allclose(run(), reference, atol=1e-10, rtol=1e-10)
    return {
        "case": args.case,
        "depth": depth,
        "source_sha256": source_hash(args.source),
        "preparation_seconds": preparation,
        "median_seconds": statistics.median(samples),
        "samples_seconds": samples,
        "counts": counts,
        "values": np.asarray(reference).ravel().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--case", choices=tuple(CASES))
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.worker:
        sys.path.insert(0, str(args.source.resolve()))
        print(json.dumps(worker(args), allow_nan=False))
        return
    if args.baseline_source is None or args.output is None:
        parser.error("--baseline-source and --output are required")
    results = []
    for case in [args.case] if args.case else CASES:
        runs = []
        for source in (args.baseline_source, args.source):
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--case",
                    case,
                    "--source",
                    str(source.resolve()),
                    "--repetitions",
                    str(args.repetitions),
                ],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            runs.append(json.loads(completed.stdout))
        before, after = (np.asarray(run.pop("values")) for run in runs)
        np.testing.assert_allclose(before, after, atol=1e-9, rtol=1e-9)
        results.append(
            dict(
                case=case,
                runs=runs,
                speedup=runs[0]["median_seconds"] / runs[1]["median_seconds"],
                max_absolute_error=float(np.max(np.abs(before - after))),
            )
        )
        args.output.write_text(json.dumps({"results": results}, indent=2) + "\n")
        print(json.dumps({k: v for k, v in results[-1].items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    main()
