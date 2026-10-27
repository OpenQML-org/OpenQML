"""Compare full hybrid block products against previous source in fresh CPU workers.

The encoder, numerical quantum layer and head all participate in JVP/VJP
products. Shared-point blocks expose repeated quantum work; distinct-point
blocks are controls. This benchmarks parameter shifts, not the fastest NumPy
algorithm: unitary simulation can alternatively use the existing adjoint engine.
Preparation/JIT is separate and every timed product is checked.
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


def leaves(value):
    if isinstance(value, dict):
        return [leaf for key in sorted(value) for leaf in leaves(value[key])]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in leaves(item)]
    return [value]


def worker(args):
    import openqml as oq
    import openqml.gradients as gradients

    assert Path(oq.__file__).resolve().is_relative_to(args.source.resolve())
    rng = np.random.default_rng(71)
    spec = oq.data_reuploading(2, 2)
    initial = {
        "encoder": rng.normal(size=(3, 2)) * 0.2,
        "bias": rng.normal(size=2) * 0.1,
        "quantum": rng.normal(size=len(spec.weight_parameters)) * 0.3,
        "head": rng.normal(size=(2, 1)) * 0.2,
        "offset": np.zeros(1),
    }
    features = rng.normal(size=(args.batch, 3))
    directions = {
        key: rng.normal(size=(args.directions,) + value.shape) * 0.1
        for key, value in initial.items()
    }
    points = {
        key: np.stack([value + 0.01 * index for index in range(args.directions)])
        for key, value in initial.items()
    }
    cotangents = rng.normal(size=(args.directions, args.batch, 1)) * 0.2
    profiling = False
    counts = dict(
        shifted_circuits=0, submissions=0, max_submitted_circuits=0, symbolic_derivatives=0
    )
    execute, derivative = gradients.expectations_batch, oq.Expression.derivative

    def counted(circuits, *a, **kw):
        if profiling:
            counts["shifted_circuits"] += len(circuits)
            counts["submissions"] += 1
            counts["max_submitted_circuits"] = max(counts["max_submitted_circuits"], len(circuits))
        return execute(circuits, *a, **kw)

    def differentiated(expr, name):
        if profiling:
            counts["symbolic_derivatives"] += 1
        return derivative(expr, name)

    with (
        patch.object(gradients, "expectations_batch", counted),
        patch.object(oq.Expression, "derivative", differentiated),
    ):
        if args.interface == "torch":
            import torch

            from openqml.interfaces.torch import QuantumLayer

            torch.set_num_threads(1)
            api, array = torch.func, lambda value: torch.tensor(value, dtype=torch.float64)
            axis, tanh = "in_dims", torch.tanh

            def host(value):
                return value.detach().numpy()

            layer_options = dict(dtype=torch.float64)
        else:
            import jax
            import jax.numpy as jnp

            from openqml.interfaces.jax import QuantumLayer

            jax.config.update("jax_enable_x64", True)
            api, array, axis, tanh, host = jax, jnp.asarray, "in_axes", jnp.tanh, np.asarray
            layer_options = {}
        start = time.perf_counter()
        layer = QuantumLayer.from_spec(
            spec,
            backend=args.backend,
            gradient_method="parameter-shift",
            gradient_chunk_size=args.chunk,
            **layer_options,
        )
        params = {key: array(value) for key, value in initial.items()}
        inputs = array(features)
        block_directions = {key: array(value) for key, value in directions.items()}
        block_points = {key: array(value) for key, value in points.items()}
        block_cotangents = array(cotangents)

        def model(p):
            encoded = tanh(inputs @ p["encoder"] + p["bias"])
            quantum = (
                layer.functional(p["quantum"], encoded)
                if args.interface == "torch"
                else layer(p["quantum"], encoded)
            )
            return quantum @ p["head"] + p["offset"]

        if args.product == "jvp":

            def product(p, direction):
                return api.jvp(model, (p,), (direction,))[1]

            vectors = block_directions
        else:

            def product(p, cotangent):
                return api.vjp(model, p)[1](cotangent)[0]

            vectors = block_cotangents
        if args.points == "shared":
            evaluate = api.vmap(lambda vector: product(params, vector))
            arguments = (vectors,)
        else:
            evaluate = api.vmap(product, **{axis: (0, 0)})
            arguments = (block_points, vectors)
        if args.interface == "jax":
            evaluate = jax.jit(evaluate)

        def run():
            value = evaluate(*arguments)
            return np.concatenate([host(item).reshape(-1) for item in leaves(value)])

        checked, samples = [run()], []
        preparation = time.perf_counter() - start
        for _ in range(args.repeats):
            start = time.perf_counter()
            checked.append(run())
            samples.append(time.perf_counter() - start)
        profiling = True
        run()
    print(
        json.dumps(
            dict(
                source_sha256=source_hash(args.source),
                interface=args.interface,
                product=args.product,
                points=args.points,
                backend=args.backend,
                directions=args.directions,
                batch=args.batch,
                gradient_chunk_size=args.chunk,
                preparation_seconds=preparation,
                median_seconds=statistics.median(samples),
                samples_seconds=samples,
                **counts,
                checked_values=np.stack(checked).tolist(),
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--interface", choices=("torch", "jax", "all"), default="all")
    parser.add_argument(
        "--backend", choices=("numpy", "qiskit", "pennylane", "cirq"), default="numpy"
    )
    parser.add_argument("--product", choices=("jvp", "vjp", "all"), default="all")
    parser.add_argument("--points", choices=("shared", "unique", "all"), default="all")
    parser.add_argument("--directions", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.directions, args.batch, args.repeats) < 1 or args.chunk < 2:
        parser.error("Counts must be positive and chunk must be at least two.")
    if args.worker:
        worker(args)
        return
    report = []
    for interface in ("torch", "jax") if args.interface == "all" else (args.interface,):
        for product in ("jvp", "vjp") if args.product == "all" else (args.product,):
            for points in ("shared", "unique") if args.points == "all" else (args.points,):
                runs = []
                for source in (args.baseline_source, args.source):
                    source = source.resolve()
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        "--baseline-source",
                        str(args.baseline_source.resolve()),
                        "--source",
                        str(source),
                        "--interface",
                        interface,
                        "--backend",
                        args.backend,
                        "--product",
                        product,
                        "--points",
                        points,
                        "--directions",
                        str(args.directions),
                        "--batch",
                        str(args.batch),
                        "--chunk",
                        str(args.chunk),
                        "--repeats",
                        str(args.repeats),
                    ]
                    env = {
                        **os.environ,
                        "PYTHONPATH": os.pathsep.join(
                            (str(source), os.environ.get("PYTHONPATH", ""))
                        ),
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                    }
                    result = subprocess.run(command, env=env, text=True, capture_output=True)
                    if result.returncode:
                        raise RuntimeError(
                            f"{interface}/{product}/{points}: {result.stderr[-3000:]}"
                        )
                    runs.append(json.loads(result.stdout))
                first, second = (np.asarray(run.pop("checked_values")) for run in runs)
                np.testing.assert_allclose(second, first, atol=4e-11, rtol=4e-11)
                report.append(
                    dict(
                        interface=interface,
                        product=product,
                        points=points,
                        runs=runs,
                        speedup=runs[0]["median_seconds"] / runs[1]["median_seconds"],
                        max_absolute_error=float(np.max(np.abs(first - second))),
                    )
                )
    result = json.dumps(dict(results=report), indent=2)
    print(result)
    if args.output:
        args.output.write_text(result + "\n")


if __name__ == "__main__":
    main()
