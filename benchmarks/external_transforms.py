"""Compare mapped hybrid training with previous source in fresh CPU workers.

Each step trains a classical encoder, an ensemble of numerical quantum layers,
and a shared classical head with one SGD update. JAX compares the same vmap API;
PyTorch compares the previous manual loop with torch.func.vmap. Both sources
also run the existing explicit broadcast path as a control. Every timed loss,
gradient and optimizer trajectory is checked. Warm-up/JIT time is separate.
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
    import openqml.adjoint as adjoint
    import openqml.interfaces._spec as bridge
    from openqml.templates import data_reuploading

    assert Path(oq.__file__).resolve().is_relative_to(args.source.resolve())
    rng = np.random.default_rng(71)
    configuration = data_reuploading(4, depth=2)
    initial = {
        "encoder": rng.normal(size=(4, 4)) * 0.2,
        "bias": rng.normal(size=4) * 0.1,
        "quantum": rng.normal(size=(args.ensemble, len(configuration.weight_parameters))) * 0.3,
        "head": rng.normal(size=(4, 1)) * 0.2,
        "offset": np.zeros(1),
    }
    features, targets = rng.normal(size=(args.batch, 4)), rng.normal(size=(args.batch, 1))
    profiling = False
    counts = dict(forward_callbacks=0, reverse_callbacks=0, matrix_kernel_calls=0, max_state_rows=0)
    forward, reverse, apply = bridge.numeric_expectations, bridge.numeric_vjp, adjoint._apply_matrix

    def counted_forward(*a, **kw):
        if profiling:
            counts["forward_callbacks"] += 1
        return forward(*a, **kw)

    def counted_reverse(*a, **kw):
        if profiling:
            counts["reverse_callbacks"] += 1
        return reverse(*a, **kw)

    def counted_matrix(state, *a, **kw):
        if profiling:
            counts["matrix_kernel_calls"] += 1
            counts["max_state_rows"] = max(counts["max_state_rows"], math.prod(state.shape[:-1]))
        return apply(state, *a, **kw)

    options = dict(backend="numpy", gradient_method="adjoint", batch_chunk_size=args.chunk)
    with (
        patch.object(bridge, "numeric_expectations", counted_forward),
        patch.object(bridge, "numeric_vjp", counted_reverse),
        patch.object(adjoint, "_apply_matrix", counted_matrix),
    ):
        start = time.perf_counter()
        if args.interface == "jax":
            import jax
            import jax.numpy as jnp

            from openqml.interfaces.jax import QuantumLayer

            jax.config.update("jax_enable_x64", True)
            layer = QuantumLayer.from_spec(configuration, **options)
            params = jax.tree_util.tree_map(jnp.asarray, initial)
            inputs, target = jnp.asarray(features), jnp.asarray(targets)

            def objective(p):
                encoded = jnp.tanh(inputs @ p["encoder"] + p["bias"])
                if args.mode == "broadcast":
                    quantum = layer(p["quantum"][:, None, :], encoded[None, ...])
                else:
                    quantum = jax.vmap(layer, in_axes=(0, None))(p["quantum"], encoded)
                output = quantum @ p["head"] + p["offset"]
                return jnp.square(output - target[None, ...]).mean()

            evaluate = jax.jit(jax.value_and_grad(objective))

            def run():
                nonlocal params
                value, grads = evaluate(params)
                vector = np.concatenate(
                    [np.asarray(value).reshape(1)]
                    + [np.asarray(v).reshape(-1) for v in jax.tree_util.tree_leaves(grads)]
                )
                params = jax.tree_util.tree_map(lambda p, g: p - 0.01 * g, params, grads)
                jax.block_until_ready(params)
                return vector

        else:
            import torch

            from openqml.interfaces.torch import QuantumLayer

            torch.set_num_threads(1)
            layer = QuantumLayer.from_spec(configuration, dtype=torch.float64, **options)
            params = torch.nn.ParameterDict(
                {name: torch.nn.Parameter(torch.tensor(value)) for name, value in initial.items()}
            )
            inputs, target = torch.tensor(features), torch.tensor(targets)
            optimizer = torch.optim.SGD(params.parameters(), lr=0.01)

            def run():
                optimizer.zero_grad()
                encoded = torch.tanh(inputs @ params["encoder"] + params["bias"])
                if args.mode == "broadcast":
                    quantum = layer.functional(params["quantum"][:, None, :], encoded[None, ...])
                elif args.mode == "loop":
                    quantum = torch.stack([layer.functional(w, encoded) for w in params["quantum"]])
                else:
                    quantum = torch.func.vmap(layer.functional, in_dims=(0, None))(
                        params["quantum"], encoded
                    )
                output = quantum @ params["head"] + params["offset"]
                loss = (output - target[None, ...]).square().mean()
                loss.backward()
                vector = torch.cat(
                    [loss.detach().reshape(1)]
                    + [params[name].grad.reshape(-1) for name in sorted(params)]
                ).numpy()
                optimizer.step()
                return vector

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
                mode=args.mode,
                ensemble=args.ensemble,
                batch=args.batch,
                batch_chunk_size=args.chunk,
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
    parser.add_argument("--interface", choices=("jax", "torch", "all"), default="all")
    parser.add_argument("--ensemble", type=int, default=32)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode", choices=("mapped", "broadcast", "loop"), default="mapped", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if min(args.ensemble, args.batch, args.chunk, args.repeats) < 1:
        parser.error("Ensemble, batch, chunk and repeat counts must be positive.")
    if args.worker:
        worker(args)
        return
    report = []
    for interface in ("jax", "torch") if args.interface == "all" else (args.interface,):
        runs = []
        for label, source in (("baseline", args.baseline_source), ("current", args.source)):
            source = source.resolve()
            mode = "loop" if label == "baseline" and interface == "torch" else "mapped"
            for selected in (mode, "broadcast"):
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
                    "--mode",
                    selected,
                    "--ensemble",
                    str(args.ensemble),
                    "--batch",
                    str(args.batch),
                    "--chunk",
                    str(args.chunk),
                    "--repeats",
                    str(args.repeats),
                ]
                env = {
                    **os.environ,
                    "PYTHONPATH": str(source),
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                }
                result = subprocess.run(command, env=env, text=True, capture_output=True)
                if result.returncode:
                    raise RuntimeError(f"{label}/{interface}/{selected}: {result.stderr[-3000:]}")
                runs.append(dict(label=label, **json.loads(result.stdout)))
        reference = np.asarray(runs[0]["checked_values"])
        for run in runs:
            values = np.asarray(run.pop("checked_values"))
            np.testing.assert_allclose(values, reference, atol=3e-11, rtol=3e-11)
            run["max_absolute_error"] = float(np.max(np.abs(values - reference)))
        report.append(
            dict(
                interface=interface,
                runs=runs,
                mapped_speedup=runs[0]["median_seconds"] / runs[2]["median_seconds"],
                current_broadcast_over_mapped=runs[3]["median_seconds"] / runs[2]["median_seconds"],
            )
        )
    result = json.dumps(dict(results=report), indent=2)
    print(result)
    if args.output:
        args.output.write_text(result + "\n")


if __name__ == "__main__":
    main()
