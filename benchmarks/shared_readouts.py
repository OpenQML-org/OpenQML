"""Compare exact CPU readout kernels in fresh, single-threaded source workers.

Measurements cover readout evaluation on an already prepared state, not complete
training or provider execution. Linux reset-window RSS includes cold readout
constants. Reference allocation and validation occur outside the measured call.
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
from planned_execution import source_hash


def worker(args):
    import numpy as np

    import openqml as oq
    import openqml.simulator as sim

    assert Path(oq.__file__).resolve().is_relative_to(args.source.resolve())
    rng = np.random.default_rng(29)
    state = rng.normal(size=(args.batch, 2**args.qubits)) + 1j * rng.normal(
        size=(args.batch, 2**args.qubits)
    )
    state /= np.linalg.norm(state, axis=-1, keepdims=True)
    if args.case == "sparse":
        words = ["I" * q + "Z" + "I" * (args.qubits - q - 1) for q in range(args.qubits)]
    elif args.case == "diagonal":
        masks = rng.choice(2**args.qubits, size=args.readouts, replace=False)
        words = [
            "".join("Z" if mask & (1 << (args.qubits - q - 1)) else "I" for q in range(args.qubits))
            for mask in masks
        ]
    else:
        unique = ["".join(rng.choice(list("IXYZ"), size=args.qubits)) for _ in range(8)]
        words = [unique[i % len(unique)] for i in range(args.readouts)]
    observables = tuple(
        oq.PauliString(word, (-1) ** i * (1 + i / 100)) for i, word in enumerate(words)
    )
    ops = sim._TensorOps()

    def run():
        if hasattr(sim, "_expectations_from_state"):
            return sim._expectations_from_state(state, observables, args.qubits, ops)
        return np.stack(
            [sim._expectation_from_state(state, ob, args.qubits, ops) for ob in observables],
            axis=-1,
        )

    result, memory = measure_peak(run)
    matrices = {
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    reference, evaluated = [], {}
    for observable in observables:
        if observable.paulis not in evaluated:
            vector = state.reshape((args.batch,) + (2,) * args.qubits)
            for q, character in enumerate(observable.paulis):
                if character != "I":
                    vector = np.moveaxis(vector, q + 1, -1) @ matrices[character].T
                    vector = np.moveaxis(vector, -1, q + 1)
            evaluated[observable.paulis] = (
                (state.conj() * vector.reshape(state.shape)).sum(axis=-1).real
            )
        reference.append(observable.coefficient * evaluated[observable.paulis])
    reference = np.stack(reference, axis=-1)
    np.testing.assert_allclose(result, reference, atol=2e-12, rtol=2e-12)
    samples = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        actual = run()
        samples.append(time.perf_counter() - start)
        np.testing.assert_allclose(actual, reference, atol=2e-12, rtol=2e-12)
    return {
        "case": args.case,
        "qubits": args.qubits,
        "batch": args.batch,
        "readouts": len(observables),
        "unique_words": len(set(words)),
        "state_bytes": state.nbytes,
        "warm_median_seconds": statistics.median(samples),
        "warm_samples_seconds": samples,
        "max_absolute_error": float(np.max(np.abs(result - reference))),
        "cold_readout_memory": memory,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--case", choices=("diagonal", "sparse", "duplicate"), default="diagonal")
    parser.add_argument("--qubits", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--readouts", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.qubits <= 20 or min(args.batch, args.readouts, args.repetitions) < 1:
        parser.error("Use one to twenty qubits and positive batch/readout/repetition counts.")
    if args.case == "diagonal" and args.readouts > 2**args.qubits:
        parser.error("Diagonal readouts must not exceed the number of distinct words.")
    if args.worker:
        sys.path.insert(0, str(args.source.resolve()))
        print(json.dumps(worker(args), allow_nan=False))
        return
    runs = []
    for source in ([args.baseline_source] if args.baseline_source else []) + [args.source]:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--source",
            str(source.resolve()),
        ]
        for name in ("case", "qubits", "batch", "readouts", "repetitions"):
            command.extend((f"--{name}", str(getattr(args, name))))
        environment = os.environ | {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        completed = subprocess.run(
            command, capture_output=True, text=True, check=True, env=environment
        )
        runs.append({"source_sha256": source_hash(source), **json.loads(completed.stdout)})
    payload = json.dumps({"cpu_only": True, "runs": runs}, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
