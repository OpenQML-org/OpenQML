"""Reproducible compiler and algorithm-selection audits in isolated CPU workers.

Compiler checks compare complete maps at explicit bindings against native SDK
execution, separately from exact algebraic certificates. Large non-Clifford
MPS readouts use an independently implemented Aer reference, small dense
checks, and inverse echoes. These families do not imply general large-scale
simulation. Dense execution is measured only if observed memory permits it.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

from _memory import measure_peak
from planned_execution import source_hash


def random_unitary(oq, seed):
    rng = random.Random(seed)
    builder = oq.CircuitBuilder(3)
    theta = oq.Parameter("theta")
    for _ in range(24):
        a, b = rng.sample(range(3), 2)
        name = rng.choice(("h", "x", "ry", "rz", "rx", "cx", "cz", "swap", "cp"))
        if name in {"ry", "rz", "rx"}:
            getattr(builder, name)(theta * rng.choice((-0.5, 0.5, 1.0)), a)
        elif name == "cp":
            builder.cp(theta / 2, a, b)
        elif name in {"cx", "cz", "swap"}:
            getattr(builder, name)(a, b)
        else:
            getattr(builder, name)(a)
    return builder.build()


def chain(oq, width, depth=4, seed=7):
    rng = random.Random(seed)
    builder = oq.CircuitBuilder(width)
    for layer in range(depth):
        for q in range(width):
            builder.ry(rng.uniform(-1, 1), q).rz(rng.uniform(-1, 1), q)
        for q in range(layer % 2, width - 1, 2):
            builder.cx(q, q + 1)
    return builder.build()


def compiler_audit(oq, args):
    results, maximum = [], 0.0
    for seed in range(args.seeds):
        source = random_unitary(oq, seed)
        for target in ("ir", "json", "qiskit", "pennylane", "cirq", "openqasm2", "openqasm3"):
            bound_only = target in {"pennylane", "openqasm2"}
            plan = oq.plan_conversion(
                source, target=target, bindings={"theta": 0.319} if bound_only else None
            )
            mode = "roundtrip" if target in {"ir", "json"} else "native"
            errors, checks = [], 0
            for value in (0.319,) if bound_only else (0.0, 0.319, -0.73):
                check = plan.verify(
                    None if bound_only else {"theta": value},
                    mode=mode,
                    max_qubits=3,
                    atol=1e-12,
                    rtol=0,
                )
                check.raise_for_errors()
                errors.append(check.max_absolute_error)
                checks += 1
            maximum = max(maximum, *errors)
            try:
                certificate = plan.certify().proof
                proof, reason = certificate.method, None
            except oq.ConversionError as error:
                proof, reason = "inconclusive", [note.code for note in error.diagnostics]
            results.append(
                {
                    "seed": seed,
                    "target": target,
                    "input_parameters": "bound" if bound_only else "symbolic",
                    "comparison": mode,
                    "complete_input_basis_checks": checks,
                    "max_absolute_error": max(errors),
                    "algebraic_proof": proof,
                    "proof_diagnostics": reason,
                }
            )
    theta = oq.Parameter("theta")
    a, b = oq.CircuitBuilder(1024), oq.CircuitBuilder(1024)
    for q in range(256):
        a.rzz(theta / 2, q, 1023 - q)
        b.cx(q, 1023 - q).rz(theta / 2, 1023 - q).cx(q, 1023 - q)
    large = oq.prove_equivalence(a.build(), b.build())
    large.raise_for_errors()
    wrong = oq.prove_equivalence(oq.Circuit(1).ry(theta, 0), oq.Circuit(1).ry(-theta, 0))
    assert not wrong.proven
    return {
        "scope": "supported_IR_semantics_not_arbitrary_Python_translation",
        "coverage": oq.semantic_coverage(),
        "randomized_conversions": results,
        "max_absolute_error": maximum,
        "algebraically_proven": sum(row["algebraic_proof"] != "inconclusive" for row in results),
        "complete_input_basis_checks": sum(row["complete_input_basis_checks"] for row in results),
        "independent_native_checks": sum(
            row["complete_input_basis_checks"] for row in results if row["comparison"] == "native"
        ),
        "inconclusive": sum(row["algebraic_proof"] == "inconclusive" for row in results),
        "large_symbolic_phase_proof": large.to_dict(),
        "wrong_sign_not_certified": True,
    }


def execution_audit(oq, args):
    import numpy as np

    if args.case == "mps":
        circuit = chain(oq, 128)
        observables = (
            oq.PauliString("Z" * 128),
            oq.PauliString("X" * 128),
            oq.PauliString("Z" + "I" * 127),
        )
        start = time.perf_counter()
        plan = oq.plan_execution(circuit, observables, max_memory_mb=128, max_pauli_terms=8)
        preparation = time.perf_counter() - start
        assert plan.selected_method == "mps"
        reference = oq.expectations(circuit, observables, backend="aer-mps")
        small = chain(oq, 6)
        small_obs = (oq.PauliString("Z" * 6), oq.PauliString("X" * 6))
        np.testing.assert_allclose(
            oq.expectations(small, small_obs, backend="mps"),
            oq.expectations(small, small_obs),
            atol=1e-12,
            rtol=0,
        )
        echo_source = chain(oq, 128, 2)
        echo = echo_source.compose(oq.inverse(echo_source))
        echo_obs = (oq.PauliString("Z" * 128), oq.PauliString("X" * 128))
        echo_plan = oq.plan_execution(echo, echo_obs, method="mps", max_memory_mb=128)
        np.testing.assert_allclose(echo_plan.evaluate(), (1, 0), atol=1e-10, rtol=0)
        verification = {
            "independent_reference": "Aer_MPS_without_intentional_truncation",
            "small_dense_qubits": 6,
            "large_inverse_echo_qubits": 128,
            "nonclifford_depth": 4,
            "global_readouts": 2,
        }
    elif args.case == "stabilizer":
        from openqml.semantics import _conjugate

        width, half, rng = 1024, 512, random.Random(31)
        builder = oq.CircuitBuilder(width)
        for q in range(half):
            builder.h(q).cx(q, q + half)
        for _ in range(1000):
            offset = rng.choice((0, half))
            a, b = (q + offset for q in rng.sample(range(half), 2))
            builder.h(a).s(b).cx(a, b)
        circuit = builder.build()
        observables = []
        for q in (0, 7, 128, 255, 512, 519, 640, 767):
            x, z, phase = 0, 1 << q, 0
            for operation in circuit.operations:
                x, z, phase = _conjugate(x, z, phase, operation.name, operation.qubits)
            word = "".join(
                "Y" if x & z & (1 << i) else "X" if x & (1 << i) else "Z" if z & (1 << i) else "I"
                for i in range(width)
            )
            coefficient = 1 if (phase - (x & z).bit_count()) % 4 == 0 else -1
            observables.append(oq.PauliString(word, coefficient))
        start = time.perf_counter()
        plan = oq.plan_execution(circuit, observables, max_memory_mb=128)
        preparation = time.perf_counter() - start
        assert plan.selected_method == "stabilizer"
        reference = oq.PauliPlan(circuit, tuple(observables)).evaluate()
        np.testing.assert_allclose(reference, np.ones(8), atol=1e-12, rtol=0)
        verification = {
            "independent_reference": "symbolic_Pauli_propagation_and_stabilizer_identities",
            "analytic_expectations": [1] * 8,
            "log2_middle_cut_Schmidt_rank": 512,
            "entanglement_reason": "crossed_Bell_pairs_followed_by_local_Cliffords_on_each_half",
        }
    else:
        from openqml.adaptive import _available_memory

        requested = args.dense_qubits
        # This audit avoids forcing an explicit budget beyond observed space.
        free = _available_memory()
        capacity = (min(1024 * 2**20, free // 4) // 64).bit_length() - 1
        width = min(requested, capacity)
        if width < 21:
            return {
                "status": "skipped",
                "reason": "Observed memory cannot safely demonstrate a width above twenty.",
                "available_bytes": free,
            }
        circuit = oq.Circuit(width).ry(0.37, 0).x(width - 1)
        observables = (
            oq.PauliString("Z" + "I" * (width - 1)),
            oq.PauliString("I" * (width - 1) + "Z"),
        )
        start = time.perf_counter()
        plan = oq.plan_execution(
            circuit, observables, method="statevector", max_memory_mb=(64 * (1 << width)) / 2**20
        )
        preparation = time.perf_counter() - start
        reference = np.array([np.cos(0.37), -1])
        verification = {
            "independent_reference": "analytic_product_state_readouts",
            "requested_qubits": requested,
            "dense_width_is_memory_derived": True,
            "available_bytes_before_run": free,
        }
    # Record cold first execution separately from the reset-window warm call.
    start = time.perf_counter()
    first = plan.evaluate()
    cold = time.perf_counter() - start
    tolerance = 1e-10 if args.case == "mps" else 1e-12
    np.testing.assert_allclose(first, reference, atol=tolerance, rtol=0)
    start = time.perf_counter()
    values, memory = measure_peak(plan.evaluate)
    elapsed = time.perf_counter() - start
    np.testing.assert_allclose(values, reference, atol=tolerance, rtol=0)
    return {
        "status": "passed",
        "num_qubits": circuit.num_qubits,
        "operations": len(circuit.operations),
        "readouts": len(observables),
        "preparation_seconds": preparation,
        "cold_first_execution_seconds": cold,
        "warm_execution_seconds": elapsed,
        "memory": memory,
        "selection": plan.to_dict(),
        "max_absolute_error": float(np.max(np.abs(values - reference))),
        "comparison_atol": tolerance,
        "verification": verification,
    }


def worker(args):
    import openqml as oq

    assert Path(oq.__file__).resolve().is_relative_to(args.source.resolve())
    result = compiler_audit(oq, args) if args.case == "compiler" else execution_audit(oq, args)
    return {"case": args.case, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1] / "src")
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--dense-qubits", type=int, default=24)
    parser.add_argument("--case", choices=("compiler", "mps", "stabilizer", "dense"))
    args = parser.parse_args()
    if args.seeds < 1 or not 1 <= args.dense_qubits <= 30:
        parser.error("Use positive seeds and a dense request between one and thirty.")
    if args.case:
        print(json.dumps(worker(args), allow_nan=False))
        return
    runs = []
    for case in ("compiler", "stabilizer", "mps", "dense"):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--case",
            case,
            "--source",
            str(args.source),
            "--seeds",
            str(args.seeds),
            "--dense-qubits",
            str(args.dense_qubits),
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=os.environ
            | {"PYTHONPATH": str(args.source), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
        runs.append(json.loads(completed.stdout))
    result = {
        "scope": "bounded_compiler_and_representation_specific_execution_evidence",
        "source_sha256": source_hash(args.source),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "numpy": importlib.metadata.version("numpy"),
            "qiskit_aer": importlib.metadata.version("qiskit-aer"),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "torch",
                    "jax",
                    "qiskit",
                    "qiskit-aer",
                    "pennylane",
                    "cirq-core",
                    "sympy",
                )
            },
            "threads": {"OMP_NUM_THREADS": 1, "OPENBLAS_NUM_THREADS": 1},
        },
        "accelerator_execution_checked": False,
        "runs": runs,
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
