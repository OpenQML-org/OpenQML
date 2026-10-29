"""Reproduce bounded hybrid equivalence, mutation and factorization evidence."""

import argparse
import json
import platform
import time
from pathlib import Path

import openqml as oq


def cases():
    p, q = oq.Parameter("p"), oq.Parameter("q")
    yield "strict_operator_identity", oq.Circuit(1).h(0).z(0).h(0), oq.Circuit(1).x(0), True
    yield (
        "strict_global_phase_mutation",
        oq.Circuit(1).h(0).z(0).h(0),
        oq.Circuit(1).x(0).with_global_phase(0.125),
        False,
    )
    reset = oq.Circuit(1).reset(0).measure(0, 0)
    feedback = (
        oq.Circuit(1)
        .measure(0, 0)
        .if_else(oq.ClassicalCondition((0,), 1), oq.Circuit(1).x(0))
        .measure(0, 0)
    )
    yield "reset_vs_overwritten_feedback", reset, feedback, True
    yield "final_record_mutation", reset, oq.Circuit(1).measure(0, 0).reset(0), False
    yield (
        "symbolic_stochastic_composition",
        oq.Circuit(1).bit_flip(p, 0).bit_flip(q, 0),
        oq.Circuit(1).bit_flip(p + q - 2 * p * q, 0),
        True,
    )
    prefix = oq.Circuit(2).h(0).measure(0, 0)
    condition = oq.ClassicalCondition((0,), 1)
    a = prefix.if_else(condition, oq.Circuit(2).rx(p, 1))
    b = prefix.if_else(condition, oq.Circuit(2).h(1).rz(p, 1).h(1))
    yield "symbolic_controlled_rotation", a, b, True
    yield (
        "symbolic_rotation_sign_mutation",
        a,
        prefix.if_else(condition, oq.Circuit(2).rx(-p, 1)),
        False,
    )
    yield "coherence_mutation", oq.Circuit(1), oq.Circuit(1).phase_flip(0.5, 0), False
    left, right = oq.CircuitBuilder(1024), oq.CircuitBuilder(1024)
    for wire in range(128):
        left.reset(wire).x(wire)
        right.x(wire).reset(wire).x(wire)
    yield "1024_wires_128_independent_single_wire_components", left.build(), right.build(), True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/semantic_kernels_cpu.json")
    )
    args = parser.parse_args()
    rows = []
    for name, source, target, expected in cases():
        started = time.perf_counter()
        proof = oq.prove_equivalence(source, target)
        elapsed = time.perf_counter() - started
        assert proof.proven == expected, name
        rows.append(
            {
                "case": name,
                "expected_proven": expected,
                "elapsed_seconds": elapsed,
                "proof": proof.to_dict(),
            }
        )
    report = {
        "schema": "openqml.semantic_kernel_audit",
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "quantum_sdk_required": False,
        "numeric_simulation_used_for_proofs": False,
        "scope": "selected_regression_cases_not_general_compiler_coverage",
        "large_wire_case": (
            "128 independent active one-wire factors and 896 idle wires; "
            "not a connected 1024-qubit proof"
        ),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(rows), "passed": True, "output": str(args.output)}))


if __name__ == "__main__":
    main()
