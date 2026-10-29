"""Reproduce selected geometry, measurement, shadow and MPS regression cases."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import openqml as oq
from openqml.mps import _layout
from openqml.shadows import _dual, _frame, _pauli_basis


def audit():
    rows = {}
    circuit = oq.CircuitBuilder(3)
    for j in range(12):
        circuit.ry(oq.Parameter(f"w{j}"), j % 3).cx(j % 3, (j + 1) % 3)
    circuit = circuit.build()
    bindings = {name: 0.1 + i * 0.02 for i, name in enumerate(circuit.parameters)}
    metric = oq.quantum_metric(circuit)
    gradient = np.linspace(-0.1, 0.3, len(metric.parameters))
    started = time.perf_counter()
    solution = metric.natural_gradient(bindings, gradient, damping=0.01)
    elapsed = time.perf_counter() - started
    matrix = metric.matrix(bindings)
    expected = np.linalg.solve(matrix + 0.01 * np.eye(len(gradient)), gradient)
    error = float(np.max(np.abs(expected - solution.direction)))
    assert solution.converged and error < 1e-8
    rows["geometry"] = {
        "parameters": len(gradient),
        "metric_products": solution.metric_products,
        "solve_seconds": elapsed,
        "explicit_solve_max_error": error,
        "residual_norm": solution.residual_norm,
        "matrix_constructed_by_solver": False,
        "scope": "pure_state_CPU_statevector_not_hardware_or_mixed_state_metric",
    }
    observables = (oq.PauliString("X"), oq.PauliString("Z"))
    source = [oq.Circuit(1)]
    fixed = oq.estimate(source, observables, shots=6000, seed=11)
    adaptive = oq.estimate_adaptive(source, observables, shots=6000, pilot_shots=64, seed=11)
    fixed_counts = {
        g.basis: n for g, n in zip(fixed.plan.groups, fixed.shots_per_group, strict=True)
    }
    adaptive_counts = {
        g.basis: n
        for g, n in zip(
            adaptive.production.plan.groups, adaptive.production.shots_per_group, strict=True
        )
    }
    rows["adaptive_measurement"] = {
        "total_budget": adaptive.shots_per_circuit,
        "pilot_per_group": 64,
        "fixed_group_shots": fixed_counts,
        "production_group_shots": adaptive_counts,
        "conditional_true_total_variance_fixed": 1 / fixed_counts["X"],
        "conditional_true_total_variance_adaptive": 1 / adaptive_counts["X"],
        "pilot_reused_in_reported_mean": False,
        "scope": "Z_eigenstate_case_not_a_universal_variance_improvement",
    }
    state = np.array([1, 0, 0, 1], complex) / np.sqrt(2)
    rho = np.outer(state, state.conj())
    probabilities = _frame(2) @ np.einsum("pij,ji->p", _pauli_basis(2), rho).real
    canonical, _ = _dual(np.eye(4) / 4)
    local, residual = _dual(0.95 * rho + 0.05 * np.eye(4) / 4)
    old = float(probabilities @ np.square(canonical[:, 5]) - 1)
    new = float(probabilities @ np.square(local[:, 5]) - 1)
    acquired = oq.shadow_estimate(
        oq.Circuit(2).h(0).cx(0, 1),
        (oq.PauliString("XX"), oq.PauliString("ZZ")),
        shots=1200,
        pilot_shots=200,
        seed=7,
    )
    rows["correlated_shadows"] = {
        "canonical_XX_single_shot_variance": old,
        "regularized_reference_XX_single_shot_variance": new,
        "dual_reconstruction_residual": residual,
        "production_values": acquired.values.tolist(),
        "production_standard_errors": acquired.standard_errors.tolist(),
        "pilot_selected_blocks": acquired.blocks,
        "scope": (
            "exact_variance_case_uses_known_Bell_reference; acquisition_uses_independent_pilot"
        ),
        "full_paper_benchmarks_reproduced": False,
    }
    builder = oq.CircuitBuilder(10)
    for layer in range(4):
        for q in range(10):
            builder.ry(0.07 * (q + 1), q).rz(0.1 * layer, q)
        for q in range(9):
            builder.crx(0.08, q, q + 1)
    source = builder.build()
    observables = (oq.PauliString("ZZIIIIIIII"), oq.PauliString("XIXIIIIIII"))
    report = oq.TruncatedMPSBackend(max_bond_dimension=2, max_state_error=2).evaluate(
        source, observables
    )
    exact = oq.expectations(source, observables)
    errors = np.abs(report.values - exact)
    assert np.all(errors <= report.observable_error_bounds + 1e-12)
    shape, reason = _layout(source, 2, 1_000_000)
    rows["truncated_mps"] = {
        "qubits": 10,
        "bond_cap": 2,
        "actual_max_bond": report.max_bond_dimension,
        "state_distance_bound": report.state_distance_bound,
        "readout_errors": errors.tolist(),
        "readout_bounds": report.observable_error_bounds.tolist(),
        "workspace_admission_bytes": report.workspace_bytes,
        "full_SVD_cap_2_admitted": bool(shape and shape["analysis_complete"]),
        "full_SVD_cap_2_reason": reason,
        "floating_point_roundoff_included_in_bound": False,
        "scope": "weakly_entangling_nearest_neighbor_unitary_case",
    }
    return {
        "schema": "openqml.research_methods_audit",
        "openqml": oq.__version__,
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "scope": "selected_reproducible_regressions_not_industry_leadership_or_exclusive_moat",
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/research_methods_cpu.json")
    )
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
