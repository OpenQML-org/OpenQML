"""Representation-specific storage estimates, independent of code or IR size."""

import json
import math
from dataclasses import asdict, dataclass

from .backends import ExecutionOptions
from .compiler import _check_circuit_limits
from .errors import Diagnostic, ResourceError
from .ir import integer


@dataclass(frozen=True)
class ResourceEstimate:
    num_qubits: int
    operations: int
    batch_size: int
    max_qubits: int | None
    max_memory_mb: float
    statevector_log2_bytes: float
    density_log2_bytes: float
    statevector_bytes: int | None
    density_bytes: int | None
    statevector_budget_bytes: int | None
    within_statevector_policy: bool
    diagnostics: tuple[Diagnostic, ...]

    def require_statevector(self):
        if not self.within_statevector_policy:
            raise ResourceError(
                "Dense statevector policy rejects the width, budget or quantum effects."
            )

    def to_dict(self):
        return {
            "schema": "openqml.resource_estimate",
            "scope": "storage_and_policy_only",
            "dtype": "complex128",
            "budget_model": "four_state_buffers_per_batch_row",
            "excludes": [
                "local_matrices",
                "autodiff_tapes",
                "backend_workspace",
                "density_branches",
            ],
            "large_byte_counts": "null above signed-64-bit range; log2 fields remain available",
            **asdict(self),
        }

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


def estimate_resources(
    circuit,
    *,
    batch_size=1,
    max_qubits=None,
    max_memory_mb=256,
    max_operations=100_000,
    max_depth=64,
):
    """Estimate dense storage without constructing a state or declaring solver feasibility.

    Circuit/source operation budgets and quantum-state storage are separate.
    A structured expectation plan may fit while dense execution cannot; such
    a plan still needs its own bounded compilation and correctness checks.
    """
    _check_circuit_limits(circuit, max_operations=max_operations, max_depth=max_depth)
    batch = integer(batch_size, "batch_size", 1)
    options = ExecutionOptions(max_qubits, max_memory_mb)
    width = circuit.num_qubits
    if width > 10_000_000:
        raise ResourceError("Resource-report width exceeds the bounded exponent range.")
    state_log = width + 4 + math.log2(batch)
    density_log = 2 * width + 4 + math.log2(batch)

    def byte_count(exponent):
        return batch * (1 << exponent) if exponent + batch.bit_length() <= 63 else None

    budget = float(options.max_memory_mb)
    numerator, denominator = budget.as_integer_ratio()
    available_bytes = (numerator << 20) // denominator
    memory_ok = batch <= available_bytes >> (width + 6)
    width_ok = max_qubits is None or width <= max_qubits
    notes = []
    if not width_ok:
        notes.append(
            Diagnostic(
                "DENSE_WIDTH_LIMIT",
                "Source width exceeds the selected dense execution limit.",
                "num_qubits",
            )
        )
    if not memory_ok:
        notes.append(
            Diagnostic(
                "DENSE_MEMORY_LIMIT",
                "Four state buffers exceed the budget; dense storage grows exponentially.",
                "max_memory_mb",
            )
        )
    if circuit.has_effects:
        notes.append(
            Diagnostic(
                "DENSITY_BRANCH_COST",
                "Effects may require density matrices and classical branches; "
                "statevector storage is not an effectful execution estimate.",
                severity="info",
            )
        )
    return ResourceEstimate(
        width,
        sum(1 for _ in circuit.walk_operations()),
        batch,
        max_qubits,
        budget,
        state_log,
        density_log,
        byte_count(width + 4),
        byte_count(2 * width + 4),
        byte_count(width + 6),
        memory_ok and width_ok and not circuit.has_effects,
        tuple(notes),
    )
