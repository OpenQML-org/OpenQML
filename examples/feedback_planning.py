"""Plan a long adaptive program and request only its final count marginal."""

import numpy as np

from openqml import Circuit, ClassicalCondition, PauliString, estimate, plan_density, sample

circuit = Circuit(2)
for bit in range(64):
    circuit = (
        circuit.h(0)
        .measure(0, bit)
        .if_else(ClassicalCondition((bit,), 1), Circuit(2).x(1))
        .reset(0)
    )
plan = plan_density(circuit)
assert plan.peak_branches == 2 and plan.output_branches == 1
print("Exact readout plan:", plan)
print("Estimated state working MiB:", plan.estimated_memory_mb())
counts = sample(circuit, shots=1000, seed=7, classical_bits=(63,), max_branches=2)
assert set(counts) == {"0", "1"}
print("Final selected classical bit:", counts)
report = estimate(
    circuit, [PauliString("ZI"), PauliString("IZ")], shots=4096, seed=7, max_branches=4
)
np.testing.assert_allclose(report.values, [[1, 0]], atol=0.05)
print("Shot readout without retaining old histories:", report.values)
