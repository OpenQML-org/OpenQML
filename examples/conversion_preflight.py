"""Inspect target lowering, SDK requirements and expansion limits before exporting."""

import numpy as np

from openqml import Circuit, ClassicalCondition, plan_conversion

single_qubit = Circuit(1).unitary(np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2), 0)
plan = plan_conversion(single_qubit, target="openqasm3")
print(plan.to_json())
assert plan.preparation_complete and plan.required_extras == ()
print(plan.export().value)

# Multi-qubit matrix synthesis is explicitly deferred to the Qiskit extra.
deferred = plan_conversion(Circuit(2).unitary(np.eye(4), 0, 1), target="openqasm3")
assert deferred.compatible and not deferred.preparation_complete
print("Required export extras:", deferred.required_extras)

branch = Circuit(2).crx(0.4, 0, 1)
adaptive = Circuit(2).if_else(ClassicalCondition((0,), 0), branch, branch)
limited = plan_conversion(adaptive, target="openqasm3", max_operations=10)
assert not limited.compatible
for diagnostic in limited.diagnostics:
    print(diagnostic)
