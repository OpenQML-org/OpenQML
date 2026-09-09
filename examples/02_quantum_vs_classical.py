"""Run a benchmark suite with the classical control arm alongside.

The table below is the point of the whole package: the quantum kernel wins on
some of these tasks and loses on others, and both belong in the report.

    python examples/02_quantum_vs_classical.py
"""

from openqml import (
    ClassicalKernelClassifier,
    MajorityClassifier,
    QuantumKernelClassifier,
    get_suite,
)

suite = get_suite("qml-cls-1")
print(repr(suite))
print(suite.description, "\n")

table = suite.compare([
    QuantumKernelClassifier(feature_map="iqp"),
    QuantumKernelClassifier(feature_map="zz"),
    ClassicalKernelClassifier(kernel="rbf"),
    MajorityClassifier(),
])
print(table.to_string(index=False))

print("""
Read the whole row, not the best cell:

  * parity            the IQP kernel gets it exactly right and the RBF kernel scores
                      0.00 -- worse than the majority floor, because a smooth
                      interpolator extrapolates the opposite label on unseen bitstrings.
  * tfim phases       the fidelity kernel is ahead, on data that already is a quantum
                      state, where no encoding step is needed at all.
  * moons / circles   the RBF kernel matches or beats every quantum entry, in
                      microseconds rather than seconds.

Three of those are wins for someone and all three belong in the report.""")
