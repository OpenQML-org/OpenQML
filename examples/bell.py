"""Run with: python examples/bell.py"""

from openqml import Circuit, PauliString, execute, expectation

bell = Circuit(2).h(0).cx(0, 1)
print("Bell state:", execute(bell).statevector)
print("<ZZ>:", expectation(bell, PauliString("ZZ")))
print("Counts:", execute(bell.measure_all(), shots=1000, seed=42).counts)
