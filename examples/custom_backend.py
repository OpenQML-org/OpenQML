"""Register a backend instance with a restricted supported basis; NumPy only."""

from openqml import (
    BackendCapabilities,
    Circuit,
    execute,
    list_backends,
    register_backend,
    statevector,
    unregister_backend,
)


class MySimulator:
    name = "my_simulator"
    capabilities = BackendCapabilities(gates=frozenset({"p", "ry", "cz"}), max_qubits=10)

    def run(self, circuit, *, options):
        # Replace this reference implementation with your local simulator.
        # The input is bound, unitary and already lowered to the declared basis.
        assert {op.name for op in circuit.operations} <= self.capabilities.gates
        return statevector(
            circuit, max_qubits=options.max_qubits, max_memory_mb=options.max_memory_mb
        )


register_backend(MySimulator())
try:
    print("Registered:", list_backends())
    result = execute(
        Circuit(2).h(0).cx(0, 1).measure_all(), backend="my_simulator", shots=1000, seed=42
    )
    print("Counts:", result.counts)
    assert set(result.counts) <= {"00", "11"}
finally:
    unregister_backend("my_simulator")
