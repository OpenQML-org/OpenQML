from dataclasses import dataclass, field

import numpy as np
import pytest

from openqml import (
    BackendCapabilities,
    BackendError,
    Circuit,
    ConversionError,
    Parameter,
    PauliString,
    ResourceError,
    ValidationError,
    execute,
    expectation,
    list_backends,
    parameter_shift,
    register_backend,
    statevector,
    unregister_backend,
)


@dataclass
class RecordingBackend:
    name: str = "recording"
    capabilities: BackendCapabilities = BackendCapabilities(gates=frozenset({"p", "ry", "cx"}))
    calls: list = field(default_factory=list)

    def run(self, circuit, *, options):
        self.calls.append((circuit, options))
        return statevector(
            circuit, max_qubits=options.max_qubits, max_memory_mb=options.max_memory_mb
        )


def test_custom_backend_gets_bound_lowered_unitary_circuit():
    backend = RecordingBackend()
    circuit = Circuit(3).h(2).crx(Parameter("theta"), 2, 0).barrier().measure(0, 1)
    result = execute(circuit, {"theta": 0.51}, backend=backend, shots=300, seed=19)
    lowered, options = backend.calls[0]
    assert not lowered.parameters and not lowered.has_measurements
    assert {op.name for op in lowered.operations} <= {"p", "ry", "cx"}
    assert options.seed == 19
    assert result.backend == "recording" and result.statevector is None
    reference = execute(circuit, {"theta": 0.51}, shots=300, seed=19)
    assert sum(result.counts.values()) == 300
    assert set(result.counts) <= {"00", "01"}
    repeated = execute(circuit, {"theta": 0.51}, backend=backend, shots=300, seed=19)
    assert repeated.counts == result.counts
    np.testing.assert_allclose(result.probabilities, reference.probabilities, atol=1e-12)


def test_registration_is_explicit_reversible_and_protects_builtins():
    backend = RecordingBackend()
    register_backend(backend)
    try:
        assert "recording" in list_backends() and not backend.calls
        execute(Circuit(1).h(0), backend="recording")
        assert len(backend.calls) == 1
        with pytest.raises(ValidationError, match="already registered"):
            register_backend(backend)
        with pytest.raises(ValidationError, match="cannot be replaced"):
            register_backend(RecordingBackend(name="numpy"), replace=True)
    finally:
        assert unregister_backend("recording") is backend
    assert "recording" not in list_backends()
    with pytest.raises(ValidationError, match="Unknown backend"):
        execute(Circuit(1), backend="recording")


@pytest.mark.parametrize(
    "bad_state",
    [
        np.ones(3),
        np.ones((2, 2)),
        np.array([np.nan, 0]),
        np.array([2, 0]),
        np.array([np.sqrt(1.000005), 0]),
        np.array([1e308, 0]),
    ],
)
def test_invalid_backend_results_are_rejected(bad_state):
    class Broken:
        name, capabilities = "broken", BackendCapabilities()

        def run(self, circuit, *, options):
            return bad_state

    with pytest.raises(BackendError):
        execute(Circuit(1), backend=Broken())


def test_backend_capability_and_execution_errors():
    small = RecordingBackend(capabilities=BackendCapabilities(max_qubits=1))
    with pytest.raises(ResourceError, match="supports at most"):
        execute(Circuit(2), backend=small)
    narrow = RecordingBackend(capabilities=BackendCapabilities(gates=frozenset({"p"})))
    with pytest.raises(ConversionError, match="DECOMPOSITION_UNAVAILABLE"):
        execute(Circuit(2).cx(0, 1), backend=narrow)
    with pytest.raises(ValidationError):
        register_backend(object())

    class Failed:
        name, capabilities = "failed", BackendCapabilities()

        def run(self, circuit, *, options):
            raise RuntimeError("sample failure")

    with pytest.raises(BackendError, match="sample failure"):
        execute(Circuit(1), backend=Failed())


def test_external_expectation_and_shift_gradient_share_backend_contract():
    backend = RecordingBackend()
    circuit = Circuit(2).h(0).crx(Parameter("theta"), 0, 1)
    observable, theta = PauliString("XI"), 0.73
    value = expectation(circuit, observable, {"theta": theta}, backend=backend)
    gradient = parameter_shift(circuit, observable, {"theta": theta}, backend=backend)
    assert value == pytest.approx(np.cos(theta / 2), abs=1e-12)
    assert gradient["theta"] == pytest.approx(-np.sin(theta / 2) / 2, abs=1e-12)


@pytest.mark.parametrize("framework", ["qiskit", "pennylane", "cirq"])
def test_builtin_execution_ignores_scheduling_barriers(framework):
    pytest.importorskip(framework)
    circuit = Circuit(2).h(0).barrier().cx(0, 1)
    np.testing.assert_allclose(
        execute(circuit, backend=framework).statevector, statevector(circuit), atol=1e-12
    )
