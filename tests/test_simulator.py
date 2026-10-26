import math

import numpy as np
import pytest

from openqml import (
    Circuit,
    Hamiltonian,
    Parameter,
    PauliString,
    ResourceError,
    ValidationError,
    execute,
    expectation,
    sample,
    statevector,
)


def test_bell_entanglement_and_observables():
    circuit = Circuit(2).h(0).cx(0, 1)
    np.testing.assert_allclose(statevector(circuit), [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
    for paulis, expected in [("ZI", 0), ("ZZ", 1), ("XX", 1), ("YY", -1)]:
        assert expectation(circuit, PauliString(paulis)) == pytest.approx(expected)
    observable = Hamiltonian((PauliString("XX", 0.4), PauliString("YY", 0.2)))
    assert expectation(circuit, observable) == pytest.approx(0.2)


def test_wire_order_reversed_controls_and_ccx():
    np.testing.assert_allclose(statevector(Circuit(3).x(0)), np.eye(8)[4])
    np.testing.assert_allclose(statevector(Circuit(3).x(2).cx(2, 0)), np.eye(8)[5])
    np.testing.assert_allclose(statevector(Circuit(3).x(2).x(0).ccx(2, 0, 1)), np.eye(8)[7])


def test_measurement_mapping_seed_and_overwrite():
    circuit = Circuit(3).x(0).measure(0, 2).measure(1, 0)
    result = execute(circuit, shots=27, seed=3)
    assert result.counts == {"001": 27}
    assert result.statevector is None
    assert sample(Circuit(2).x(0).measure(0, 0).measure(1, 0), 19) == {"0": 19}
    bell = Circuit(2).h(0).cx(0, 1).measure_all()
    counts = sample(bell, 1000, seed=19)
    assert set(counts) == {"00", "11"} and sum(counts.values()) == 1000
    assert counts == sample(bell, 1000, seed=19)


def test_execution_rejects_unsupported_measurement_and_bad_limits():
    with pytest.raises(ValidationError):
        statevector(Circuit(1).measure_all())
    assert sample(Circuit(1).measure(0).x(0), 10) == {"0": 10}
    for shots in (0, -1, 1.5, True):
        with pytest.raises(ValidationError):
            execute(Circuit(1), shots=shots)
    with pytest.raises(ResourceError):
        statevector(Circuit(100_000))
    with pytest.raises(ResourceError):
        statevector(Circuit(8), max_memory_mb=0.001)
    with pytest.raises(ValidationError):
        expectation(Circuit(2), PauliString("Z"))


@pytest.mark.parametrize("gate", ["rx", "ry", "rz", "p", "rxx", "ryy", "rzz"])
def test_rotation_inverse_and_global_phase(gate):
    q = (0, 2) if len(gate) == 3 else (1,)
    circuit = Circuit(3).h(0).ry(0.71, 1).cx(1, 2)
    roundtrip = circuit.gate(gate, *q, params=(0.72,)).gate(gate, *q, params=(-0.72,))
    np.testing.assert_allclose(statevector(roundtrip), statevector(circuit), atol=1e-14)
    phase = circuit.with_global_phase(Parameter("phase"))
    np.testing.assert_allclose(
        statevector(phase, {"phase": math.pi / 3}), statevector(circuit) * np.exp(1j * math.pi / 3)
    )
