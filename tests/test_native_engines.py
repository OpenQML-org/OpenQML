import numpy as np
import pytest

from openqml import (
    Circuit,
    Parameter,
    PauliString,
    QuantumProgram,
    execute_batch,
    expectation,
    expectations_batch,
    parameter_shift_jacobian,
    sample,
)


def test_batched_numeric_execution_and_nonlinear_jacobian():
    c = Circuit(2).h(0).ry(Parameter("x").sin() * Parameter("w"), 1).crx(Parameter("x") ** 2, 0, 1)
    rows = [{"x": x, "w": 0.37} for x in [0.1, 0.2, 0.3, 0.4]]
    obs = (PauliString("XZ"), PauliString("ZZ"))
    batch = expectations_batch(c, obs, rows)
    results = execute_batch(c, rows)
    assert len(results) == 4
    for i, row in enumerate(rows):
        for j, ob in enumerate(obs):
            assert batch[i, j] == pytest.approx(expectation(c, ob, row), abs=1e-12)
    jac = parameter_shift_jacobian(c, obs, rows)
    for i, row in enumerate(rows):
        for k, name in enumerate(c.parameters):
            plus, minus = dict(row), dict(row)
            plus[name] += 1e-6
            minus[name] -= 1e-6
            for j, ob in enumerate(obs):
                finite = (expectation(c, ob, plus) - expectation(c, ob, minus)) / 2e-6
                assert jac[i, j, k] == pytest.approx(finite, abs=1e-8)


def test_mps_64_qubit_expectation_and_native_sampling_without_dense_outputs():
    pytest.importorskip("qiskit_aer")
    c = Circuit(64).h(0)
    for q in range(1, 64):
        c = c.cx(q - 1, q)
    assert expectation(c, PauliString("Z" + "I" * 62 + "Z"), backend="aer-mps") == pytest.approx(
        1, abs=1e-10
    )
    result = execute_batch(c, [{}], backend="aer-mps", shots=128, seed=17)[0]
    assert result.probabilities is None and result.statevector is None
    assert set(result.counts) <= {"0" * 64, "1" * 64} and sum(result.counts.values()) == 128


def test_native_mid_circuit_measurement_and_structured_program_execution():
    pytest.importorskip("qiskit_aer")
    c = Circuit(2).h(0).measure(0, 0).cx(0, 1).measure(1, 1)
    assert set(sample(c, shots=128, backend="aer", seed=11)) <= {"00", "11"}
    from qiskit import QuantumCircuit

    q = QuantumCircuit(2, 2)
    q.h(0)
    q.measure(0, 0)
    with q.if_test((q.clbits[0], 1)):
        q.x(1)
    q.measure(1, 1)
    result = QuantumProgram.capture(q).run(shots=128, seed=3)
    assert set(result.get_counts()) <= {"00", "11"}
    ir = QuantumProgram.capture(q).to_ir()
    assert any(op.name == "if_else" for op in ir.operations)
    assert set(sample(ir, 128, seed=3)) <= {"00", "11"}


def test_aer_expectation_batches_match_numpy_and_jacobians():
    pytest.importorskip("qiskit_aer")
    c = Circuit(2).h(0).ry(Parameter("x").sin(), 1).cp(Parameter("x") ** 2, 0, 1)
    rows = [{"x": 0.23}, {"x": -0.41}]
    obs = (PauliString("XZ"), PauliString("YY"))
    np.testing.assert_allclose(
        expectations_batch(c, obs, rows, backend="aer"),
        expectations_batch(c, obs, rows),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        parameter_shift_jacobian(c, obs, rows, backend="aer"),
        parameter_shift_jacobian(c, obs, rows),
        atol=1e-12,
    )


def test_expectation_and_sampling_only_plugins_without_dense_state_contract():
    from openqml import BackendCapabilities, execute, register_backend, unregister_backend

    class Analytic:
        name = "analytic_readout"
        capabilities = BackendCapabilities(gates=frozenset({"ry"}), native_expectations=True)

        def expectations_batch(self, circuits, observables, *, options):
            return np.asarray([[np.cos(c.operations[0].params[0].constant)] for c in circuits])

    class Sampler:
        name = "direct_sampler"
        capabilities = BackendCapabilities(native_sampling=True, max_qubits=1)

        def sample_batch(self, circuits, shots, *, options):
            return [{"0": shots} for _ in circuits]

    register_backend(Analytic())
    register_backend(Sampler())
    try:
        c = Circuit(1).ry(Parameter("x"), 0)
        assert expectation(
            c, PauliString("Z"), {"x": 0.37}, backend="analytic_readout"
        ) == pytest.approx(np.cos(0.37))
        j = parameter_shift_jacobian(
            c, (PauliString("Z"),), ({"x": 0.37},), backend="analytic_readout"
        )
        assert j[0, 0, 0] == pytest.approx(-np.sin(0.37), abs=1e-12)
        assert execute(Circuit(1), shots=5, backend="direct_sampler").counts == {"0": 5}
        from openqml import ResourceError

        with pytest.raises(ResourceError):
            execute(Circuit(2), shots=5, backend="direct_sampler")
    finally:
        unregister_backend("analytic_readout")
        unregister_backend("direct_sampler")
