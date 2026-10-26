"""Execution, interoperability and resource contracts."""

import numpy as np
import pytest

from openqml import (
    BackendCapabilities,
    BackendError,
    Circuit,
    ConversionError,
    Expression,
    Parameter,
    PauliString,
    QuantumProgram,
    ResourceError,
    ValidationError,
    convert,
    execute,
    execute_batch,
    expectations_batch,
    parameter_shift,
    statevector,
)
from openqml.parameters import expression


@pytest.mark.parametrize("bad", ["norm", "shape", "nan", "overflow", "short", "long", "none"])
@pytest.mark.parametrize("entry", ["expectations", "execute"])
def test_bad_state_batches_fail_the_same_contract_as_single_execution(bad, entry):
    class Broken:
        name = "broken-batch"
        capabilities = BackendCapabilities()

        def run(self, circuit, *, options):
            return np.array([1.0, 0.0])

        def run_batch(self, circuits, *, options):
            if bad == "none":
                return None
            if bad in {"short", "long"}:
                return [np.array([1.0, 0.0])] * (len(circuits) + (1 if bad == "long" else -1))
            value = {
                "norm": [2.0, 0.0],
                "shape": [1.0, 0.0, 0.0],
                "nan": [np.nan, 0.0],
                "overflow": [1e308, 0.0],
            }[bad]
            return [np.asarray(value) for _ in circuits]

    with pytest.raises(BackendError):
        if entry == "expectations":
            expectations_batch([Circuit(1)] * 2, (PauliString("Z"),), backend=Broken())
        else:
            execute_batch([Circuit(1)] * 2, backend=Broken())


@pytest.mark.parametrize("payload", [[[1 + 0.1j]], [[True]], [[np.inf]], [[4.0]], [[0.0, 1.0]]])
def test_native_readouts_reject_complex_nonfinite_wrong_shape_and_spectral_violations(payload):
    class Broken:
        name = "broken-readout"
        capabilities = BackendCapabilities(native_expectations=True)

        def expectations_batch(self, circuits, observables, *, options):
            return np.asarray(payload)

    with pytest.raises(BackendError):
        expectations_batch(Circuit(1), (PauliString("Z"),), backend=Broken())


def test_multiple_observables_reuse_each_state_once():
    class Counter:
        name = "state-counter"
        capabilities = BackendCapabilities()

        def __init__(self):
            self.calls = 0

        def run(self, circuit, *, options):
            self.calls += 1
            return statevector(circuit)

    engine = Counter()
    values = expectations_batch(
        [Circuit(1), Circuit(1).h(0)],
        (PauliString("X"), PauliString("Y"), PauliString("Z")),
        backend=engine,
    )
    np.testing.assert_allclose(values, [[0, 0, 1], [1, 0, 0]], atol=1e-12)
    assert engine.calls == 2


def test_dense_state_batch_is_chunked_within_working_budget():
    circuits = [Circuit(8).h(0)] * 64
    results = execute_batch(circuits, max_memory_mb=0.5)
    assert len(results) == 64
    reference = statevector(circuits[0])
    for result in results:
        np.testing.assert_allclose(result.statevector, reference, atol=1e-12)


def test_resource_rejection_happens_before_untrusted_provider_allocation():
    class Engine:
        name = "unchecked-allocation"
        capabilities = BackendCapabilities()

        def run_batch(self, circuits, *, options):
            raise AssertionError("Resource limits must be checked first")

        def run(self, circuit, *, options):
            raise AssertionError("Resource limits must be checked first")

    with pytest.raises(ResourceError):
        expectations_batch(
            Circuit(10), (PauliString("Z" * 10),), backend=Engine(), max_memory_mb=0.001
        )


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("option", [{"max_qubits": 1}, {"max_memory_mb": 1e-8}])
def test_numeric_tensor_batch_honors_caller_limits(backend, option):
    pytest.importorskip(backend)
    with pytest.raises(ResourceError):
        expectations_batch(Circuit(2), (PauliString("ZZ"),), backend=backend, **option)


def test_provider_exceptions_and_invalid_sampler_batches_are_backend_errors():
    class Failed:
        name = "failed-provider"
        capabilities = BackendCapabilities(native_expectations=True, native_sampling=True)

        def expectations_batch(self, circuits, observables, *, options):
            raise RuntimeError("provider failure")

        def sample_batch(self, circuits, shots, *, options):
            return None

    with pytest.raises(BackendError, match="provider failure"):
        expectations_batch(Circuit(1), (PauliString("Z"),), backend=Failed())
    for call in (
        lambda: execute(Circuit(1), backend=Failed(), shots=3),
        lambda: execute_batch(Circuit(1), backend=Failed(), shots=3),
    ):
        with pytest.raises(BackendError, match="batch length"):
            call()
    with pytest.raises(ValidationError, match="seed"):
        execute_batch([], seed=-1)
    with pytest.raises(ValidationError, match="64-bit"):
        execute_batch([], shots=2**63)


def test_explicit_provider_instance_is_not_replaced_by_its_builtin_name():
    class NamedNumpy:
        name = "numpy"
        capabilities = BackendCapabilities()

        def run(self, circuit, *, options):
            return np.array([0.0, 1.0])

    assert expectations_batch(Circuit(1), (PauliString("Z"),), backend=NamedNumpy())[0, 0] == -1


def test_constant_graph_power_derivative_matches_finite_difference_and_shift():
    exponent = Expression(op="add", args=(expression(1), expression(1)))
    power = Expression(op="pow", args=(expression(Parameter("x")), exponent))
    assert power.evaluate({"x": 3}) == pytest.approx(9)
    assert power.derivative("x").evaluate({"x": 3}) == pytest.approx(6)
    c = Circuit(1).ry(power, 0)
    derivative = parameter_shift(c, PauliString("Z"), {"x": 3})["x"]
    assert derivative == pytest.approx(-6 * np.sin(9), abs=1e-12)
    h = 1e-6
    finite = (power.evaluate({"x": 3 + h}) - power.evaluate({"x": 3 - h})) / (2 * h)
    assert derivative / (-np.sin(9)) == pytest.approx(finite, abs=1e-8)


@pytest.mark.parametrize("selected", [None, ("x",), "generator"])
def test_shift_dictionary_accepts_none_and_one_shot_parameter_iterables(selected):
    if selected == "generator":
        selected = (p for p in ("x",))
    value = parameter_shift(
        Circuit(1).ry(Parameter("x"), 0), PauliString("Z"), {"x": 0.37}, parameters=selected
    )
    assert value == pytest.approx({"x": -np.sin(0.37)}, abs=1e-12)


def test_affine_overflow_is_rejected_before_trigonometric_execution():
    circuit = Circuit(1).rx(1e308 * Parameter("x"), 0)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValidationError, match="undefined"):
            statevector(circuit, {"x": 1e308})


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32", "float64"])
def test_empty_quantum_layer_keeps_dtype_shape_and_zero_gradients(dtype_name):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    dtype = getattr(torch, dtype_name)
    layer = QuantumLayer.hardware_efficient(2, 1, dtype=torch.float64)
    x = torch.empty(2, 0, 2, dtype=dtype, requires_grad=True)
    out = layer(x)
    assert out.shape == (2, 0, 2) and out.dtype == dtype
    out.sum().backward()
    assert x.grad is not None and x.grad.dtype == dtype
    torch.testing.assert_close(layer.weights.grad, torch.zeros_like(layer.weights))


def test_empty_quantum_image_batch_keeps_all_parameter_gradients():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumConv2d

    layer = QuantumConv2d(1, 2, num_qubits=2, dtype=torch.float64)
    x = torch.empty(0, 1, 4, 4, dtype=torch.float64, requires_grad=True)
    out = layer(x)
    assert out.shape == (0, 2, 2, 2)
    out.sum().backward()
    for p in layer.parameters():
        assert p.grad is not None
        torch.testing.assert_close(p.grad, torch.zeros_like(p))


@pytest.mark.parametrize("parser", ["builtin", "auto", "qiskit"])
def test_qasm_operation_limit_cannot_be_bypassed_by_frontend_switch(parser):
    if parser == "qiskit":
        pytest.importorskip("qiskit")
    source = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[1]; h q[0]; h q[0];'
    with pytest.raises(ConversionError, match="OPERATION_LIMIT"):
        convert(
            source,
            source="openqasm2",
            target="ir",
            source_options={"parser": parser, "max_operations": 1},
        )


def test_sdk_limits_apply_before_import_and_header_ignores_comments(monkeypatch):
    def forbidden(*args):
        raise AssertionError("Oversized source must be rejected before importing a parser")

    monkeypatch.setattr("openqml.adapters._common.dependency", forbidden)
    with pytest.raises(ConversionError, match="REGISTER_LIMIT"):
        QuantumProgram.from_qasm("OPENQASM 3.0; qubit[1000000] q;", max_register_bits=2)
    with pytest.raises(ConversionError, match="SOURCE_LIMIT"):
        QuantumProgram.from_qasm("OPENQASM 3.0; qubit q;", max_source_bytes=5)

    class TooLarge(str):
        def encode(self, *args, **kwargs):
            raise AssertionError("Oversized text must be rejected before creating an encoded copy")

    with pytest.raises(ConversionError, match="SOURCE_LIMIT"):
        QuantumProgram.from_qasm(TooLarge("OPENQASM 3.0; qubit q;"), max_source_bytes=5)
    with pytest.raises(ConversionError, match="QASM_VERSION"):
        QuantumProgram.from_qasm("// OPENQASM 2.0;\nOPENQASM 3.0; qubit q;", version=2)


def test_custom_gate_expansion_budget_persists_on_native_program():
    pytest.importorskip("qiskit")
    source = 'OPENQASM 2.0; include "qelib1.inc"; gate block a { h a; x a; } qreg q[1]; block q[0];'
    program = QuantumProgram.from_qasm(source, max_operations=1)
    with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
        program.to_ir()
    with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
        convert(source, source="openqasm2", target="ir", source_options={"max_operations": 1})


def test_auto_frontend_preserves_classical_control_diagnostic():
    pytest.importorskip("qiskit_qasm3_import")
    source = (
        'OPENQASM 3.0; include "stdgates.inc"; qubit q; bit c; '
        "c = measure q; while(c) { x q; c = measure q; }"
    )
    with pytest.raises(ConversionError, match="CLASSICAL_CONTROL"):
        convert(source, source="openqasm3", target="ir")


def test_explicit_include_path_and_comments_work_through_auto_frontend(tmp_path):
    pytest.importorskip("qiskit")
    (tmp_path / "custom.inc").write_text("gate block a { U(pi,0,pi) a; }")
    source = (
        '// include "blocked.inc"\nOPENQASM 2.0; include // between tokens\n'
        ' "custom.inc"; qreg q[1]; block q[0];'
    )
    result = convert(
        source, source="openqasm2", target="ir", source_options={"include_paths": (str(tmp_path),)}
    )
    np.testing.assert_allclose(statevector(result.circuit), [0, 1], atol=1e-12)
    assert any(d.code == "SDK_FRONTEND" for d in result.diagnostics)
    with pytest.raises(ConversionError, match="INCLUDE_PATH"):
        QuantumProgram.from_qasm(source)


def test_native_qiskit_standard_and_depth_limits():
    qiskit = pytest.importorskip("qiskit")
    from openqml.adapters.qiskit import from_qiskit

    native = qiskit.QuantumCircuit(1)
    native.h(0)
    native.x(0)
    with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
        from_qiskit(native, max_operations=1)
    wrapped = qiskit.QuantumCircuit(1)
    wrapped.append(native.to_gate(), [0])
    with pytest.raises(ConversionError, match="max_depth"):
        from_qiskit(wrapped, max_depth=0)


def test_qiskit_matrix_fallback_checks_budget_before_operator_allocation(monkeypatch):
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import Gate

    from openqml.adapters.qiskit import from_qiskit

    native = qiskit.QuantumCircuit(11)
    native.append(Gate("unknown_big_unitary", 11, []), range(11))

    def forbidden(*args):
        raise AssertionError("Matrix allocator must not be called above budget")

    monkeypatch.setattr("qiskit.quantum_info.Operator", forbidden)
    with pytest.raises(ConversionError, match="MATRIX_RESOURCE_LIMIT"):
        from_qiskit(native)


def test_cirq_matrix_budget_precedes_unitary_protocol():
    cirq = pytest.importorskip("cirq")
    from openqml.adapters.cirq import from_cirq

    class Bomb(cirq.Gate):
        def _num_qubits_(self):
            return 11

        def _unitary_(self):
            raise AssertionError("Matrix protocol must not be called above budget")

    native = cirq.Circuit(Bomb().on(*cirq.LineQubit.range(11)))
    with pytest.raises(ConversionError, match="MATRIX_RESOURCE_LIMIT"):
        from_cirq(native)


def test_pennylane_template_budget_and_depth_are_enforced():
    qml = pytest.importorskip("pennylane")
    from openqml.adapters.pennylane import from_pennylane

    native = qml.tape.QuantumScript(
        [qml.StronglyEntanglingLayers(np.ones((1, 2, 3)), wires=[0, 1])], []
    )
    with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
        from_pennylane(native, max_operations=2)
    with pytest.raises(ConversionError, match="max_depth"):
        from_pennylane(native, max_depth=0)
    matrix = qml.tape.QuantumScript([qml.QubitUnitary(np.eye(4), wires=[0, 1])], [])
    with pytest.raises(ConversionError, match="MATRIX_RESOURCE_LIMIT"):
        from_pennylane(matrix, max_matrix_qubits=1)
    assert len(from_pennylane(matrix, max_matrix_qubits=2).operations) == 1


@pytest.mark.parametrize("version", [2, 3])
def test_bounded_nested_custom_includes_compile_in_both_qasm_versions(tmp_path, version):
    pytest.importorskip("qiskit")
    if version == 3:
        pytest.importorskip("qiskit_qasm3_import")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/second.inc").write_text("gate helper a { U(pi,0,pi) a; }")
    (tmp_path / "first.inc").write_text('include "sub/second.inc"; gate block a { helper a; }')
    decl = "qreg q[1]; block q[0];" if version == 2 else "qubit q; block q;"
    source = f'OPENQASM {version}.0; include "first.inc"; {decl}'
    result = convert(
        source,
        source=f"openqasm{version}",
        target="ir",
        source_options={"include_paths": (tmp_path,)},
    )
    np.testing.assert_allclose(statevector(result.circuit), [0, 1], atol=1e-12)
    with pytest.raises(ConversionError, match="INCLUDE_DEPTH_LIMIT"):
        QuantumProgram.from_qasm(source, include_paths=(tmp_path,), max_include_depth=1)


def test_custom_includes_reject_cycles_and_cumulative_byte_overflow(tmp_path):
    (tmp_path / "a.inc").write_text('include "b.inc";')
    (tmp_path / "b.inc").write_text('include "a.inc";')
    source = 'OPENQASM 2.0; include "a.inc"; qreg q[1];'
    with pytest.raises(ConversionError, match="INCLUDE_CYCLE"):
        QuantumProgram.from_qasm(source, include_paths=(tmp_path,))
    (tmp_path / "a.inc").write_text(" " * 500)
    with pytest.raises(ConversionError, match="SOURCE_LIMIT"):
        QuantumProgram.from_qasm(source, include_paths=(tmp_path,), max_source_bytes=100)


def test_include_roots_block_traversal_and_symlink_escape(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    outside = tmp_path / "outside.inc"
    outside.write_text("gate block a { U(pi,0,pi) a; }")
    source = 'OPENQASM 2.0; include "../outside.inc"; qreg q[1];'
    with pytest.raises(ConversionError, match="INCLUDE_PATH"):
        QuantumProgram.from_qasm(source, include_paths=(root,))
    try:
        (root / "link.inc").symlink_to(outside)
    except OSError:
        pytest.skip("Symbolic links unavailable")
    source = 'OPENQASM 2.0; include "link.inc"; qreg q[1];'
    with pytest.raises(ConversionError, match="INCLUDE_PATH"):
        QuantumProgram.from_qasm(source, include_paths=(root,))


def test_invalid_utf8_include_and_malformed_directive_have_structured_diagnostics(tmp_path):
    (tmp_path / "a.inc").write_bytes(b"\xff")
    with pytest.raises(ConversionError, match="INCLUDE_ENCODING"):
        QuantumProgram.from_qasm(
            'OPENQASM 2.0; include "a.inc"; qreg q[1];', include_paths=(tmp_path,)
        )
    with pytest.raises(ConversionError, match="SYNTAX"):
        QuantumProgram.from_qasm(
            'OPENQASM 2.0; include "a.inc" qreg q[1];', include_paths=(tmp_path,)
        )
