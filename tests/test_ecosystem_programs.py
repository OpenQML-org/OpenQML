import numpy as np
import pytest

from openqml import (
    AmplitudeOracle,
    Circuit,
    LayerSpec,
    Parameter,
    PauliString,
    QuantumProgram,
    as_circuit,
    convert,
    expectation,
    finance,
    hardware_efficient,
    parameter_shift,
    sim,
    statevector,
)


def test_nonlinear_expression_roundtrip_derivative_and_batch():
    x, w = Parameter("inputs[0]"), Parameter("θ")
    angle = (x.sin() * w + x**2 / (1 + w**2)).exp().log()
    c = Circuit(1).ry(angle, 0)
    restored = Circuit.from_json(c.to_json())
    assert restored == c
    bindings = {"inputs[0]": 0.37, "θ": -0.61}
    g = parameter_shift(c, PauliString("Z"), bindings)
    for name in bindings:
        plus, minus = dict(bindings), dict(bindings)
        plus[name] += 1e-6
        minus[name] -= 1e-6
        finite = (
            expectation(c, PauliString("Z"), plus) - expectation(c, PauliString("Z"), minus)
        ) / 2e-6
        assert g[name] == pytest.approx(finite, abs=1e-8)
    values = statevector(c, {"inputs[0]": np.array([0.1, 0.2, 0.3]), "θ": -0.61})
    assert values.shape == (3, 2)
    for i, xvalue in enumerate([0.1, 0.2, 0.3]):
        np.testing.assert_allclose(
            values[i], statevector(c, {"inputs[0]": xvalue, "θ": -0.61}), atol=1e-12
        )


@pytest.mark.parametrize("framework", ["qiskit", "pennylane", "cirq"])
def test_arbitrary_complex_unitary_cross_framework(framework):
    pytest.importorskip(framework)
    rng = np.random.default_rng(19)
    matrix, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    c = Circuit(3).h(0).ry(0.43, 1).x(2).unitary(matrix, 2, 0).with_global_phase(0.27)
    assert Circuit.from_json(c.to_json()) == c
    result = convert(c, source="ir", target=framework)
    restored = convert(result.value, source=framework, target="ir").value
    np.testing.assert_allclose(statevector(restored), statevector(c), atol=2e-11)
    from openqml import execute

    np.testing.assert_allclose(
        execute(c, backend=framework).statevector, statevector(c), atol=2e-11
    )
    np.testing.assert_allclose(statevector(c.compose(c.inverse())), np.eye(8)[0], atol=2e-11)


_SIX_WIRE_PATH = (0, 4, 1, 3, 2, 5)


@pytest.fixture
def six_wire_layer():
    template = hardware_efficient(6, 1, wire_order=_SIX_WIRE_PATH)
    layer = LayerSpec(
        template.circuit.rx(0.31, 5),
        template.input_parameters,
        template.weight_parameters,
        (PauliString("IIIIIZ"), PauliString("XXXIII")),
    )
    inputs = [0.48, 0.55, 0.72, 0.84, 0.68, 0.91]
    weights = [0.12, 0.22, -0.10, -0.34, 0.16, 0.29, -0.07, 0.42, 0.14, -0.38, 0.08, 0.31]
    return layer, np.array(inputs + weights)


def _six_wire_state(coordinates):
    # Each CX output is the parity of its path prefix. Recover the input bits
    # directly for all 64 basis states before applying the final RX.
    theta = coordinates[:6] + coordinates[6::2]
    phase = coordinates[7::2]
    output = (np.arange(64)[:, None] >> np.arange(5, -1, -1)) & 1
    inputs = output.copy()
    inputs[:, _SIX_WIRE_PATH[1:]] ^= output[:, _SIX_WIRE_PATH[:-1]]
    zero = np.cos(theta / 2) * np.exp(-0.5j * phase)
    one = np.sin(theta / 2) * np.exp(0.5j * phase)
    state = np.prod(np.where(inputs, one, zero), axis=1)
    return np.cos(0.155) * state - 1j * np.sin(0.155) * state[np.arange(64) ^ 1]


def _six_wire_observables(state):
    z = np.sum(abs(state[::2]) ** 2 - abs(state[1::2]) ** 2)
    x = np.vdot(state, state[np.arange(64) ^ 0b111000]).real
    return np.array([z, x])


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "pennylane", "openqasm3"])
def test_six_wire_framework_state_and_readouts(six_wire_layer, framework):
    pytest.importorskip("qiskit_qasm3_import" if framework == "openqasm3" else framework)
    layer, values = six_wire_layer
    bindings = dict(zip(layer.input_parameters + layer.weight_parameters, values, strict=True))
    exported = convert(layer.circuit.bind(bindings), target=framework).value
    program = (
        QuantumProgram.from_qasm(exported, version=3)
        if framework == "openqasm3"
        else QuantumProgram.capture(exported)
    )
    reference = _six_wire_state(values)
    np.testing.assert_allclose(statevector(program.to_ir()), reference, atol=2e-12, rtol=0)
    report = program.certified_readout(layer.observables, absolute_error=1e-12)
    assert report.simulated_wires == tuple(range(6))
    np.testing.assert_allclose(report.values, _six_wire_observables(reference), atol=2e-12, rtol=0)


def test_six_wire_hybrid_gradient_and_discounted_payoff(six_wire_layer):
    layer, values = six_wire_layer
    names = layer.input_parameters + layer.weight_parameters
    bindings = dict(zip(names, values, strict=True))
    reference = _six_wire_state(values)
    expected = _six_wire_observables(reference)
    actual = statevector(layer.circuit, bindings)
    np.testing.assert_allclose(actual, reference, atol=2e-12, rtol=0)
    # Leave wire 5 on the right to visit each nontrivial bipartition once.
    for mask in range(1, 32):
        left = tuple(q for q in range(5) if mask & (1 << q))
        right = tuple(q for q in range(6) if q not in left)
        matrix = actual.reshape((2,) * 6).transpose(left + right).reshape(2 ** len(left), -1)
        assert np.linalg.svd(matrix, compute_uv=False)[1] > 1e-8
    plan = sim.plan_execution(layer.circuit, layer.observables, method="tensor-network")
    np.testing.assert_allclose(plan.evaluate(bindings), expected, atol=2e-12, rtol=0)

    head = np.array([1.0, 0.7])
    result = layer.certified_vjp(bindings, head, parameters=names, absolute_error=1e-11)

    def objective(point):
        return head @ _six_wire_observables(_six_wire_state(point))

    step_size = 2e-5
    derivatives = np.array(
        [
            (objective(values + step) - objective(values - step)) / (2 * step_size)
            for step in step_size * np.eye(len(names))
        ]
    )
    assert min(abs(derivatives)) > 1e-4
    assert result.parameters == names
    np.testing.assert_allclose(result.values, derivatives, atol=5e-10, rtol=0)

    oracle = AmplitudeOracle(layer.circuit.bind(bindings), 5)
    price = finance.price(
        oracle=oracle,
        bounds=(0, 100),
        discount=0.96,
        backend=sim.MPSBackend(layout="auto"),
        error=1e-9,
    )
    payoff = 96 * np.sum(abs(reference[1::2]) ** 2)
    np.testing.assert_allclose(price.interval, [payoff, payoff], atol=1e-9, rtol=0)


def test_qiskit_python_factory_custom_gate_paramvector_and_nonlinearity():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import Statevector

    x = ParameterVector("inputs", 2)
    w = ParameterVector("weights", 2)

    def factory():
        block = qiskit.QuantumCircuit(2)
        block.ry(x[0].sin() * w[0] + x[1] ** 2, 0)
        block.rzx(w[1].cos(), 0, 1)
        c = qiskit.QuantumCircuit(3)
        for targets in [(2, 0), (0, 1)]:
            c.append(block.to_gate(label="library_block"), targets)
        return c

    p = QuantumProgram.from_callable(factory)
    ir = p.to_ir()
    bindings = {x[0].name: 0.2, x[1].name: -0.43, w[0].name: 0.61, w[1].name: -0.27}
    bound = p.bind(bindings).native
    expected = (
        np.asarray(Statevector.from_instruction(bound).data)
        .reshape(2, 2, 2)
        .transpose(2, 1, 0)
        .reshape(-1)
    )
    np.testing.assert_allclose(statevector(ir, bindings), expected, atol=2e-12)
    exported = convert(ir, source="ir", target="openqasm3")
    assert exported.parameter_map and any(
        d.code == "PARAMETER_RENAMED" for d in exported.diagnostics
    )
    back = convert(exported.value, source="openqasm3", target="ir").value
    aliases = {exported.parameter_map.get(k, k): v for k, v in bindings.items()}
    np.testing.assert_allclose(statevector(back, aliases), expected, atol=2e-12)


@pytest.mark.parametrize("version", [2, 3])
def test_official_qasm_frontend_custom_gate_definitions(version):
    pytest.importorskip("qiskit")
    if version == 3:
        pytest.importorskip("qiskit_qasm3_import")
    include = "qelib1.inc" if version == 2 else "stdgates.inc"
    declaration = "qreg q[2];" if version == 2 else "qubit[2] q;"
    source = (
        f'OPENQASM {version}.0; include "{include}"; '
        f"gate learned(t) a,b {{ ry(t) a; cx a,b; }} {declaration} learned(0.43) q[0],q[1];"
    )
    p = QuantumProgram.from_qasm(source)
    expected = statevector(Circuit(2).ry(0.43, 0).cx(0, 1))
    np.testing.assert_allclose(statevector(p.to_ir()), expected, atol=1e-12)
    np.testing.assert_allclose(
        statevector(convert(source, source=f"openqasm{version}", target="ir").value),
        expected,
        atol=1e-12,
    )


def test_pennylane_templates_expand_and_qasm_synthesizes_matrix():
    qml = pytest.importorskip("pennylane")
    pytest.importorskip("qiskit")
    tape = qml.tape.QuantumScript(
        [
            qml.AngleEmbedding([0.23, -0.41], wires=[0, 1]),
            qml.StronglyEntanglingLayers(np.ones((2, 2, 3)) * 0.17, wires=[0, 1]),
        ],
        [],
    )
    ir = as_circuit(tape)
    device = qml.device("default.qubit", wires=2)
    expected = qml.execute((qml.tape.QuantumScript(tape.operations, [qml.state()]),), device)[0]
    np.testing.assert_allclose(statevector(ir), expected, atol=1e-12)
    matrix, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(4, 4)) + 1j * np.eye(4))
    c = Circuit(2).h(0).unitary(matrix, 1, 0)
    qasm = convert(c, source="ir", target="openqasm3").value
    restored = convert(qasm, source="openqasm3", target="ir").value
    np.testing.assert_allclose(statevector(c), statevector(restored), atol=1e-10)
