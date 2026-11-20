"""Independent execution and accuracy checks on the public simulator entry points."""

import math

import numpy as np
import pytest

import openqml as oq


def _training_circuit():
    x, y = oq.Parameter("x"), oq.Parameter("y")
    circuit = oq.Circuit(5).h(0).ry(x.sin() * y, 4).cx(4, 1)
    return circuit.u(x, y, x + y, 1).rxx(y, 3, 0).crz(x / 3, 0, 4).with_global_phase(x / 5)


def test_native_torch_inverse_sweep_matches_complex_gradients_and_hessian():
    torch = pytest.importorskip("torch")
    circuit = _training_circuit()
    x = torch.tensor([[0.21], [-0.31]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.3, -0.71, 0.5], dtype=x.dtype, requires_grad=True)

    def evaluate(mode, higher):
        state = oq.statevector(circuit, {"x": x, "y": y}, backend="torch", checkpoint_interval=mode)
        score = (state.real[..., ::3] - 0.7 * state.imag[..., 1::3].sum(-1, keepdim=True)).sum()
        grads = torch.autograd.grad(score, (x, y), create_graph=higher)
        if higher:
            grads = torch.autograd.grad(sum(v.sum() for v in grads), (x, y))
        return grads

    for higher in (False, True):
        torch.testing.assert_close(evaluate("adjoint", higher), evaluate(None, higher))


def test_torch_quantum_layer_retains_constant_full_registers_across_depth():
    torch = pytest.importorskip("torch")
    from openqml.qml.interfaces.torch import QuantumLayer

    def saved_registers(depth):
        width, seen = 7, []
        x, y = oq.Parameter("x"), oq.Parameter("y")
        circuit = oq.Circuit(width)
        for _ in range(depth):
            circuit = circuit.ry(x + y, 0).cx(0, 6).rz(y, 6).barrier()
        layer = QuantumLayer(
            circuit,
            ["x"],
            ["y"],
            [oq.PauliString("Z" + "I" * 6)],
            dtype=torch.float64,
            device="cpu",
            checkpoint_interval="adjoint",
        )

        def pack(value):
            if value.is_complex() and value.shape[-1:] == (1 << width,):
                seen.append(value.numel())
            return value

        inputs = torch.tensor([[0.3], [-0.2]], dtype=torch.float64, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda v: v):
            output = layer(inputs)
        output.sum().backward()
        assert inputs.grad is not None and layer.weights.grad is not None
        return len(seen)

    assert 0 < saved_registers(2) == saved_registers(24)


def test_jax_reversible_jit_vjp_and_shared_nonlinear_inputs():
    jax = pytest.importorskip("jax")
    circuit = _training_circuit()

    def score(x, y, mode):
        state = oq.statevector(circuit, {"x": x, "y": y}, backend="jax", checkpoint_interval=mode)
        return (state.real[..., ::2] + 0.3 * state.imag[..., 1::2]).sum()

    x, y = jax.numpy.array([[0.3], [-0.4]]), jax.numpy.array([0.1, -0.3, 0.74])
    for a, b in zip(
        jax.jit(jax.grad(lambda a, b: score(a, b, "adjoint"), (0, 1)))(x, y),
        jax.grad(lambda a, b: score(a, b, None), (0, 1))(x, y),
        strict=True,
    ):
        np.testing.assert_allclose(a, b, atol=2e-6)


def test_dag_fusion_preserves_barriers_wire_order_and_reduces_state_traversals():
    from openqml.sim._fusion import fusion_program

    circuit = oq.Circuit(6)
    for _ in range(8):
        circuit = circuit.cx(5, 0).cx(4, 1).ry(0.13, 5).rx(0.41, 4)
    circuit = circuit.barrier().cx(0, 3)
    groups = fusion_program(circuit.operations, 6).groups
    assert len(groups) == 3
    assert groups[-1].qubits == (0, 3)
    np.testing.assert_allclose(
        oq.statevector(circuit, backend="numpy"),
        oq.statevector(circuit, backend="numpy", fusion=False),
        atol=2e-14,
    )


def test_routed_mps_preserves_asymmetric_readouts_and_both_operand_orders():
    rng = np.random.default_rng(43)
    matrix = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    c = oq.Circuit(6).h(4).ry(0.4, 0).cx(4, 0).swap(0, 3).unitary(matrix, 5, 1).cy(1, 5).swap(5, 2)
    words = [oq.PauliString(s) for s in ("XIZIYI", "IIIIZI", "IZIIII", "IIZIII", "ZZZZZZ")]
    exact = oq.expectations(c, words, backend="numpy")
    report = oq.sim.MPSBackend().readout(c, words)
    assert report.routing_swaps > 0 and report.trace_distance_bound == 0
    np.testing.assert_allclose(report.values, exact, atol=1e-13)
    np.testing.assert_allclose(oq.expectations(c, words, backend="mps"), exact, atol=1e-13)


def test_schmidt_certificate_for_known_truncation_and_readout_norm():
    angle = 0.32
    circuit = oq.Circuit(3).ry(2 * angle, 0).cx(0, 2)
    obs = (
        oq.PauliString("ZII"),
        oq.PauliString("III"),
        oq.Hamiltonian(
            (oq.PauliString("XIX", coefficient=0.75), oq.PauliString("III", coefficient=4))
        ),
    )
    engine = oq.sim.MPSBackend(max_bond_dimension=1, max_discarded_weight=0.0)
    assert not engine.capabilities.exact_expectations
    report = engine.readout(circuit, obs)
    assert report.trace_distance_bound == pytest.approx(math.sin(angle))
    np.testing.assert_allclose(report.absolute_error_bounds, np.sin(angle) * np.array([2, 0, 1.5]))
    exact = oq.expectations(circuit, obs, backend="numpy")
    assert np.all(np.abs(report.values - exact) <= np.asarray(report.absolute_error_bounds) + 1e-13)
    with pytest.raises(oq.ResourceError):
        oq.sim.MPSBackend(max_bond_dimension=1).readout(circuit, obs)


def test_routed_mps_handles_wide_register_under_budget_and_admits_before_svd(monkeypatch):
    from openqml.sim import _mps

    width = 96
    c = oq.Circuit(width).ry(0.73, 0).cx(0, width - 1).swap(1, width - 1)
    obs = [oq.PauliString("I" + "Z" + "I" * (width - 2))]
    report = oq.sim.MPSBackend().readout(c, obs, options=oq.ExecutionOptions(max_memory_mb=1))
    assert report.values[0] == pytest.approx(np.cos(0.73))
    assert report.routing_swaps == width - 2
    monkeypatch.setattr(_mps, "_two_site", lambda *args: pytest.fail("unadmitted SVD"))
    with pytest.raises(oq.ResourceError):
        oq.sim.MPSBackend(max_work=96).readout(c, obs)
    with pytest.raises(oq.ResourceError):
        oq.sim.MPSBackend().readout(c, obs, options=oq.ExecutionOptions(max_memory_mb=0.001))


def test_routed_payoff_pricing_propagates_truncation_to_discounted_units():
    circuit = oq.Circuit(72).ry(0.64, 0).cx(0, 71)
    oracle = oq.AmplitudeOracle(circuit, 71)
    kwargs = dict(oracle=oracle, bounds=(-2, 3), discount=0.95, max_memory_mb=1)
    actual = 0.95 * (-2 + 5 * np.sin(0.32) ** 2)
    exact = oq.finance.price(backend="mps", **kwargs)
    assert exact.value == pytest.approx(actual)
    engine = oq.sim.MPSBackend(max_bond_dimension=1, max_discarded_weight=0)
    result = oq.finance.price(backend=engine, strict=False, **kwargs)
    assert result.interval[0] <= actual <= result.interval[1]
    assert result.interval[1] - result.value == pytest.approx(0.95 * 5 * np.sin(0.32))
    assert result.oracle_queries == 0
    with pytest.raises(oq.ResourceError, match="tolerance"):
        oq.finance.price(backend=engine, error=0.1, **kwargs)


@pytest.mark.parametrize("framework", ["qiskit", "cirq"])
def test_imported_circuit_runs_through_dense_and_routed_readouts(framework):
    native = pytest.importorskip(framework)
    if framework == "qiskit":
        circuit = native.QuantumCircuit(4)
        circuit.ry(0.41, 3)
        circuit.cx(3, 0)
        circuit.swap(0, 1)
        circuit.rz(-0.77, 1)
    else:
        q = native.LineQubit.range(4)
        circuit = native.Circuit(
            native.I(q[2]),
            native.ry(0.41)(q[3]),
            native.CNOT(q[3], q[0]),
            native.SWAP(q[0], q[1]),
            native.rz(-0.77)(q[1]),
        )
    ir = oq.as_circuit(circuit, framework=framework)
    obs = tuple(oq.PauliString("I" * i + "Z" + "I" * (3 - i)) for i in range(4))
    reference = oq.expectations(ir, obs, backend="numpy", fusion=False)
    np.testing.assert_allclose(oq.sim.MPSBackend().readout(ir, obs).values, reference, atol=1e-13)
    np.testing.assert_allclose(
        sorted(reference), sorted([1, 1, np.cos(0.41), np.cos(0.41)]), atol=1e-13
    )


def test_phase_only_parameter_retains_native_second_derivative():
    torch = pytest.importorskip("torch")
    p = oq.Parameter("p")
    circuit = oq.Circuit(3).h(0).with_global_phase(p)
    x = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    state = oq.statevector(circuit, {"p": x}, backend="torch", checkpoint_interval="adjoint")
    (first,) = torch.autograd.grad(state[0].imag, x, create_graph=True)
    (second,) = torch.autograd.grad(first, x)
    torch.testing.assert_close(second, -x.sin() / math.sqrt(2))
