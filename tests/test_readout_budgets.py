"""Observable bounds must enclose independent dense results and fail closed on ties."""

import math

import numpy as np
import pytest

import openqml as oq


def _pair(weight):
    return oq.Circuit(2).ry(2 * math.asin(math.sqrt(weight)), 0).cx(0, 1)


def test_one_sided_schmidt_weight_and_cross_cut_coherence():
    weight = 0.01
    c = _pair(weight)
    words = tuple(oq.PauliString(s) for s in ("ZI", "XX", "II"))
    engine = oq.sim.MPSBackend(max_bond_dimension=1)
    r = engine.readout(c, words, absolute_error=(0.021, 0.201, 0))
    np.testing.assert_allclose(r.absolute_error_bounds, [2 * weight, 2 * math.sqrt(weight), 0])
    assert r.max_bond_dimension == 1
    # A two-sided observable can sense coherence to first order in sqrt(weight).
    np.testing.assert_allclose(
        oq.expectations(c, words, backend="numpy"), [0.98, 2 * math.sqrt(0.0099), 1]
    )
    with pytest.raises(oq.ResourceError, match="tolerance"):
        engine.readout(c, words, absolute_error=0.021)
    # The tolerance itself selects different ranks under one physical cap.
    assert (
        oq.sim.MPSBackend(max_bond_dimension=2)
        .readout(c, words[:1], absolute_error=0.021)
        .max_bond_dimension
        == 1
    )
    assert (
        oq.sim.MPSBackend(max_bond_dimension=2)
        .readout(c, words[1:2], absolute_error=0.021)
        .max_bond_dimension
        == 2
    )


@pytest.mark.parametrize("seed", range(7))
def test_routed_split_bounds_cover_distinct_observables_after_later_entanglers(seed):
    rng = np.random.default_rng(seed)
    c = oq.Circuit(6)
    for i in range(12):
        a, b = rng.choice(6, 2, replace=False).tolist()
        c = c.ry(rng.uniform(-0.6, 0.6), a).rx(rng.uniform(-0.6, 0.6), b).cx(a, b)
        if i % 3 == 0:
            c = c.swap(b, (b + 1) % 6)
    words = ("ZIIZII", "XXIIYZ", "IIIIIZ")
    obs = [oq.PauliString(w) for w in words]
    obs += [
        oq.Hamiltonian(
            tuple(oq.PauliString(s, k) for s, k in zip(words, (-1.1, 0.5, -0.1), strict=True))
        )
    ]
    r = oq.sim.certified_readout(
        c,
        obs,
        backend=oq.sim.MPSBackend(max_bond_dimension=2),
        absolute_error=(0.1, 0.2, 0.05, 0.2),
        strict=False,
    )
    expected = oq.expectations(c, obs, backend="numpy", fusion=False)
    assert np.all(abs(expected - r.values) <= np.asarray(r.absolute_error_bounds) + 3e-13)


def test_clifford_and_commuting_suffix_keep_control_z_on_one_side():
    c = _pair(0.01).cx(0, 1).rzz(0.81, 0, 1).h(0)
    word = oq.PauliString("XI")
    r = oq.sim.certified_readout(
        c, [word], backend=oq.sim.MPSBackend(max_bond_dimension=1), absolute_error=0.021
    )
    assert r.absolute_error_bounds == pytest.approx((0.02,))
    assert abs(r.values[0] - oq.expectations(c, [word], backend="numpy")[0]) == pytest.approx(0.02)


def test_affine_classification_cancels_shared_observables_and_abstains_on_overlap():
    c = _pair(0.01)
    spec = oq.LayerSpec(c, (), (), (oq.PauliString("XX"), oq.PauliString("ZI")))
    engine = oq.sim.MPSBackend(max_bond_dimension=1)
    # Both logits contain the same large uncertain XX signal.
    r = spec.certify_class(
        {}, head=[[100, 0], [100, 1]], bias=[0, -0.95], backend=engine, absolute_error=0.03
    )
    assert r.label == r.candidate == 1
    assert r.margin_intervals[0][0] == pytest.approx(0.03)
    r = spec.certify_class(
        {}, head=[[0, 0], [1, 0]], bias=[0, -0.1], backend=engine, absolute_error=0.01
    )
    assert r.label is None and r.margin_intervals[1][0] < 0 < r.margin_intervals[1][1]
    assert spec.certify_class({}, head=[[1, 1], [1, 1]], backend=engine).label is None


@pytest.mark.parametrize("interface", ["torch", "jax"])
def test_trained_layer_certificate_uses_framework_weights(interface):
    xp = pytest.importorskip(interface)
    x, y = oq.Parameter("x"), oq.Parameter("y")
    spec = oq.LayerSpec(
        oq.Circuit(2).ry(x + y, 0).cx(0, 1),
        ("x",),
        ("y",),
        (oq.PauliString("ZI"), oq.PauliString("XX")),
    )
    if interface == "torch":
        layer = oq.torch.QuantumLayer.from_spec(spec, device="cpu", dtype=xp.float64)
        with xp.no_grad():
            layer.weights.fill_(0.04)
        inputs, kwargs = xp.tensor([0.1], dtype=xp.float64), {}
    else:
        layer = oq.jax.QuantumLayer.from_spec(spec)
        inputs, kwargs = xp.numpy.array([0.1]), {"weights": xp.numpy.array([0.04])}
    r = layer.certify_class(inputs, **kwargs)
    assert r.label == 0
    g = layer.certified_vjp(inputs, [1, 0], absolute_error=1e-12, **kwargs)
    assert g.parameters == ("y",) and g.values[0] == pytest.approx(-math.sin(0.14))


def test_terminal_bits_and_postmeasurement_observables_across_writes():
    c = oq.Circuit(3, num_clbits=4).h(0).x(1).measure(0, 3).measure(1, 1).measure(2, 3)
    p = oq.QuantumProgram.capture(c)
    np.testing.assert_allclose(
        p.certified_readout(classical_bits=[3, 1, 0, 1], absolute_error=0).values, [0, 1, 0, 1]
    )
    words = (oq.PauliString("XII"), oq.PauliString("IZI"))
    np.testing.assert_allclose(
        p.certified_readout(words, absolute_error=0).values, [0, -1], atol=1e-14
    )
    with pytest.raises(oq.ValidationError):
        oq.sim.certified_readout(c.x(1), words, absolute_error=0)
    stored = c.classical_set(1, 0).classical_set(3, 1)
    report = oq.sim.certified_readout(stored, absolute_error=0)
    np.testing.assert_allclose(report.values, [0, 0, 0, 1])
    assert not any(report.absolute_error_bounds)
    assert oq.sim.certified_readout(stored, words[:1], absolute_error=0).values == (0,)


def test_qiskit_and_qasm_classical_outputs_share_ir_semantics():
    sdk = pytest.importorskip("qiskit")
    c = sdk.QuantumCircuit(3, 4)
    c.x(0)
    c.h(2)
    c.measure(2, 3)
    c.measure(0, 1)
    c.measure(1, 3)
    program = oq.QuantumProgram.capture(c)
    np.testing.assert_allclose(program.certified_readout(absolute_error=0).values, [0, 1, 0, 0])
    from qiskit import qasm2

    p = oq.QuantumProgram.from_qasm(qasm2.dumps(c), version=2)
    assert (
        p.certified_readout(absolute_error=0).values
        == program.certified_readout(absolute_error=0).values
    )


def test_cirq_named_multiwire_key_respects_explicit_qubit_and_bit_order():
    cirq = pytest.importorskip("cirq")
    a, b = cirq.NamedQubit("a"), cirq.NamedQubit("b")
    circuit = cirq.Circuit(cirq.X(b), cirq.measure(b, a, key="pair"))
    result = oq.sim.certified_readout(
        circuit,
        source_options={"qubit_order": [b, a], "measurement_map": {"pair": (3, 1)}},
        absolute_error=0,
    )
    sampled = cirq.Simulator().run(circuit, repetitions=1).measurements["pair"][0]
    np.testing.assert_allclose(result.values, [0, sampled[1], 0, sampled[0]])


def test_pennylane_analytic_pauli_returns_include_measurement_only_named_wires():
    qml = pytest.importorskip("pennylane")
    tape = qml.tape.QuantumScript(
        [qml.RY(0.23, "signal")],
        [
            qml.expval(qml.Z("idle") @ qml.X("signal")),
            qml.expval(0.3 * qml.Z("signal") - 0.7 * qml.Identity("idle")),
        ],
    )
    got = oq.QuantumProgram.capture(tape).certified_readout(
        source_options={"wire_order": ["idle", "signal"]}, absolute_error=0
    )
    expected = qml.execute((tape,), qml.device("default.qubit", wires=["idle", "signal"]))[0]
    np.testing.assert_allclose(got.values, expected, atol=1e-13)
    assert got.simulated_wires == (0, 1)
    for unsupported in (
        qml.tape.QuantumScript(tape.operations, [qml.probs(wires="signal")]),
        qml.tape.QuantumScript(tape.operations, tape.measurements, shots=10),
    ):
        with pytest.raises(oq.ConversionError):
            oq.sim.certified_readout(unsupported, absolute_error=0)


def test_price_in_payoff_units_and_causal_cone():
    weight = 0.01
    c = oq.Circuit(96).ry(2 * math.asin(math.sqrt(weight)), 0).cx(0, 95)
    # These gates touch none of the causal ancestors of the marked measurement.
    for k in range(2, 72, 2):
        c = c.h(k).cx(k, k + 1)
    engine = oq.sim.MPSBackend(max_bond_dimension=1, max_discarded_weight=0.1)
    r = oq.sim.certified_readout(
        c, [oq.PauliString("I" * 95 + "Z")], backend=engine, absolute_error=0.03
    )
    assert r.simulated_wires == (0, 95) and r.routing_swaps == 0
    oracle = oq.AmplitudeOracle(c, 95)
    price = oq.finance.price(oracle=oracle, bounds=(0, 100), error=1.01, backend=engine)
    np.testing.assert_allclose(price.interval, [0, 1], atol=1e-12)
    with pytest.raises(oq.ResourceError, match="tolerance"):
        oq.finance.price(oracle=oracle, bounds=(0, 100), error=0.9, backend=engine)


def test_plan_admission_precedes_tensor_allocation_and_unbounded_bit_iterators(monkeypatch):
    from openqml.sim import _mps

    monkeypatch.setattr(_mps, "_two_site", lambda *args: pytest.fail("admission must precede SVD"))
    c = _pair(0.04)
    with pytest.raises(oq.ResourceError):
        oq.sim.certified_readout(
            c, [oq.PauliString("ZZ")], backend=oq.sim.MPSBackend(max_work=2), absolute_error=0.1
        )

    def infinite():
        while True:
            yield 0

    with pytest.raises(oq.ResourceError):
        oq.sim.certified_readout(
            oq.Circuit(1, num_clbits=1).measure(0, 0),
            classical_bits=infinite(),
            backend=oq.sim.MPSBackend(max_work=32),
            absolute_error=0.01,
        )


def test_certified_shift_vjp_covers_nonlinear_reuse_and_controlled_four_point_shifts():
    x, y, phase = oq.Parameter("x"), oq.Parameter("y"), oq.Parameter("phase")
    c = (
        oq.Circuit(5)
        .h(1)
        .ry(x * y, 0)
        .cx(0, 3)
        .crx(x.sin(), 1, 4)
        .ry(y + x, 4)
        .crz(y * y, 3, 1)
        .with_global_phase(phase)
    )
    obs = (oq.PauliString("ZIIII"), oq.PauliString("IIIIZ"))
    b = {"x": 0.25, "y": -0.41, "phase": 0.3}
    names = ("y", "phase", "x")
    plan = oq.plan_readout(c, obs)
    for cap in (1, 8):
        engine = oq.sim.MPSBackend(max_bond_dimension=cap)
        r = plan.certified_vjp(
            b, [0.9, -0.35], parameters=names, absolute_error=0.04, backend=engine, strict=False
        )

        def cost(row):
            return np.dot(oq.expectations(c, obs, row, backend="numpy", fusion=False), [0.9, -0.35])

        eps = 1e-5
        exact = np.array(
            [(cost(b | {n: b[n] + eps}) - cost(b | {n: b[n] - eps})) / (2 * eps) for n in names]
        )
        assert np.all(
            np.abs(np.asarray(r.values) - exact) <= np.asarray(r.absolute_error_bounds) + 2e-9
        )
        assert (
            r.parameters == names
            and r.values[1] == r.absolute_error_bounds[1] == 0
            and r.evaluations > 4
        )
        if cap == 8:
            assert max(r.absolute_error_bounds) <= 0.04


def test_cancelling_loss_needs_no_shift_calls_or_shared_quantum_state(monkeypatch):
    x = oq.Parameter("x")
    c = oq.Circuit(4).ry(x, 0).cx(0, 3)
    word = oq.PauliString("ZIII")
    spec = oq.LayerSpec(c, (), ("x",), (word, word))
    monkeypatch.setattr(
        oq.sim.MPSBackend, "readout", lambda *a, **k: pytest.fail("zero gradient submitted")
    )
    result = spec.certified_vjp({"x": 0.23}, [1, -1], absolute_error=0)
    assert result.evaluations == 0 and result.intervals == ((0.0, 0.0),)


def test_prepared_plan_rebinds_values_and_rejects_invalid_eliminated_parameter():
    x, y = oq.Parameter("x"), oq.Parameter("y")
    c = oq.Circuit(4).ry(x, 0).rx(y, 3)
    plan = oq.plan_readout(c, [oq.PauliString("ZIII")])
    for theta in (0.2, 0.6):
        assert plan.evaluate({"x": theta, "y": 0.8}, absolute_error=0).values[0] == pytest.approx(
            np.cos(theta)
        )
    with pytest.raises(oq.ValidationError):
        plan.evaluate({"x": 0.3, "y": float("nan")}, absolute_error=0.1)
