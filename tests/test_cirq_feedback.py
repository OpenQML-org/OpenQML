"""Cirq feedback semantics against Boolean references and actual SDK sampling."""

import numpy as np
import pytest

from openqml import (
    Circuit,
    CirqDensityBackend,
    ClassicalCondition,
    ConversionError,
    Parameter,
    PauliString,
    QuantumProgram,
    ShotBackend,
    ValidationError,
    convert,
    density_matrix,
    diagnose,
    expectations,
    sample,
)


def feedback_program():
    x, w = Parameter("x"), Parameter("w")
    return (
        Circuit(2)
        .ry(x, 0)
        .measure(0, 0)
        .if_else(
            ClassicalCondition((0,), 1),
            Circuit(2).ry(w * w, 1).with_global_phase(w / 3),
            Circuit(2).rx(w / 2, 1),
        )
    )


@pytest.mark.parametrize("index", [-1, 0])
def test_key_condition_indices_preserve_single_record(index):
    cirq = pytest.importorskip("cirq")
    q = cirq.LineQubit.range(2)
    native = cirq.Circuit(
        cirq.ry(0.7)(q[0]),
        cirq.measure(q[0], key="c0"),
        cirq.X(q[1]).with_classical_controls(cirq.KeyCondition(cirq.MeasurementKey("c0"), index)),
    )
    ir = convert(native, target="ir").value
    expected = np.diag([np.cos(0.35) ** 2, 0, 0, np.sin(0.35) ** 2])
    np.testing.assert_allclose(density_matrix(ir), expected, atol=1e-13)


@pytest.mark.parametrize(
    "kind", ["eq", "ne", "and", "or", "xor", "equivalent", "implies", "sum", "lt"]
)
def test_boolean_predicates_match_independent_truth_tables(kind):
    cirq = pytest.importorskip("cirq")
    sp = pytest.importorskip("sympy")
    a, b = sp.symbols("c0 c1")
    expressions = {
        "eq": sp.Eq(a, b),
        "ne": sp.Ne(a, b),
        "and": sp.And(sp.Eq(a, 1), sp.Eq(b, 0)),
        "or": sp.Or(sp.Eq(a, 1), sp.Eq(b, 0)),
        "xor": sp.Xor(sp.Eq(a, 1), sp.Eq(b, 1)),
        "equivalent": sp.Equivalent(sp.Eq(a, 1), sp.Eq(b, 0)),
        "implies": sp.Implies(sp.Eq(a, 1), sp.Eq(b, 0)),
        "sum": sp.Eq(a + b, 1),
        "lt": sp.Lt(a, b),
    }
    references = {
        "eq": lambda x, y: x == y,
        "ne": lambda x, y: x != y,
        "and": lambda x, y: x == 1 and y == 0,
        "or": lambda x, y: x == 1 or y == 0,
        "xor": lambda x, y: bool(x) != bool(y),
        "equivalent": lambda x, y: bool(x) == (y == 0),
        "implies": lambda x, y: not x or not y,
        "sum": lambda x, y: x + y == 1,
        "lt": lambda x, y: x < y,
    }
    q = cirq.LineQubit.range(3)
    native = cirq.Circuit(
        cirq.H.on_each(q[:2]),
        cirq.measure(q[0], key="c0"),
        cirq.measure(q[1], key="c1"),
        cirq.X(q[2]).with_classical_controls(expressions[kind]),
    )
    ir = convert(native, target="ir").value
    probabilities = np.zeros(8)
    for x in (0, 1):
        for y in (0, 1):
            probabilities[4 * x + 2 * y + references[kind](x, y)] = 0.25
    np.testing.assert_allclose(density_matrix(ir), np.diag(probabilities), atol=1e-13)


@pytest.mark.parametrize("predicate", ["nonzero", "eq", "and"])
def test_multiqubit_key_mapping_order_and_real_native_joint_counts(predicate):
    cirq = pytest.importorskip("cirq")
    sp = pytest.importorskip("sympy")
    q = cirq.LineQubit.range(3)
    condition = {
        "nonzero": "syndrome",
        "eq": sp.Eq(sp.Symbol("syndrome"), 2),
        "and": sp.And(sp.Ne(sp.Symbol("syndrome"), 0), sp.Eq(sp.Symbol("syndrome"), 2)),
    }[predicate]
    native = cirq.Circuit(
        cirq.H.on_each(q[:2]),
        cirq.measure(q[1], q[0], key="syndrome"),
        cirq.X(q[2]).with_classical_controls(condition),
        cirq.measure(q[2], key="out"),
    )
    result = convert(
        native, target="ir", source_options={"measurement_map": {"syndrome": (2, 0), "out": (1,)}}
    )
    assert len([d for d in result.diagnostics if d.code == "MEASUREMENT_KEY_MAPPED"]) == 2
    records = cirq.Simulator(seed=73).run(native, repetitions=4000).measurements
    native_counts = {}
    for syndrome, out in zip(records["syndrome"], records["out"], strict=True):
        key = f"{syndrome[1]}{out[0]}{syndrome[0]}"
        native_counts[key] = native_counts.get(key, 0) + 1
    actual = sample(result.value, 4000, seed=72)
    for key in set(actual) | set(native_counts):
        assert actual.get(key, 0) / 4000 == pytest.approx(
            native_counts.get(key, 0) / 4000, abs=0.035
        )


@pytest.mark.parametrize("target", ["cirq", "qiskit", "openqasm3", "json"])
def test_symbolic_feedback_native_roundtrip_and_full_branch_phases(target):
    pytest.importorskip("cirq")
    if target in {"qiskit", "openqasm3"}:
        pytest.importorskip("qiskit_qasm3_import")
    original = feedback_program()
    cirq_native = convert(original, target="cirq").value
    ir = QuantumProgram.capture(cirq_native).to_ir()
    assert ir.parameters == ("w", "x")
    assert any(b.global_phase.parameters for _, op in ir.walk_operations() for b in op.blocks)
    exported = convert(cirq_native, target=target)
    restored = convert(exported.value, source=target, target="ir").value
    from openqml.compiler import rename_parameters

    restored = rename_parameters(
        restored, {alias: name for name, alias in exported.parameter_map.items()}
    )
    values = {"x": 0.8, "w": 0.6}
    np.testing.assert_allclose(
        density_matrix(restored, values), density_matrix(original, values), atol=2e-12
    )


def test_nested_atomic_blocks_channels_reset_and_unwritten_zero_bits():
    pytest.importorskip("cirq")
    c = (
        Circuit(3)
        .h(0)
        .measure(0, 0)
        .if_else(
            ClassicalCondition((0, 2), 1),
            Circuit(3)
            .h(1)
            .if_else(
                ClassicalCondition((0,), 0),
                Circuit(3).x(1),
                Circuit(3).amplitude_damping(0.2, 1).reset(2),
            ),
            Circuit(3).x(2),
        )
        .if_else(ClassicalCondition((1,), 0), Circuit(3).ry(0.4, 1))
        .measure(1, 1)
        .measure(2, 2)
    )
    assert not diagnose(c, "cirq")
    native = convert(c, target="cirq").value
    restored = convert(native, target="ir").value
    np.testing.assert_allclose(density_matrix(restored), density_matrix(c), atol=1e-13)
    from openqml import execute

    # Native sampling runs the exported feedback/channel/reset operations.
    result = execute(c, shots=4000, backend=CirqDensityBackend(), seed=31)
    expected = sample(c, 4000, seed=32)
    for bits in set(result.counts) | set(expected):
        assert result.counts.get(bits, 0) / 4000 == pytest.approx(
            expected.get(bits, 0) / 4000, abs=0.04
        )


def test_native_sampler_repeated_storage_writes_use_event_at_each_read():
    pytest.importorskip("cirq")
    c = (
        Circuit(2)
        .h(0)
        .measure(0, 5)
        .if_else(ClassicalCondition((5,), 1), Circuit(2).x(1))
        .reset(0)
        .measure(0, 5)
        .if_else(ClassicalCondition((5, 4), 0), Circuit(2).ry(0.6, 1))
        .measure(1, 1)
    )
    # Public conversion preserves record schemas and still diagnoses overwrite;
    # the sampler explicitly replays private event records into final storage.
    assert "REPEATED_MEASUREMENT_KEY" in {d.code for d in diagnose(c, "cirq")}
    counts = sample(c, 5000, backend=CirqDensityBackend(), seed=12)
    assert set(counts) <= {"000000", "010000"}
    assert counts.get("010000", 0) / 5000 == pytest.approx(0.5, abs=0.035)
    assert sample(c, 100, backend=CirqDensityBackend(), seed=12, classical_bits=(5, 4)) == {
        "00": 100
    }
    with pytest.raises(ValidationError, match="exact feedback"):
        expectations(c, [PauliString("IZ")], backend=CirqDensityBackend())


@pytest.mark.parametrize("issue", ["missing", "history", "expression", "custom", "repeat"])
def test_unsupported_condition_contracts_are_diagnosed(issue):
    cirq = pytest.importorskip("cirq")
    sp = pytest.importorskip("sympy")
    q = cirq.LineQubit.range(2)
    if issue == "custom":

        class Custom(cirq.Condition):
            @property
            def keys(self):
                return (cirq.MeasurementKey("c0"),)

            def replace_key(self, current, replacement):
                return self

            def resolve(self, classical_data):
                raise AssertionError("must not execute custom code")

            @property
            def qasm(self):
                return ""

        condition, code = Custom(), "CLASSICAL_CONDITION"
    else:
        condition, code = {
            "missing": ("absent", "CLASSICAL_KEY_UNDEFINED"),
            "history": (cirq.KeyCondition(cirq.MeasurementKey("c0"), 1), "MEASUREMENT_HISTORY"),
            "expression": (sp.Eq(sp.sin(sp.Symbol("c0")), 0), "CLASSICAL_EXPRESSION"),
            "repeat": ("c0", "REPEATED_MEASUREMENT_KEY"),
        }[issue]
    ops = [cirq.measure(q[0], key="c0")]
    if issue == "repeat":
        ops.append(cirq.measure(q[0], key="c0"))
    ops.append(cirq.X(q[1]).with_classical_controls(condition))
    with pytest.raises(ConversionError, match=code):
        convert(cirq.Circuit(ops), target="ir")


@pytest.mark.parametrize(
    "mapping",
    [
        {"a": ()},
        {"a": (0, 0)},
        {"a": (True,)},
        {"a": (-1,)},
        {"a": {0}},
        {"a": (0, 1)},
        {"unused": (0,)},
        {"a": (1,), "c1": (1,)},
    ],
)
def test_invalid_measurement_maps_are_explicit_errors(mapping):
    cirq = pytest.importorskip("cirq")
    q = cirq.LineQubit.range(2)
    native = cirq.Circuit(cirq.measure(q[0], key="a"), cirq.measure(q[1], key="c1"))
    with pytest.raises(ConversionError, match="MEASUREMENT_MAP|MEASUREMENT_KEY"):
        convert(native, target="ir", source_options={"measurement_map": mapping})


def test_condition_limits_precede_dense_fallback_and_expansion_is_bounded():
    cirq = pytest.importorskip("cirq")
    sp = pytest.importorskip("sympy")

    class Expensive(cirq.Gate):
        def _num_qubits_(self):
            return 1

        def _unitary_(self):
            raise AssertionError("limit must precede dense fallback")

    q = cirq.LineQubit.range(9)
    native = cirq.Circuit(
        cirq.measure(q[0], key="c0"), Expensive()(q[1]).with_classical_controls("c0")
    )
    with pytest.raises(ConversionError, match="CONDITION_RESOURCE_LIMIT"):
        convert(native, target="ir", source_options={"max_condition_bits": 0})
    native = cirq.Circuit(
        [cirq.measure(q[i], key=f"c{i}") for i in range(8)],
        cirq.X(q[8]).with_classical_controls(
            sp.Xor(*(sp.Eq(sp.Symbol(f"c{i}"), 1) for i in range(8)))
        ),
    )
    with pytest.raises(ConversionError, match="EXPANSION_LIMIT"):
        convert(native, target="ir", source_options={"max_operations": 30})
    with pytest.raises(ConversionError, match="CONDITION_RESOURCE_LIMIT"):
        convert(native, target="ir", source_options={"max_condition_bits": 17})


def test_atomic_measurement_body_cannot_be_flattened_into_rechecked_conditions():
    pytest.importorskip("cirq")
    c = (
        Circuit(1)
        .h(0)
        .measure(0, 0)
        .if_else(ClassicalCondition((0,), 1), Circuit(1).x(0).measure(0, 0).x(0))
    )
    diagnostics = diagnose(c, "cirq")
    assert any(
        d.code == "CONTROL_FLOW_UNSUPPORTED" and "operations[2]" in d.location for d in diagnostics
    )
    with pytest.raises(ConversionError, match="CONTROL_FLOW_UNSUPPORTED"):
        convert(c, target="cirq")
    with pytest.raises(ConversionError, match="CONTROL_FLOW_UNSUPPORTED"):
        sample(c, 10, backend=CirqDensityBackend())


def test_cirq_import_trains_in_torch_with_second_order_and_shared_optimizer():
    pytest.importorskip("cirq")
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    native = convert(feedback_program(), target="cirq").value
    layer = QuantumLayer.from_program(native, ["x"], dtype=torch.float64)
    inputs = torch.tensor([[0.4], [1.1]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)

    def function(x, w):
        return torch.func.functional_call(layer, {"weights": w}, (x,))

    assert torch.autograd.gradcheck(function, (inputs, weights))
    assert torch.autograd.gradgradcheck(function, (inputs, weights))
    model = torch.nn.Sequential(layer, torch.nn.Linear(2, 1, dtype=torch.float64))
    with torch.no_grad():
        layer.weights.fill_(0.7)
        model[1].weight.fill_(0.4)
        model[1].bias.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    target = torch.tensor([[0.1], [-0.2]], dtype=torch.float64)
    initial = float(((model(inputs) - target) ** 2).mean().detach())
    for _ in range(25):
        optimizer.zero_grad()
        loss = ((model(inputs) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < initial * 0.25


def test_cirq_import_jax_jit_hessian_and_native_shot_training():
    pytest.importorskip("cirq")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    original = feedback_program()
    native = convert(original, target="cirq").value
    a = QuantumLayer.from_program(native, ["x"])
    b = QuantumLayer(original, ["x"], ["w"])
    x, w = jnp.array([[0.6]], dtype=jnp.float32), jnp.array([0.8], dtype=jnp.float32)
    np.testing.assert_allclose(
        jax.jit(jax.hessian(lambda t: a(t, x).sum()))(w),
        jax.hessian(lambda t: b(t, x).sum())(w),
        atol=2e-6,
    )
    # The native Cirq sampler supports feedback; its exact trajectory API does
    # not become an ensemble estimator. ShotBackend supplies that distinction.
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer as TorchLayer

    layer = TorchLayer.from_program(
        native, ["x"], dtype=torch.float64, backend=ShotBackend(CirqDensityBackend(), 4000, seed=8)
    )
    with torch.no_grad():
        layer.weights.fill_(0.8)
    value = layer(torch.tensor([[0.6]], dtype=torch.float64))[:, 1].sum()
    value.backward()
    prob = np.sin(0.6 / 2) ** 2
    analytic = -prob * 1.6 * np.sin(0.8**2) - (1 - prob) * 0.5 * np.sin(0.4)
    assert layer.weights.grad.item() == pytest.approx(analytic, abs=0.055)
