"""Shared symbolic programs retain real domains, native AD and bounded graph work."""

import pickle
import weakref

import numpy as np
import pytest

import openqml as oq
from openqml.parameters import expression


def shared(value, depth):
    for _ in range(depth):
        value = (value + value).sin()
    return value


def reference(value, depth, first=1.0, second=0.0):
    for _ in range(depth):
        angle = 2 * value
        second = -4 * np.sin(angle) * first**2 + 2 * np.cos(angle) * second
        first = 2 * np.cos(angle) * first
        value = np.sin(angle)
    return value, first, second


def graph_size(root):
    seen, pending = set(), [root]
    while pending:
        value = pending.pop()
        if id(value) not in seen:
            seen.add(id(value))
            pending.extend(value.args)
    return len(seen)


def test_shared_evaluation_calls_each_transcendental_once(monkeypatch):
    depth = 32
    program = shared(expression(oq.Parameter("x")), depth)
    x = np.linspace(-0.3, 0.4, 16)
    expected = reference(x, depth)[0]
    original, calls = np.sin, []

    def counted(value):
        calls.append(None)
        return original(value)

    monkeypatch.setattr(np, "sin", counted)
    np.testing.assert_allclose(program.evaluate({"x": x}), expected, atol=0, rtol=0)
    assert len(calls) == depth


def test_symbolic_derivative_keeps_shared_structure_and_matches_independent_chain_rule():
    depth = 30
    program = shared(expression(oq.Parameter("x")), depth)
    derivative = program.derivative("x")
    assert graph_size(derivative) <= 8 * depth
    assert derivative.parameters == frozenset({"x"})
    x = np.array([0.12, 0.3, -0.2])
    np.testing.assert_allclose(derivative.evaluate({"x": x}), reference(x, depth)[1], rtol=2e-13)
    assert program.derivative("unused").evaluate({}) == 0


def test_second_symbolic_derivative_matches_independent_recurrence():
    program = shared(expression(oq.Parameter("x")), 12)
    x = np.array([0.04, 0.13, -0.2])
    actual = program.derivative("x").derivative("x").evaluate({"x": x})
    np.testing.assert_allclose(actual, reference(x, 12)[2], rtol=2e-12, atol=1e-9)


@pytest.mark.parametrize("operation", ["evaluate", "bind", "substitute", "derivative", "equality"])
def test_deep_graphs_do_not_depend_on_python_recursion_limit(operation):
    value = expression(oq.Parameter("x"))
    expected, expected_derivative = 0.3, 1.0
    for _ in range(1600):
        value = value.sin()
        expected_derivative *= np.cos(expected)
        expected = np.sin(expected)
    if operation == "evaluate":
        actual = value.evaluate({"x": 0.3})
    elif operation == "bind":
        actual = value.bind({"x": 0.3}).constant
    elif operation == "substitute":
        actual = value.substitute({"x": oq.Parameter("y")}).evaluate({"y": 0.3})
    elif operation == "derivative":
        actual = value.derivative("x").evaluate({"x": 0.3})
        expected = expected_derivative
    else:
        other = expression(oq.Parameter("x"))
        for _ in range(1600):
            other = other.sin()
        assert value == other and hash(value) == hash(other)
        assert value != value.substitute({"x": oq.Parameter("y")})
        assert len({value, other}) == 1
        return
    np.testing.assert_allclose(actual, expected, rtol=2e-13)


def test_numeric_evaluation_releases_dead_array_temporaries(monkeypatch):
    import openqml.parameters as parameters

    program = shared(expression(oq.Parameter("x")), 400)
    references, peaks = [], []
    original = parameters._apply

    def tracked(op, args):
        result = original(op, args)
        if isinstance(result, np.ndarray):
            references.append(weakref.ref(result))
            peaks.append(sum(reference() is not None for reference in references))
        return result

    monkeypatch.setattr(parameters, "_apply", tracked)
    result = program.evaluate({"x": np.linspace(0.1, 0.2, 128)})
    assert np.isfinite(result).all()
    assert max(peaks) <= 4
    assert sum(reference() is not None for reference in references) == 1


def test_rewrites_preserve_shared_children_and_simultaneous_replacement():
    x, y, z = (oq.Parameter(name) for name in ("x", "y", "z"))
    inner = (x.sin() + y).cos()
    program = inner * inner
    for rewritten in (program.bind({"x": 0.3}), program.substitute({"x": z.sin()})):
        assert rewritten.args[0] is rewritten.args[1]
    assert program.bind({}) is program
    assert program.substitute({}) is program
    assert program.substitute({"x": x, "y": y}) is program
    circuit = oq.Circuit(1).ry(program, 0).with_global_phase(program)
    assert circuit.substitute({"x": x, "y": y}) is circuit
    swapped = (x.sin() + y).substitute({"x": y, "y": x})
    np.testing.assert_allclose(swapped.evaluate({"x": 0.2, "y": 0.4}), np.sin(0.4) + 0.2)
    replaced = program.substitute({"x": x + 1})
    np.testing.assert_allclose(
        replaced.evaluate({"x": 0.2, "y": 0.4}), np.cos(np.sin(1.2) + 0.4) ** 2
    )


@pytest.mark.parametrize("method", ["bind", "substitute"])
def test_parameter_free_raw_nodes_are_folded_and_validated_when_rewritten(method):
    valid = oq.Expression(op="sin", args=(expression(0.4),))
    assert getattr(valid, method)({}).constant == pytest.approx(np.sin(0.4))
    for invalid in (
        oq.Expression(op="log", args=(expression(-1),)),
        oq.Expression(op="div", args=(expression(1), expression(0))),
    ):
        with np.errstate(all="ignore"), pytest.raises(oq.ValidationError):
            getattr(invalid, method)({})
        circuit = oq.Circuit(1).ry(invalid, 0)
        with np.errstate(all="ignore"), pytest.raises(oq.ValidationError):
            getattr(circuit, method)({})


def test_circuit_forest_shares_rewrites_across_operations_phases_and_feedback():
    angle = shared(expression(oq.Parameter("x")), 24)
    circuit = (
        oq.Circuit(2)
        .ry(angle, 0)
        .measure(0, 0)
        .if_else(
            oq.ClassicalCondition((0,), 1),
            oq.Circuit(2).rx(angle, 1).with_global_phase(angle),
            oq.Circuit(2).ry(angle, 1).with_global_phase(angle),
        )
        .with_global_phase(angle)
    )
    for rewritten in (circuit.bind({"x": 0.3}), circuit.substitute({"x": oq.Parameter("w")})):
        root = rewritten.operations[0].params[0]
        assert rewritten.global_phase is root
        for block in rewritten.operations[2].blocks:
            assert block.operations[0].params[0] is root and block.global_phase is root
    value = reference(0.3, 24)[0]
    expected = (
        oq.Circuit(2)
        .ry(value, 0)
        .measure(0, 0)
        .if_else(
            oq.ClassicalCondition((0,), 1), oq.Circuit(2).rx(value, 1), oq.Circuit(2).ry(value, 1)
        )
    )
    np.testing.assert_allclose(
        oq.density_matrix(circuit, {"x": 0.3}), oq.density_matrix(expected), atol=1e-12
    )


@pytest.mark.parametrize("operation", ["div", "sqrt", "log", "pow", "exp"])
def test_real_domain_failures_are_not_hidden_by_shared_graphs(operation):
    x = expression(oq.Parameter("x"))
    program = {
        "div": lambda: 1 / x,
        "sqrt": x.sqrt,
        "log": x.log,
        "pow": lambda: x**0.5,
        "exp": x.exp,
    }[operation]()
    program = program + program
    bad = {"div": 0.0, "sqrt": -1.0, "log": -1.0, "pow": -1.0, "exp": 1000.0}[operation]
    for evaluate in (lambda: program.evaluate({"x": bad}), lambda: program.bind({"x": bad})):
        with np.errstate(all="ignore"), pytest.raises(oq.ValidationError):
            evaluate()


@pytest.mark.parametrize(
    "operation", ["sin", "cos", "tan", "exp", "log", "sqrt", "add", "mul", "div", "pow"]
)
def test_all_chain_rules_match_independent_finite_differences(operation):
    x, w = expression(oq.Parameter("x")), expression(oq.Parameter("w"))
    a, b = x.sin() + 1.4, w.cos() + 1.2
    program = (
        {"add": lambda: a + b, "mul": lambda: a * b, "div": lambda: a / b, "pow": lambda: a**b}[
            operation
        ]()
        if operation in {"add", "mul", "div", "pow"}
        else getattr(a, operation)()
    )
    row = {"x": 0.3, "w": -0.4}
    for name in row:
        delta = 1e-6
        finite = (
            program.evaluate(row | {name: row[name] + delta})
            - program.evaluate(row | {name: row[name] - delta})
        ) / (2 * delta)
        np.testing.assert_allclose(
            program.derivative(name).evaluate(row), finite, rtol=2e-8, atol=2e-8
        )


def test_numeric_values_are_never_reused_between_calls_and_pickle_remains_compatible():
    program = shared(expression(oq.Parameter("x")), 8)
    restored = pickle.loads(pickle.dumps(program))
    assert program == restored and hash(program) == hash(restored)
    # Simulate an older pickle that predates derived constant metadata.
    for node in [restored, restored.args[0]]:
        node.__dict__.pop("_parameter_free", None)
    row = {"x": np.array([0.1, 0.2])}
    for x in (0.1, 0.3):
        row["x"][:] = x
        np.testing.assert_allclose(restored.evaluate(row), reference(x, 8)[0])
    np.testing.assert_allclose(restored.bind({"x": 0.3}).constant, reference(0.3, 8)[0])
    constant_exponent = expression(2.0)
    constant_exponent.__dict__.pop("_parameter_free", None)
    squared = oq.Expression(op="pow", args=(expression(oq.Parameter("x")), constant_exponent))
    assert squared.derivative("x").evaluate({"x": -0.4}) == pytest.approx(-0.8)


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_native_hybrid_training_keeps_first_second_and_mapped_derivatives(framework):
    lib = pytest.importorskip(framework)
    x, w = oq.Parameter("x"), oq.Parameter("w")
    circuit = oq.Circuit(1).ry(shared(x * w + 0.1, 8), 0)
    value, first, second = reference(0.3 * 0.4 + 0.1, 8, first=0.3)
    expected_first = -np.sin(value) * first
    expected_second = -np.cos(value) * first**2 - np.sin(value) * second
    if framework == "torch":
        layer = oq.torch.QuantumLayer(circuit, ["x"], ["w"], dtype=lib.float64)
        weight = lib.tensor([0.4], dtype=lib.float64, requires_grad=True)
        inputs = lib.tensor([[0.3]], dtype=lib.float64)
        output = layer.functional(weight, inputs).sum()
        derivative = lib.autograd.grad(output, weight, create_graph=True)[0]
        curvature = lib.autograd.grad(derivative.sum(), weight)[0]
        directions = lib.linspace(0.1, 0.5, 8, dtype=lib.float64)[:, None]
        mapped = lib.func.vmap(
            lambda d: lib.func.jvp(lambda v: layer.functional(v, inputs), (weight,), (d,))[1]
        )(directions)
        actual = [float(output.detach()), float(derivative.detach()[0]), float(curvature[0])]
        np.testing.assert_allclose(
            mapped.detach().numpy()[:, 0, 0], expected_first * directions.numpy()[:, 0], atol=1e-10
        )
        tolerance = 1e-9
    else:
        import jax.numpy as jnp

        layer = oq.jax.QuantumLayer(circuit, ["x"], ["w"])
        weight, inputs = jnp.array([0.4]), jnp.array([[0.3]])

        def loss(v):
            return layer(v, inputs).sum()

        output, derivative = lib.jit(lib.value_and_grad(loss))(weight)
        curvature = lib.jit(lib.jacfwd(lib.grad(loss)))(weight)
        directions = jnp.linspace(0.1, 0.5, 8)[:, None]
        mapped = lib.jit(lib.vmap(lambda d: lib.jvp(loss, (weight,), (d,))[1]))(directions)
        actual = [float(output), float(derivative[0]), float(curvature[0, 0])]
        np.testing.assert_allclose(
            np.asarray(mapped), expected_first * np.asarray(directions)[:, 0], rtol=2e-4, atol=2e-5
        )
        tolerance = 2e-3
    np.testing.assert_allclose(
        actual, [np.cos(value), expected_first, expected_second], rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("target", ["qiskit", "cirq", "openqasm3"])
def test_bound_conversion_preserves_nonlinear_angles_phase_and_source_ownership(target):
    if target != "openqasm3":
        pytest.importorskip(target)
    x, w = oq.Parameter("input"), oq.Parameter("weight")
    angle = shared(x * w + 0.1, 20)
    source = oq.LayerSpec(
        oq.Circuit(2).ry(angle, 0).cx(0, 1).with_global_phase(angle / 3), ["input"], ["weight"]
    )
    row = {"input": 0.3, "weight": 0.4}
    exported = oq.convert(source.circuit, target=target, bindings=row).value
    restored = oq.convert(exported, source=target, target="ir").value
    np.testing.assert_allclose(
        oq.statevector(restored), oq.statevector(source.circuit, row), atol=1e-11
    )
    assert source.circuit.parameters == ("input", "weight")
    assert source.input_parameters == ("input",) and source.weight_parameters == ("weight",)
