"""Invalid public inputs fail before array, backend or optional-SDK dispatch."""

import numpy as np
import pytest

import openqml as oq


@pytest.mark.parametrize(
    "entry",
    [
        lambda value: oq.statevector(value),
        lambda value: oq.density_matrix(value),
        lambda value: oq.expectation(value, oq.PauliString("Z")),
        lambda value: oq.expectations(value, [oq.PauliString("Z")]),
        lambda value: oq.execute(value),
        lambda value: oq.decompose(value),
        lambda value: oq.ShiftPlan(value, [oq.PauliString("Z")]),
        lambda value: oq.compile_circuit(value),
        lambda value: oq.diagnose(value, "ir"),
    ],
)
def test_circuit_entry_types_are_checked(entry):
    with pytest.raises(oq.ValidationError, match="Circuit"):
        entry(object())


@pytest.mark.parametrize(
    "entry",
    [
        lambda c: oq.statevector(c, []),
        lambda c: oq.density_matrix(c, []),
        lambda c: oq.expectation(c, oq.PauliString("Z"), []),
        lambda c: oq.execute(c, []),
        lambda c: oq.plan_conversion(c, target="ir").bind([]),
        lambda c: oq.expectations_batch(c, [oq.PauliString("Z")], [None]),
    ],
)
def test_binding_containers_do_not_leak_attribute_errors(entry):
    with pytest.raises(oq.ValidationError, match="mapping"):
        entry(oq.Circuit(1))


@pytest.mark.parametrize("method", ["evaluate", "bind", "substitute"])
@pytest.mark.parametrize("bindings", [None, [], "x", 1])
@pytest.mark.parametrize("constant", [False, True])
def test_expression_methods_require_mappings_even_without_parameters(method, bindings, constant):
    from openqml.ir.parameters import expression

    value = expression(2 if constant else oq.Parameter("x").sin())
    with pytest.raises(oq.ValidationError, match="mapping"):
        getattr(value, method)(bindings)


@pytest.mark.parametrize("name", [None, [], {}, 1, "", "x\n"])
def test_expression_derivative_requires_a_parameter_name(name):
    from openqml.ir.parameters import expression

    with pytest.raises(oq.ValidationError, match="Parameter names"):
        expression(oq.Parameter("x")).derivative(name)


@pytest.mark.parametrize(
    "options",
    [
        {"op": []},
        {"op": {}},
        {"op": "sin", "args": None},
        {"op": "sin", "args": 1},
        {"args": None},
        {"terms": None},
        {"terms": [1]},
        {"terms": [("x",)]},
        {"terms": [("x", 1, 2)]},
        {"op": "sin", "args": (1,), "constant": [0, 0]},
    ],
)
def test_expression_constructor_rejects_malformed_fields_with_domain_errors(options):
    with pytest.raises(oq.ValidationError):
        oq.Expression(**options)


def test_expression_graph_node_and_deserialization_guard_unhashable_operations():
    from openqml.ir.parameters import node

    for op in ([], {}):
        with pytest.raises(oq.ValidationError, match="Expression op"):
            node(op, 1)
        with pytest.raises(oq.ValidationError):
            oq.Expression.from_dict({"op": op, "args": []})


def test_expression_guards_preserve_mapping_and_iterable_inputs():
    from types import MappingProxyType

    value = oq.Expression(terms=iter([("x", 1)]), args=[])
    bindings = MappingProxyType({"x": 0.3})
    np.testing.assert_allclose(value.sin().evaluate(bindings), np.sin(0.3))
    assert value.bind(bindings).constant == 0.3
    assert value.substitute(MappingProxyType({"x": oq.Parameter("y")})).parameters == {"y"}
    assert oq.Expression(op="sin", args=[oq.Parameter("x")], terms=[]) == value.sin()
    assert hash(oq.Expression(op="sin", args=[oq.Parameter("x")], terms=[])) == hash(value.sin())


@pytest.mark.parametrize("automatic", [True, False])
def test_automatic_readout_wraps_incompatible_broadcast_before_execution(automatic, monkeypatch):
    x, y = oq.Parameter("x"), oq.Parameter("y")
    circuit = oq.Circuit(1).ry(x, 0).rx(y, 0)
    backend = oq.get_backend("auto") if automatic else oq.PriorityBackend()

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid batch shapes must fail before backend planning or execution.")

    monkeypatch.setattr(type(backend), "plan", forbidden)
    with pytest.raises(oq.ValidationError, match="broadcast") as error:
        oq.expectations(
            circuit, [oq.PauliString("Z")], {"x": np.ones(2), "y": np.ones(3)}, backend=backend
        )
    assert isinstance(error.value.__cause__, ValueError)


def test_automatic_readout_retains_multiaxis_broadcast_semantics():
    x, y = oq.Parameter("x"), oq.Parameter("y")
    circuit = oq.Circuit(1).ry(x, 0).rx(y, 0)
    a, b = np.array([[0.2], [-0.4]]), np.array([[0.1, -0.3, 0.7]])
    values = oq.expectations(
        circuit,
        [oq.PauliString("Z")],
        {"x": a, "y": b},
        backend=oq.PriorityBackend(),
    )
    assert values.shape == (2, 3, 1)
    np.testing.assert_allclose(values[..., 0], np.cos(a) * np.cos(b), atol=2e-12)


@pytest.mark.parametrize("options", [{"task": []}, {"framework": {}}, {"task": "unknown"}])
def test_model_choices_reject_unhashable_and_unknown_values(options):
    with pytest.raises(oq.ValidationError, match="Model"):
        oq.Model(**options)


def test_conversion_names_and_sdk_export_types_are_checked_before_import(monkeypatch):
    from openqml.interop.adapters import _common, cirq, pennylane, qiskit
    from openqml.interop.frontends import get_adapter

    def forbidden(*args):
        pytest.fail("Invalid circuit imported an optional SDK")

    for module in (cirq, pennylane, qiskit):
        monkeypatch.setattr(module, "dependency", forbidden)
        with pytest.raises(oq.ValidationError, match="Circuit"):
            getattr(module, f"to_{module.__name__.split('.')[-1]}")(None)
        with pytest.raises(oq.ValidationError, match="mapping"):
            getattr(module, f"to_{module.__name__.split('.')[-1]}")(oq.Circuit(1), [])
    for name in ([], {}):
        for call in (get_adapter, oq.unregister_adapter):
            with pytest.raises(oq.ValidationError, match="str"):
                call(name)
        with pytest.raises(oq.ValidationError, match="format"):
            oq.convert(oq.Circuit(1), target=name)
    with pytest.raises(oq.ConversionError, match="Supported formats"):
        oq.convert(oq.Circuit(1), target="unknown")
    assert callable(_common.dependency)


def test_optional_import_errors_identify_a_missing_transitive_dependency(monkeypatch):
    from openqml.runtime import _dependencies

    cause = ModuleNotFoundError("No module named 'native_extension'", name="native_extension")

    def broken(name):
        raise cause

    monkeypatch.setattr(_dependencies, "import_module", broken)
    with pytest.raises(oq.MissingDependencyError, match="native_extension") as error:
        _dependencies.load_dependency("torch", "torch")
    assert error.value.__cause__ is cause
    assert "openqml[torch]" in str(error.value)


def test_financial_risk_types_and_ragged_inputs_have_domain_errors():
    f = oq.finance
    for operation in (
        lambda: f.dense_covariance(None),
        lambda: f.covariance_matvec(None, [1, 0]),
        lambda: f.portfolio_variance([1, 0], None),
        lambda: f.AllocationPlan().solve([0, 0], None),
    ):
        with pytest.raises(oq.ValidationError, match="FactorRisk"):
            operation()
    with pytest.raises(oq.ValidationError, match="real numeric"):
        f.factor_risk([[1, 2], [3]])
    with pytest.raises(oq.ValidationError, match="nonempty finite real"):
        oq.Model().fit([[1, 2], [3]], [0, 1])
    risk = f.FactorRisk([0, 0], [[1, 2]], [0.5], [0.1, 0.2])
    np.testing.assert_allclose(f.dense_covariance(risk), [[0.6, 1], [1, 2.2]])
    for vector in (1, []):
        with pytest.raises(oq.ValidationError, match="asset axis"):
            f.covariance_matvec(risk, vector)


def test_financial_broadcast_errors_use_the_shared_shape_guard():
    with pytest.raises(oq.ValidationError, match="broadcast"):
        oq.finance.cost_aware_projection([0.1, 0.2, 0.3], caps=[0.5, 0.5])
    risk = oq.finance.factor_risk(np.ones((2, 4, 3)))
    with pytest.raises(oq.ValidationError, match="broadcast"):
        oq.finance.AllocationPlan().solve(np.ones((3, 3)), risk)


def test_tensor_subclasses_keep_their_framework_and_non_tensor_modules_are_rejected():
    torch = pytest.importorskip("torch")

    class Tensor(torch.Tensor):
        pass

    returns = torch.tensor([[0.01, -0.02], [0.03, 0.01]]).as_subclass(Tensor)
    risk = oq.finance.factor_risk(returns)
    assert isinstance(risk.mean, torch.Tensor)
    with pytest.raises(oq.ValidationError, match="real numeric"):
        oq.finance.factor_risk(torch.nn.Linear(2, 2))
