"""Semantic, numerical and memory regressions found during the package audit."""

import gc
import io
import json
import sys
import zipfile
from fractions import Fraction

import numpy as np
import pytest

import openqml as oq


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    if "jax" in sys.modules:
        sys.modules["jax"].clear_caches()
    gc.collect()


@pytest.mark.parametrize("operation", ["mul", "div", "pow"])
@pytest.mark.parametrize("scope", ["gate", "phase", "feedback"])
def test_qasm_affine_leaves_preserve_grouping_in_nonlinear_expressions(operation, scope):
    alpha, beta = oq.Parameter("alpha"), oq.Parameter("beta")
    left, right = alpha + 1, beta + 2
    value = {"mul": lambda: left * right, "div": lambda: left / right, "pow": lambda: left**right}[
        operation
    ]()
    body = oq.Circuit(1).ry(value, 0)
    circuit = (
        body
        if scope == "gate"
        else oq.Circuit(1).h(0).with_global_phase(value)
        if scope == "phase"
        else oq.Circuit(1).h(0).measure(0, 0).if_else(oq.ClassicalCondition((0,), 1), body)
    )
    artifact = oq.plan_conversion(circuit, target="openqasm3").export().value
    restored = oq.convert(
        artifact, source="openqasm3", target="ir", source_options={"parser": "builtin"}
    ).circuit
    evaluator = oq.density_matrix if scope == "feedback" else oq.statevector
    for a, b in [(0.2, 0.3), (-0.4, 0.1)]:
        bindings = {"alpha": a, "beta": b}
        np.testing.assert_allclose(
            evaluator(restored, bindings), evaluator(circuit, bindings), atol=2e-13, rtol=0
        )


def test_shared_expression_expansion_is_admitted_before_interchange_serialization():
    value = oq.Parameter("alpha").sin()
    for _ in range(25):
        value = value * value
    circuit = oq.Circuit(1).ry(value, 0)
    for target in ("openqasm3", "json"):
        plan = oq.plan_conversion(circuit, target=target)
        assert not plan.compatible
        assert any(d.code == "EXPRESSION_LIMIT" for d in plan.diagnostics)
        with pytest.raises(oq.ConversionError, match="EXPRESSION_LIMIT"):
            plan.export()
    with pytest.raises(oq.ValidationError, match="expression"):
        value.to_dict()
    payload = {"constant": 0.0, "terms": {"alpha": 1.0}}
    for _ in range(100):
        payload = {"op": "sin", "args": [payload]}
    with pytest.raises(oq.ValidationError, match="expression"):
        oq.Expression.from_dict(payload)


def test_qasm_negative_power_base_keeps_its_parentheses():
    value = (-2) ** oq.Parameter("beta")
    text = oq.convert(oq.Circuit(1).ry(value, 0), target="openqasm3").value
    restored = oq.convert(
        text, source="openqasm3", target="ir", source_options={"parser": "builtin"}
    ).circuit
    for exponent in (2, 3, 4):
        assert restored.operations[0].params[0].evaluate({"beta": exponent}) == (-2) ** exponent


def test_sparse_guard_export_never_materializes_the_classical_register(monkeypatch):
    from openqml.adapters import openqasm

    def bounded_range(*args):
        result = range(*args)
        assert len(result) < 1000, "Sparse selectors must not allocate the whole register"
        return result

    monkeypatch.setattr(openqasm, "range", bounded_range, raising=False)
    circuit = oq.Circuit(1, num_clbits=10**9).if_else(
        oq.ClassicalCondition((2, 0), 1), oq.Circuit(1).x(0)
    )
    text = openqasm.to_openqasm(circuit)
    assert len(text) < 300 and "c[{2, 0}] == 1" in text


@pytest.mark.parametrize("label", [False, True])
def test_native_boolean_switch_labels_lower_to_integer_guards(label):
    qiskit = pytest.importorskip("qiskit")
    circuit = qiskit.QuantumCircuit(1, 1)
    with circuit.switch(circuit.clbits[0]) as case:
        with case(label):
            circuit.x(0)
    plan = oq.plan_conversion(circuit, source="qiskit", target="qiskit")
    assert plan.circuit.operations[0].condition.value == int(label)
    assert plan.verify().matched


def test_qasm_output_byte_budget_survives_binding_sweeps_and_cli(tmp_path, capsys):
    from openqml.adapters.openqasm import to_openqasm
    from openqml.cli import main

    circuit = oq.Circuit(2).ry(oq.Parameter("alpha"), 0).cx(0, 1)
    text = to_openqasm(circuit)
    size = len(text.encode("utf-8"))
    plan = oq.plan_conversion(circuit, target="openqasm3", max_output_bytes=size)
    assert plan.compatible and plan.export().value == text
    bound = plan.bind({"alpha": 0.2})
    assert bound.max_output_bytes == size
    assert len(plan.export_many([{"alpha": 0.2}])) == 1
    limited = oq.plan_conversion(circuit, target="openqasm3", max_output_bytes=size - 1)
    assert not limited.compatible
    with pytest.raises(oq.ConversionError, match="QASM_EXPORT_LIMIT"):
        limited.export()
    path = tmp_path / "circuit.json"
    path.write_text(circuit.to_json())
    assert (
        main(
            [
                "convert",
                str(path),
                "--from",
                "json",
                "--to",
                "openqasm3",
                "--max-output-bytes",
                "10",
                "--check",
            ]
        )
        == 2
    )
    report = json.loads(capsys.readouterr().out)
    assert report["limits"]["max_output_bytes"] == 10 and not report["compatible"]


def test_json_byte_preflight_precedes_expanded_tree_materialization(monkeypatch):
    circuit = oq.Circuit(1).h(0).ry(oq.Parameter("alpha") + 0.2, 0)
    text = circuit.to_json()
    size = len(text.encode("utf-8"))
    assert oq.convert(circuit, target="json", max_output_bytes=size).value == text
    plan = oq.plan_conversion(circuit, target="json", max_output_bytes=size - 1)
    assert not plan.compatible
    assert any(d.code == "JSON_EXPORT_LIMIT" for d in plan.diagnostics)
    value = oq.Parameter("alpha").sin()
    for _ in range(8):
        value = value * value
    repeated = oq.Circuit(1).extend([oq.Operation("ry", (0,), (value,))] * 100)

    def forbidden(*args, **kwargs):
        pytest.fail("Reject oversized expanded IR before constructing dictionaries")

    monkeypatch.setattr(oq.Circuit, "to_dict", forbidden)
    with pytest.raises(oq.ValidationError, match="max_bytes"):
        repeated.to_json(max_bytes=1024)


def _tail_oracle(losses, confidence, probabilities=None):
    losses = [Fraction(float(v)) for v in losses]
    mass = (
        [Fraction(1) for _ in losses]
        if probabilities is None
        else [Fraction(float(v)) for v in probabilities]
    )
    total = sum(mass)
    tail = remaining = 1 - Fraction(confidence)
    result = Fraction(0)
    for loss, weight in sorted(zip(losses, mass, strict=True), reverse=True):
        admitted = min(weight / total, remaining)
        result += admitted * loss
        remaining -= admitted
        if remaining <= 0:
            break
    return float(result / tail)


def _native(value, backend):
    array = np.asarray(value, dtype=np.float32)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        return torch.tensor(array, dtype=torch.float32)
    if backend == "jax":
        jnp = pytest.importorskip("jax.numpy")
        return jnp.asarray(array)
    return array


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("case", ["mean", "weighted", "tiny_tail", "fractional"])
def test_cvar_matches_exact_mass_arithmetic_at_cancellation_boundaries(backend, case):
    from openqml.finance import expected_shortfall

    losses, confidence, probabilities = {
        "mean": ([-1e20, 1, 2], 0, [0, 0.5, 0.5]),
        "weighted": ([-1e20, 1, 2], 0.95, [0.95, 0.025, 0.025]),
        "tiny_tail": ([0, 5], 1 - 1e-11, [1, 1e-10]),
        "fractional": ([-1e11, -1e8, 1], 1 - 1.000001 / 3, None),
    }[case]
    losses = np.asarray(losses, dtype=np.float32)
    probabilities = None if probabilities is None else np.asarray(probabilities, dtype=np.float32)
    reference = _tail_oracle(losses, confidence, probabilities)
    actual = expected_shortfall(
        _native(losses, backend),
        confidence=confidence,
        probabilities=None if probabilities is None else _native(probabilities, backend),
    )
    np.testing.assert_allclose(actual, reference, rtol=3e-6, atol=5e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_probability_normalization_handles_large_finite_float32_weights(backend):
    from openqml.finance import expected_shortfall

    actual = expected_shortfall(
        _native([1, 2, 3], backend),
        confidence=0,
        probabilities=_native([1e38, 1e38, 1e38], backend),
    )
    np.testing.assert_allclose(actual, 2, atol=5e-6)


@pytest.mark.parametrize("issue", ["radius", "budget", "probabilities", "scenarios"])
def test_jitted_invalid_risk_budget_inputs_propagate_to_portfolio_weights(issue):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from openqml.finance import risk_budget_projection

    def project(scenarios, probabilities, radius, budget):
        return risk_budget_projection(
            jnp.array([0.4, 0.5]),
            scenarios,
            probabilities=probabilities,
            radius=radius,
            budget=budget,
        ).weights

    scenarios = jnp.array([[-0.1, -0.05], [0.05, 0.1]])
    probabilities, radius, budget = jnp.array([0.5, 0.5]), 0.001, 0.02
    if issue == "radius":
        radius = -0.001
    elif issue == "budget":
        budget = -0.01
    elif issue == "probabilities":
        probabilities = jnp.array([-0.5, 1.5])
    else:
        scenarios = scenarios.at[0, 0].set(jnp.nan)
    result = jax.jit(project)(scenarios, probabilities, radius, budget)
    assert np.isnan(result).all()


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("rank", [3, 6])
def test_zero_mass_large_factors_keep_low_rank_solves_finite(backend, rank):
    from openqml.finance._array import Ops
    from openqml.finance._linear import linear_solve

    rng = np.random.default_rng(19)
    factors = (rng.normal(size=(rank, 16)) * 0.1).astype(np.float32)
    factors[0] = 1e20
    mass = np.ones(rank, dtype=np.float32)
    mass[0] = 0
    rhs = rng.normal(size=(16, 2)).astype(np.float32)
    active = factors[1:].astype(float)
    reference = np.linalg.solve(np.eye(16) + active.T @ active, rhs)
    diagonal, factors, mass, rhs = (_native(v, backend) for v in (np.ones(16), factors, mass, rhs))
    actual = linear_solve(diagonal, factors, mass, rhs, 100, 1e-6, Ops(diagonal))
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_mixed_numpy_and_native_bindings_keep_torch_dtype_and_gradients(chunk_size):
    torch = pytest.importorskip("torch")
    alpha, weight = oq.Parameter("alpha"), oq.Parameter("weight")
    plan = oq.TensorNetworkPlan(oq.Circuit(1).ry(alpha + weight, 0), (oq.PauliString("Z"),))
    source = np.array([0.1, 0.2, 0.3])[::-1]
    source.flags.writeable = False
    weight = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    result = plan.evaluate(
        {"alpha": source, "weight": weight}, backend="torch", batch_chunk_size=chunk_size
    )
    assert result.dtype == torch.float64
    np.testing.assert_allclose(result.detach()[:, 0], np.cos(source + 0.4), atol=2e-14, rtol=0)
    result.sum().backward()
    np.testing.assert_allclose(weight.grad, -np.sin(source + 0.4).sum(), atol=2e-13, rtol=0)


def test_native_streaming_converts_only_the_current_host_chunk(monkeypatch):
    torch = pytest.importorskip("torch")
    from openqml.simulator import _TensorOps

    source = np.linspace(-0.3, 0.4, 100001)[::-1]
    alpha, weight = oq.Parameter("alpha"), oq.Parameter("weight")
    plan = oq.TensorNetworkPlan(oq.Circuit(1).ry(alpha + weight, 0), (oq.PauliString("Z"),))
    original = _TensorOps.array

    def checked(self, value, **options):
        if isinstance(value, np.ndarray):
            assert value.size <= 7, "Only the current host chunk may be transferred"
        return original(self, value, **options)

    monkeypatch.setattr(_TensorOps, "array", checked)
    stream = plan.iter_evaluate(
        {"alpha": source, "weight": torch.tensor(0.2, dtype=torch.float64)},
        backend="torch",
        batch_chunk_size=7,
    )
    actual = torch.cat((next(stream), next(stream))).detach()[:, 0]
    np.testing.assert_allclose(actual, np.cos(source[:14] + 0.2), atol=2e-14, rtol=0)


def test_numpy_streaming_checks_finiteness_per_chunk_without_a_full_grid_mask(monkeypatch):
    source = np.linspace(-0.2, 0.3, 100001)
    source[21] = np.nan
    plan = oq.TensorNetworkPlan(oq.Circuit(1).ry(oq.Parameter("alpha"), 0), (oq.PauliString("Z"),))
    original = np.isfinite

    def checked(value, *args, **kwargs):
        if isinstance(value, np.ndarray):
            assert value.size <= 7, "Streaming validation must stay within the current chunk"
        return original(value, *args, **kwargs)

    monkeypatch.setattr(np, "isfinite", checked)
    stream = plan.iter_evaluate({"alpha": source}, batch_chunk_size=7)
    for _ in range(3):
        assert next(stream).shape == (7, 1)
    with pytest.raises(oq.ValidationError, match="finite"):
        next(stream)


@pytest.mark.parametrize("defect", ["shape", "truncated", "pickle"])
def test_checkpoint_tensor_headers_are_checked_before_model_construction(
    defect, tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    model = oq.Model(hidden_features=4, depth=1, backend="torch")
    model.fit([[0.1, 0.2], [0.2, -0.1]], [0.1, 0.2])
    path = model.save(tmp_path / "model.oqm")
    with zipfile.ZipFile(path) as archive:
        metadata = archive.read("model.json")
    payload = io.BytesIO()
    if defect == "pickle":
        np.savez_compressed(payload, array_0=np.array([None], dtype=object))
    else:
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header,
            {
                "descr": "<f8",
                "fortran_order": False,
                "shape": (1 << 38,) if defect == "shape" else (1024,),
            },
        )
        with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("array_0.npy", header.getvalue())
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model.json", metadata)
        archive.writestr("training", payload.getvalue())

    def forbidden(*args, **kwargs):
        pytest.fail("Malformed tensor declarations must be rejected before building the model")

    monkeypatch.setattr(oq.Model, "_build", forbidden)
    error = oq.ResourceError if defect == "shape" else oq.ValidationError
    with pytest.raises(error):
        oq.Model.load(path, backend="torch", max_bytes=1 << 20)


def test_sympy_shared_graph_import_avoids_recursive_free_symbol_scans(monkeypatch):
    sympy = pytest.importorskip("sympy")
    from openqml.adapters._common import sympy_expression, to_sympy

    value = sympy.Symbol("alpha", real=True)
    for _ in range(20):
        value = sympy.sin(sympy.Add(value, value, evaluate=False), evaluate=False)

    def forbidden(*args):
        pytest.fail("Supported symbolic nodes must use cached child symbol information")

    with monkeypatch.context() as context:
        context.setattr(sympy.Basic, "free_symbols", property(forbidden))
        result = sympy_expression(value)
    expected = 0.3
    for _ in range(20):
        expected = np.sin(2 * expected)
    np.testing.assert_allclose(result.evaluate({"alpha": 0.3}), expected, atol=2e-13, rtol=0)
    restored = sympy_expression(to_sympy(result))
    np.testing.assert_allclose(restored.evaluate({"alpha": 0.3}), expected, atol=2e-13, rtol=0)


def test_sympy_bulk_affine_import_keeps_small_coefficients_after_cancellation():
    sympy = pytest.importorskip("sympy")
    from openqml.adapters._common import sympy_expression

    alpha = sympy.Symbol("alpha", real=True)
    value = sympy.Add(10**20 * alpha, alpha, -(10**20) * alpha, alpha, evaluate=False)
    result = sympy_expression(value)
    assert result.terms == (("alpha", 2.0),)
    assert result.evaluate({"alpha": 0.3}) == 0.6


def test_architecture_depth_is_admitted_before_expanding_operator_sequences():
    class DeferredOperators:
        def __iter__(self):
            pytest.fail("Depth admission must precede operator sequence construction")

    with pytest.raises(oq.ResourceError, match="depth"):
        oq.HybridArchitecture(
            1, 1, depth=10**9, max_parameters=100, operators=DeferredOperators()
        )


@pytest.mark.parametrize(
    "field",
    [
        "input_features",
        "output_features",
        "hidden_features",
        "depth",
        "num_qubits",
        "quantum_depth",
    ],
)
def test_forecast_checkpoint_configuration_is_checked_before_native_build(
    field, tmp_path, monkeypatch
):
    from openqml.finance.models import MarketForecastPlan

    specification = MarketForecastPlan(2, 1, hidden_features=4, depth=1).architecture()
    config = {
        "input_features": 2,
        "output_features": 1,
        "task": "forecast",
        "framework": "torch",
        "hidden_features": 4,
        "depth": 1,
        "num_qubits": 2,
        "quantum_depth": 1,
        "lr": 0.001,
        "seed": 0,
    }
    config[field] = 10**9
    metadata = {
        "format": "openqml.model.v1",
        "config": config,
        "architecture": specification.to_dict(),
        "microbatch_size": 2,
        "history": [],
        "shuffle_rng": np.random.default_rng(0).bit_generator.state,
        "best_epoch": None,
        "stopped_epoch": None,
    }
    tensors = io.BytesIO()
    np.savez_compressed(tensors, array_0=np.zeros(1, dtype=np.float32))
    path = tmp_path / "forecast.oqm"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model.json", json.dumps(metadata))
        archive.writestr("training", tensors.getvalue())

    def forbidden(*args, **kwargs):
        pytest.fail("Forecast configuration must be checked before native model construction")

    monkeypatch.setattr(oq.Model, "_build", forbidden)
    with pytest.raises(oq.ValidationError, match="Forecast configuration"):
        oq.Model.load(path, max_bytes=1 << 20)
