"""Construction, exception, memory and benchmark isolation contracts."""

import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import openqml as oq


def benchmark_module(name):
    path = Path(__file__).parents[1] / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"audit_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "kind",
    [
        "ValidationError",
        "ResourceError",
        "MissingDependencyError",
        "BackendError",
        "ConversionError",
    ],
)
def test_public_errors_have_one_unambiguous_root_and_pickle_roundtrip(kind):
    cls = getattr(oq, kind)
    assert cls.__bases__ == (oq.OpenQMLError,)
    error = (
        cls([oq.Diagnostic("TEST", "Structured diagnostic", "operation[2]")])
        if kind == "ConversionError"
        else cls("test")
    )
    assert not isinstance(error, (ValueError, MemoryError, ImportError, RuntimeError))
    with pytest.raises(oq.OpenQMLError):
        raise error
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is cls and str(restored) == str(error)
    if kind == "ConversionError":
        assert restored.diagnostics == error.diagnostics


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_native_numeric_failures_keep_conversion_diagnostics(value):
    from openqml.adapters._common import numeric

    with pytest.raises(oq.ConversionError) as caught:
        numeric(value, "operations[3].params[0]")
    diagnostic = caught.value.diagnostics[0]
    assert diagnostic.code == "NONREAL_PARAMETER"
    assert diagnostic.location == "operations[3].params[0]"


def test_builder_is_linear_and_freezes_independent_snapshots(monkeypatch):
    import openqml.ir as ir

    original, checked = ir._validate_operation, []

    def counted(*args):
        checked.append(None)
        return original(*args)

    monkeypatch.setattr(ir, "_validate_operation", counted)
    builder = oq.CircuitBuilder(3)
    for index in range(2000):
        assert builder.ry(0.1, index % 3) is builder
    circuit = builder.build()
    assert len(checked) == 4000
    assert len(circuit.operations) == 2000
    builder.x(0)
    assert len(circuit.operations) == 2000 and len(builder.build().operations) == 2001
    assert isinstance(circuit.operations, tuple)
    with pytest.raises(AttributeError):
        circuit.num_qubits = 8


def test_builder_and_immutable_circuit_preserve_effects_phase_and_matrix_payloads(monkeypatch):
    x = oq.Parameter("x")
    unitary = oq.Operation("unitary", (1,), matrix=np.diag([1, 1j]))
    branch = oq.Circuit(2).ry(x, 1).with_global_phase(x / 3)
    builder = oq.CircuitBuilder(2).h(0).append(unitary).ry(x, 1).measure(0, 3)
    builder.if_else(oq.ClassicalCondition((3,), 1), branch).with_global_phase(x / 7)
    built = builder.build()
    expected = (
        oq.Circuit(2)
        .h(0)
        .append(unitary)
        .ry(x, 1)
        .measure(0, 3)
        .if_else(oq.ClassicalCondition((3,), 1), branch)
        .with_global_phase(x / 7)
    )
    assert built == expected and built.operations[1] is unitary
    clone = oq.CircuitBuilder.from_circuit(built)
    assert clone.build() == built
    clone.reset(0).bit_flip(0.2, 1)
    np.testing.assert_allclose(
        oq.density_matrix(built, {"x": 0.4}), oq.density_matrix(expected, {"x": 0.4}), atol=1e-12
    )

    def rejected(_):
        pytest.fail("Revalidated an unchanged immutable matrix")

    monkeypatch.setattr(oq.Operation, "__post_init__", rejected)
    assert oq.CircuitBuilder.from_circuit(built).build().operations[1] is unitary


@pytest.mark.parametrize("invalid", [None, ["bad"], [oq.Operation("x", (3,))]])
def test_builder_bulk_edits_fail_before_mutating(invalid):
    builder = oq.CircuitBuilder(2).h(0)
    before = builder.build()
    with pytest.raises(oq.ValidationError):
        builder.extend(invalid)
    assert builder.build() == before
    with pytest.raises(oq.ValidationError):
        builder.if_else(oq.ClassicalCondition((7,), 1), oq.Circuit(3))
    assert builder.build() == before and builder.num_clbits == 0


def test_bulk_extend_and_measure_all_match_immutable_semantics():
    operations = [oq.Operation("x", (0,)), oq.Operation("h", (1,))]
    source = oq.Circuit(2).rz(0.2, 1)
    expanded = source.extend(iter(operations)).measure_all()
    built = oq.CircuitBuilder.from_circuit(source).extend(iter(operations)).measure_all().build()
    assert built == expanded and len(source.operations) == 1
    assert built.num_clbits == 2 and built.operations[-1].clbits == (1,)
    assert oq.Circuit.from_json(built.to_json()) == built


@pytest.mark.parametrize("factory", [oq.data_reuploading, oq.hardware_efficient])
def test_template_construction_uses_linear_builder(factory, monkeypatch):
    def reject(*args):
        pytest.fail("Template used repeated immutable append")

    monkeypatch.setattr(oq.Circuit, "append", reject)
    spec = factory(4, depth=20)
    assert isinstance(spec.circuit, oq.Circuit)
    assert spec.input_parameters == tuple(f"x_{i}" for i in range(4))
    assert len(spec.weight_parameters) >= 160
    assert np.isfinite(
        oq.expectations(
            spec.circuit, spec.observables, {name: 0.1 for name in spec.circuit.parameters}
        )
    ).all()


@pytest.mark.parametrize(
    "qubits,batch", [((11,), (16,)), ((12, 4), (8,)), ((15, 2, 7), ()), ((9, 3), (2, 32))]
)
def test_large_numpy_contraction_matches_independent_transpose_reference(qubits, batch):
    from openqml.simulator import _apply_matrix, _TensorOps

    width = max(qubits) + 1
    rng = np.random.default_rng(91)
    state = rng.normal(size=batch + (2**width,)) + 1j * rng.normal(size=batch + (2**width,))
    dim = 2 ** len(qubits)
    matrix = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    original = state.copy()
    rank = len(batch)
    order = qubits + tuple(q for q in range(width) if q not in qubits)
    axes = tuple(range(rank)) + tuple(rank + q for q in order)
    inverse = tuple(range(rank)) + tuple(rank + order.index(q) for q in range(width))
    reference = (
        (
            matrix
            @ state.reshape(batch + (2,) * width)
            .transpose(axes)
            .reshape(batch + (dim, 2 ** (width - len(qubits))))
        )
        .reshape(batch + (2,) * width)
        .transpose(inverse)
        .reshape(state.shape)
    )
    result = _apply_matrix(state, matrix, qubits, width, _TensorOps())
    np.testing.assert_allclose(result, reference, atol=2e-12, rtol=2e-12)
    np.testing.assert_array_equal(state, original)
    assert not np.shares_memory(state, result)


def test_large_numpy_contraction_uses_bounded_tiles_and_preserves_batched_matrices(
    monkeypatch,
):
    import openqml.simulator as sim

    rng = np.random.default_rng(1)
    state = rng.normal(size=(2, 3, 2**14)).astype(complex)
    matrix = rng.normal(size=(2, 1, 4, 4)).astype(complex)
    original, calls = np.matmul, []

    def checked(matrix, block, **kwargs):
        calls.append(block.nbytes)
        assert "out" in kwargs
        return original(matrix, block, **kwargs)

    monkeypatch.setattr(np, "matmul", checked)
    result = sim._apply_matrix(state, matrix, (13, 3), 14, sim._TensorOps())
    assert len(calls) > 1 and max(calls) <= 1 << 20
    for i in range(2):
        for j in range(3):
            expected = sim._apply_numpy(state[i, j], matrix[i, 0], (13, 3), 14)
            np.testing.assert_allclose(result[i, j], expected, atol=1e-12)


@pytest.mark.parametrize(
    "budget,batch,cap,expected",
    [(256, 1, None, 22), (1024, 1, None, 24), (1024, 8, None, 21), (1024, 1, 19, 19)],
)
def test_opt_in_statevector_policy_uses_memory_budget_without_relaxing_it(
    budget, batch, cap, expected
):
    options = oq.ExecutionOptions.for_statevector(budget, batch_size=batch, max_qubits=cap, seed=7)
    assert options.max_qubits == expected and options.max_memory_mb == budget and options.seed == 7
    assert 4 * 16 * batch * 2**expected <= budget * 2**20
    assert oq.ExecutionOptions().max_qubits == 20 and oq.ExecutionOptions().max_memory_mb == 256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_memory_mb": 0},
        {"max_memory_mb": float("inf")},
        {"max_memory_mb": 1, "batch_size": 0},
        {"max_memory_mb": 1, "max_qubits": True},
    ],
)
def test_invalid_statevector_policies_fail(kwargs):
    with pytest.raises(oq.ValidationError):
        oq.ExecutionOptions.for_statevector(**kwargs)


def test_statevector_policy_rejects_insufficient_budget():
    with pytest.raises(oq.ResourceError):
        oq.ExecutionOptions.for_statevector(1 / 2**20)


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_warm_bridge_keeps_one_preparation_and_one_submission_per_forward(framework):
    lib = pytest.importorskip(framework)

    class Backend:
        name = "audit-warm-bridge"
        capabilities = oq.BackendCapabilities(
            native_expectations=True, deterministic_expectations=True
        )

        def __init__(self):
            self.preparations, self.sizes = 0, []

        def prepare_gradient_circuit(self, circuit):
            self.preparations += 1
            return circuit

        def expectations_batch(self, circuits, observables, *, options):
            self.sizes.append(len(circuits))
            return np.asarray([[np.cos(c.operations[0].params[0].constant)] for c in circuits])

    backend = Backend()
    circuit = oq.Circuit(1).ry(oq.Parameter("x") * oq.Parameter("w"), 0)
    if framework == "torch":
        layer = oq.torch.QuantumLayer(circuit, ["x"], ["w"], backend=backend, dtype=lib.float64)
        weights = lib.tensor([0.4], dtype=lib.float64)
        for value in (0.1, 0.3, 0.5):
            inputs = lib.full((16, 1), value, dtype=lib.float64)
            actual = layer.functional(weights, inputs).detach().numpy()
            np.testing.assert_allclose(actual, np.cos(value * 0.4), atol=1e-14)
    else:
        import jax.numpy as jnp

        layer = oq.jax.QuantumLayer(circuit, ["x"], ["w"], backend=backend)
        call = lib.jit(layer)
        for value in (0.1, 0.3, 0.5):
            actual = np.asarray(call(jnp.array([0.4]), jnp.full((16, 1), value)))
            np.testing.assert_allclose(actual, np.cos(value * 0.4), atol=1e-6)
    assert backend.preparations == 1 and backend.sizes == [16, 16, 16]


@pytest.mark.parametrize("previous", [False, True])
def test_precision_worker_scope_restores_prior_state_even_on_error(previous):
    jax = pytest.importorskip("jax")
    precision = benchmark_module("_precision")

    @precision.scoped_x64
    def worker(args):
        assert jax.config.x64_enabled
        if getattr(args, "fail", False):
            raise RuntimeError("worker failed")
        return jax.config.x64_enabled

    with jax.enable_x64(previous):
        assert worker(SimpleNamespace(interface="jax")) is True
        assert jax.config.x64_enabled is previous
        with pytest.raises(RuntimeError):
            worker(SimpleNamespace(case="jax-hybrid", fail=True))
        assert jax.config.x64_enabled is previous


def test_memory_window_never_reports_a_lifetime_peak_as_a_call_peak(monkeypatch):
    memory = benchmark_module("_memory")

    def rejected(*args):
        raise PermissionError("clear_refs disabled")

    monkeypatch.setattr(memory.Path, "write_text", rejected)
    result, report = memory.measure_peak(lambda: 42)
    assert result == 42 and report["status"] == "unavailable" and report["peak_rss_bytes"] is None


def test_memory_window_resets_before_measuring_and_clearly_labels_scope():
    memory = benchmark_module("_memory")
    data, report = memory.measure_peak(lambda: bytearray(4 * 2**20))
    assert len(data) == 4 * 2**20
    if report["status"] == "unavailable":
        pytest.skip(report["reason"])
    assert report["scope"] == "current_process_call_window"
    assert report["approximate"] is True
    assert report["peak_rss_bytes"] >= report["start_rss_bytes"]


def test_fork_exec_worker_peak_is_not_attributed_to_parent_history():
    if not sys.platform.startswith("linux"):
        pytest.skip("Linux procfs accounting required")
    root = Path(__file__).parents[1] / "benchmarks"
    script = f"""
import json,sys
sys.path.insert(0,{str(root)!r})
from _memory import measure_peak
data,report=measure_peak(lambda:bytearray(4*2**20))
print(json.dumps(report))
"""
    # Run the fork in a separate, single-threaded launcher; the test process may
    # already own JAX threads and must not execute Python hooks after forking.
    launcher = f"""
import subprocess,sys
parent_allocation=bytearray(64*2**20)
result=subprocess.run([sys.executable,'-c',{script!r}],capture_output=True,text=True,check=True,preexec_fn=lambda:None)
print(result.stdout)
"""
    result = subprocess.run(
        [sys.executable, "-c", launcher],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    if report["status"] == "unavailable":
        pytest.skip(report["reason"])
    assert report["metric"] == "linux_procfs_reset_hwm"
    assert report["peak_over_start_bytes"] < 32 * 2**20
