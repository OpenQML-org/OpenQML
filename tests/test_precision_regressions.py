"""Independent wide-range numerical and semantic regression references."""

import gc
import io
import sys
import zipfile

import numpy as np
import pytest

import openqml as oq


@pytest.fixture(autouse=True)
def release_compilations():
    yield
    if "jax" in sys.modules:
        sys.modules["jax"].clear_caches()
    gc.collect()


def native(value, backend):
    value = np.asarray(value, dtype=np.float32)
    if backend == "torch":
        return pytest.importorskip("torch").tensor(value)
    if backend == "jax":
        return pytest.importorskip("jax.numpy").asarray(value)
    return value


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("scale", [1e20, 1e-25, 0])
@pytest.mark.parametrize("axis", [-1, 0, (0, 1)])
def test_scaled_norm_matches_independent_float64_reference(backend, scale, axis):
    from openqml.finance._array import Ops

    value = native(np.array([[1, -2, 0], [3, 1, -1]]) * scale, backend)
    reference = np.sqrt(np.sum(np.asarray(value).astype(float) ** 2, axis=axis))
    result = Ops(value).norm(value, axis=axis)
    if scale:
        np.testing.assert_allclose(result / scale, reference / scale, rtol=2e-6)
    else:
        np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("scale", [1e20, 1e-25])
def test_scaled_norm_retains_native_first_and_second_derivatives(backend, scale):
    from openqml.finance._array import Ops

    x = native([scale, -scale], backend)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        gradient = torch.func.grad(lambda v: Ops(v).norm(v))(x)
        hessian = torch.func.hessian(lambda v: Ops(v).norm(v))(x)
    else:
        jax = pytest.importorskip("jax")
        gradient = jax.jit(jax.grad(lambda v: Ops(v).norm(v)))(x)
        hessian = jax.jit(jax.hessian(lambda v: Ops(v).norm(v)))(x)
    np.testing.assert_allclose(gradient, [1 / np.sqrt(2), -1 / np.sqrt(2)], rtol=3e-6)
    np.testing.assert_allclose(hessian * scale, np.full((2, 2), 1 / (2 * np.sqrt(2))), rtol=5e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("scale", [1e20, 1e-25, 0])
def test_factor_pcg_scales_each_rhs_column_without_losing_small_solutions(backend, scale):
    from openqml.finance._array import Ops
    from openqml.finance._linear import linear_solve

    rng = np.random.default_rng(37)
    factors = (rng.normal(size=(7, 10)) * 0.1).astype(np.float32)
    diagonal = np.linspace(0.8, 1.4, 10, dtype=np.float32)
    mass = np.linspace(0.1, 0.7, 7, dtype=np.float32)
    unit = rng.normal(size=(10, 2)).astype(np.float32)
    scales = np.array([scale, 1], dtype=np.float32)
    matrix = np.diag(diagonal.astype(float)) + factors.astype(float).T @ (
        mass[:, None] * factors.astype(float)
    )
    reference = np.linalg.solve(matrix, unit.astype(float))
    d, f, p, b = (native(v, backend) for v in (diagonal, factors, mass, unit * scales))
    result = np.asarray(linear_solve(d, f, p, b, 128, 1e-7, Ops(d)))
    np.testing.assert_allclose(result[:, 1], reference[:, 1], rtol=2e-5, atol=2e-6)
    if scale:
        np.testing.assert_allclose(result[:, 0] / scale, reference[:, 0], rtol=2e-5, atol=2e-6)
    else:
        np.testing.assert_array_equal(result[:, 0], np.zeros(10))


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize("scale", [1e20, 1e-25])
def test_native_metric_solve_preserves_damping_and_residual_units(backend, scale):
    from openqml._native_training import solve_metric
    from openqml.simulator import _TensorOps

    value = native([scale, -scale, scale], backend)
    result = solve_metric(
        lambda x: x,
        {"x": value},
        _TensorOps(backend, value),
        damping=1,
        tolerance=1e-30 if scale < 1 else 1e-6,
    )
    np.testing.assert_allclose(result.direction["x"] / scale, [0.5, -0.5, 0.5], rtol=2e-6)
    assert bool(result.converged) and float(result.residual_norm) / scale < 2e-6


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("scale", [1e20, 1e-25])
def test_trainers_clip_finite_gradients_instead_of_rejecting_squared_overflow(backend, scale):
    maximum = 1 if scale > 1 else 1e-26
    expected = -0.1 * min(scale, maximum / np.sqrt(2))
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from openqml.interfaces._torch_training import HybridTrainer

        model = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        trainer = HybridTrainer(
            model, torch.optim.SGD(model.parameters(), lr=0.1), max_grad_norm=maximum
        )
        metrics = trainer.step(torch.ones((2, 2)), lambda net, x: net(x).flatten() * scale)
        actual = model.weight.detach().numpy()
    else:
        jnp = pytest.importorskip("jax.numpy")
        optax = pytest.importorskip("optax")
        from openqml.interfaces._jax_training import make_train_step

        optimizer = optax.sgd(0.1)
        params = {"x": jnp.zeros(2, dtype=jnp.float32)}
        step = make_train_step(
            optimizer,
            lambda p, x: jnp.ones(x.shape[0]) * p["x"].sum() * scale,
            max_grad_norm=maximum,
        )
        params, _, metrics = step(params, optimizer.init(params), jnp.ones((2, 2)))
        actual = np.asarray(params["x"])
    assert not bool(metrics["skipped"]) and np.isfinite(float(metrics["grad_norm"]))
    np.testing.assert_allclose(actual / abs(expected), -np.ones_like(actual), rtol=3e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
@pytest.mark.parametrize(
    "values,budget,caps,expected",
    [
        ([1e8, 1e8, 0], 1, 1, [0.5, 0.5, 0]),
        ([1e38] * 4, 1, 1, [0.25] * 4),
        ([1e8, 1e8, 0], 2.5, 1, [1, 1, 0.5]),
        ([1e20, 1e20, 0], 0.8, [0.2, 0.4, 0.9], [0.2, 0.4, 0.2]),
        ([3e38, -3e38, 0], 1, 1, [1, 0, 0]),
    ],
)
def test_projection_retains_funding_under_extreme_scores(backend, values, budget, caps, expected):
    result = oq.finance.cost_aware_projection(native(values, backend), caps=caps, budget=budget)
    np.testing.assert_allclose(result, expected, atol=3e-6)
    np.testing.assert_allclose(np.asarray(result).sum(), budget, atol=3e-6)


def test_cost_aware_projection_matches_independent_multiplier_bisection():
    rng = np.random.default_rng(63)
    for _ in range(100):
        values = rng.normal(size=7) * 2
        caps = rng.uniform(0.1, 1, size=7)
        center = rng.uniform(0, 1, size=7) * caps
        penalty = rng.uniform(0, 0.3, size=7)
        budget = rng.uniform(0.01, caps.sum() - 0.01)
        low, high = -100.0, 100.0
        for _ in range(100):
            middle = (low + high) / 2
            delta = values - middle - center
            weights = np.clip(
                center + np.sign(delta) * np.maximum(abs(delta) - penalty, 0), 0, caps
            )
            if weights.sum() > budget:
                low = middle
            else:
                high = middle
        actual = oq.finance.cost_aware_projection(
            values, previous=center, penalty=penalty, caps=caps, budget=budget
        )
        np.testing.assert_allclose(actual, weights, atol=1e-12, rtol=0)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_projection_removes_budget_constant_linear_fees(backend):
    values = native([0.1, -0.2, 0.3], backend)
    actual = oq.finance.cost_aware_projection(values, penalty=1e20)
    np.testing.assert_allclose(actual, [11 / 30, 1 / 15, 17 / 30], atol=2e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_projection_rejects_unresolved_funding_instead_of_returning_an_invalid_portfolio(backend):
    values = native([0.1, -0.2, 0.3], backend)
    with pytest.raises(oq.ResourceError, match="funding tolerance"):
        oq.finance.cost_aware_projection(values, previous=[0.1, 0.2, 0.3], penalty=1e20)


def test_compiled_projection_marks_unresolved_financing_with_nan():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    project = jax.jit(
        lambda values: oq.finance.cost_aware_projection(
            values, previous=[0.1, 0.2, 0.3], penalty=1e20
        )
    )
    assert np.isnan(project(jnp.array([0.1, -0.2, 0.3]))).all()


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_projection_preserves_a_small_representable_budget(backend):
    result = oq.finance.cost_aware_projection(native([0, 0, 0], backend), budget=1e-9)
    np.testing.assert_allclose(result / 1e-9, np.full(3, 1 / 3), rtol=2e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_projection_refines_roundoff_on_the_selected_trading_face(backend):
    result = oq.finance.cost_aware_projection(
        native([0.1, -0.2, 0.3], backend), previous=[0.2, 0.3, 0.5], penalty=100
    )
    np.testing.assert_allclose(result, [0.2, 0.3, 0.5], atol=2e-7, rtol=0)


@pytest.mark.parametrize("target", ["cirq", "qiskit", "pennylane", "openqasm3"])
@pytest.mark.parametrize("location", ["gate", "phase"])
def test_raw_constant_expression_nodes_preserve_sdk_semantics(target, location):
    if not target.startswith("openqasm"):
        pytest.importorskip(target)
    raw = oq.Expression(op="sin", args=(oq.Expression(constant=0.7),))
    circuit = (
        oq.Circuit(1).ry(raw, 0)
        if location == "gate"
        else oq.Circuit(1).h(0).with_global_phase(raw)
    )
    plan = oq.plan_conversion(circuit, target=target)
    assert plan.verify().matched
    assert raw.op == "sin"


@pytest.mark.parametrize("target", ["cirq", "qiskit", "openqasm3"])
def test_symbolic_parent_reuses_folded_constant_children(target):
    if target != "openqasm3":
        pytest.importorskip(target)
    raw = oq.Expression(op="sin", args=(oq.Expression(constant=0.7),))
    value = oq.Parameter("alpha") + raw
    plan = oq.plan_conversion(oq.Circuit(1).ry(value, 0), target=target)
    assert plan.verify({"alpha": 0.2}).matched
    np.testing.assert_allclose(
        plan.circuit.operations[0].params[0].evaluate({"alpha": 0.2}), 0.2 + np.sin(0.7)
    )


def test_qasm2_phase_diagnostics_evaluate_raw_constants():
    for value in (0, 0.7):
        phase = oq.Expression(op="sin", args=(oq.Expression(constant=value),))
        codes = {
            item.code for item in oq.diagnose(oq.Circuit(1).with_global_phase(phase), "openqasm2")
        }
        assert ("GLOBAL_PHASE_UNSUPPORTED" in codes) == bool(value)


@pytest.mark.parametrize("scale", [1e200, 1e-200, 0])
def test_tensor_train_svd_bounds_remain_finite_at_wide_scales(scale):
    matrix = np.diag([1.0, 2, 3, 4]) + np.random.default_rng(9).normal(size=(4, 4)) * 0.01
    compressed = oq.TensorTrainMatrix.from_dense(
        matrix * scale, input_modes=(2, 2), output_modes=(2, 2), relative_error=0.02
    )
    restored = compressed.to_dense()
    if scale:
        error = np.linalg.norm(restored / scale - matrix)
        assert error <= compressed.error_bound / scale + 1e-12
        assert error <= 0.02 * np.linalg.norm(matrix)
        np.testing.assert_allclose(
            compressed.source_norm / scale, np.linalg.norm(matrix), rtol=1e-14
        )
    else:
        np.testing.assert_array_equal(restored, np.zeros((4, 4)))


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_channel_apply_accepts_array_like_density_with_resource_admission(backend):
    plan = oq.QuantumChannelPlan()
    weights = native(np.zeros(len(plan.spec.weight_parameters)), backend)
    result = plan.apply(weights, native([0.1], backend), [[1, 0], [0, 0]], backend=backend)
    reference = plan.apply(weights, native([0.1], backend), np.diag([1, 0]), backend=backend)
    np.testing.assert_allclose(result, reference, atol=2e-6)
    with pytest.raises(oq.ValidationError, match="Density shape"):
        plan.apply(weights, native([0.1], backend), [[1]], backend=backend)


@pytest.mark.parametrize("defect", ["shape", "size", "corrupt"])
def test_native_jax_checkpoint_admission_precedes_tensor_loading(defect, tmp_path, monkeypatch):
    pytest.importorskip("jax")
    from openqml.interfaces._network_checkpoint import load_network

    path = tmp_path / "training.npz"
    if defect == "corrupt":
        path.write_bytes(b"corrupt")
    elif defect == "size":
        np.savez_compressed(path, leaf_0=np.zeros(1024))
    else:
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            header, {"descr": "<f8", "fortran_order": False, "shape": (1 << 38,)}
        )
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("leaf_0.npy", header.getvalue())

    def forbidden(*args, **kwargs):
        pytest.fail("Checkpoint admission must precede NumPy tensor loading")

    monkeypatch.setattr(np, "load", forbidden)
    error = oq.ValidationError if defect == "corrupt" else oq.ResourceError
    with pytest.raises(error):
        load_network(object(), path, max_bytes=16 if defect == "size" else 1 << 20)


@pytest.mark.parametrize("rank", [3, 6])
def test_compiled_factor_solves_retain_implicit_backward_and_eager_jvp(rank):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "while_loop"):
        pytest.skip("Compiled implicit boundaries require structured Torch control flow.")
    from openqml.finance._array import Ops
    from openqml.finance._linear import linear_solve

    generator = torch.Generator().manual_seed(31)
    diagonal = torch.ones(16, requires_grad=True)
    factors = (torch.randn(rank, 16, generator=generator) * 0.1).requires_grad_()
    mass = torch.ones(rank, requires_grad=True)
    rhs = torch.randn(16, 2, generator=generator, requires_grad=True)

    def solve(d, f, p, b):
        return linear_solve(d, f, p, b, 64, 1e-7, Ops(b))

    compiled = torch.compile(solve, backend="aot_eager", fullgraph=True)
    inputs = (diagonal, factors, mass, rhs)
    result, reference = compiled(*inputs), solve(*inputs)
    torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-5)
    received = torch.autograd.grad(result.sum(), inputs)
    expected = torch.autograd.grad(reference.sum(), inputs)
    for actual, wanted in zip(received, expected, strict=True):
        torch.testing.assert_close(actual, wanted, atol=2e-5, rtol=2e-5)
    _, tangent = torch.func.jvp(
        lambda b: solve(diagonal.detach(), factors.detach(), mass.detach(), b),
        (rhs.detach(),),
        (torch.ones_like(rhs),),
    )
    torch.testing.assert_close(
        tangent,
        solve(diagonal.detach(), factors.detach(), mass.detach(), torch.ones_like(rhs)),
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize("operation", ["allocation", "projection"])
def test_compiled_portfolio_layers_accept_scalar_configuration_and_retain_gradients(operation):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "while_loop"):
        pytest.skip("Compiled implicit boundaries require structured Torch control flow.")
    from openqml.finance import AllocationPlan, FactorRisk, cost_aware_projection

    plan = AllocationPlan(max_iterations=8, cg_iterations=8)
    risk = FactorRisk(torch.zeros(4), torch.empty(0, 4), torch.empty(0), torch.full((4,), 0.1))

    def evaluate(mean):
        return (
            plan.solve(mean, risk).weights
            if operation == "allocation"
            else cost_aware_projection(mean)
        )

    mean = torch.tensor([0.01, 0.02, 0.03, 0.04], requires_grad=True)
    result = torch.compile(evaluate, backend="aot_eager", fullgraph=True)(mean)
    reference = evaluate(mean)
    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(
        torch.autograd.grad(result.square().sum(), mean)[0],
        torch.autograd.grad(reference.square().sum(), mean)[0],
        atol=3e-6,
        rtol=3e-6,
    )
