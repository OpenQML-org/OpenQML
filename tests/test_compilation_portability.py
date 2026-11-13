"""Legacy Dynamo guards and missing-toolchain errors retain their public contracts."""

import numpy as np
import pytest

import openqml as oq


@pytest.mark.parametrize("compiler", ["eager", "aot_eager"])
def test_expression_broadcasts_capture_a_full_graph_and_keep_native_gradients(compiler):
    torch = pytest.importorskip("torch")
    x, y = oq.Parameter("x"), oq.Parameter("y")
    expression = x * y

    def evaluate(a, b):
        return expression.evaluate({"x": a, "y": b}).sin() + a / 3

    run = oq.compile_compute(evaluate, backend="torch", compiler=compiler, fullgraph=True)
    a = torch.tensor([[0.2], [-0.4]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([0.3, -0.1, 0.7], dtype=torch.float64)
    for offset in (0, 0.15):
        expected = torch.sin((a + offset) * b) + (a + offset) / 3
        actual = run(a + offset, b)
        torch.testing.assert_close(actual, expected)
        first = torch.autograd.grad(actual.sum(), a, create_graph=compiler == "eager")[0]
        torch.testing.assert_close(first, (torch.cos((a + offset) * b) * b + 1 / 3).sum(1)[:, None])
        if compiler == "eager":
            second = torch.autograd.grad(first.sum(), a)[0]
            torch.testing.assert_close(
                second, (-torch.sin((a + offset) * b) * b**2).sum(1)[:, None]
            )


@pytest.mark.parametrize("dtype", ["bool", "complex32", "complex64", "complex128"])
def test_real_expression_and_readout_guards_still_reject_invalid_native_dtypes(dtype):
    torch = pytest.importorskip("torch")
    x = oq.Parameter("x")
    value = torch.ones(2, dtype=getattr(torch, dtype))
    with pytest.raises(oq.ValidationError, match="real numeric"):
        x.sin().evaluate({"x": value})
    circuit = oq.Circuit(1).ry(x, 0)
    plan = oq.PauliPlan(circuit, [oq.PauliString("Z")])
    with pytest.raises(oq.ValidationError, match="real numeric"):
        plan.evaluate({"x": value}, backend="torch", like=torch.empty((), dtype=torch.float64))
    run = oq.compile_circuit(circuit, [oq.PauliString("Z")], backend="torch", compiler="eager")
    with pytest.raises(oq.ValidationError, match="real numeric"):
        run({"x": value})


def test_expression_broadcast_errors_and_nonfinite_results_remain_checked():
    torch = pytest.importorskip("torch")
    expression = oq.Parameter("x") * oq.Parameter("y")
    with pytest.raises(oq.ValidationError, match="broadcast"):
        expression.evaluate({"x": torch.ones(2), "y": torch.ones(3)})
    run = oq.compile_compute(
        lambda a, b: expression.evaluate({"x": a, "y": b}),
        backend="torch",
        compiler="eager",
        fullgraph=True,
    )
    with pytest.raises(oq.NumericalError, match="non-finite"):
        run(torch.tensor(float("nan")), torch.tensor(0.3))


def test_missing_cxx_compiler_is_actionable_without_changing_global_dynamo_settings():
    torch = pytest.importorskip("torch")
    from torch._inductor.exc import InvalidCxxCompiler

    original = torch._dynamo.config.suppress_errors

    def unavailable(graph, inputs):
        try:
            error = InvalidCxxCompiler("cl")
        except TypeError:  # PyTorch 2.2/2.4 infer the compiler from their configuration.
            error = InvalidCxxCompiler()
        raise error

    run = oq.compile_compute(
        lambda x: x.sin(), backend="torch", compiler=unavailable, fullgraph=True
    )
    with pytest.raises(oq.ResourceError, match="compiler='aot_eager'") as caught:
        run(torch.tensor(0.2))
    assert "Visual Studio" in str(caught.value)
    assert caught.value.__cause__ is not None
    assert torch._dynamo.config.suppress_errors is original


def test_unrelated_compiler_failures_are_not_reclassified_as_environment_errors():
    torch = pytest.importorskip("torch")

    def broken(graph, inputs):
        raise RuntimeError("deliberate backend error")

    run = oq.compile_compute(lambda x: x.cos(), backend="torch", compiler=broken, fullgraph=True)
    with pytest.raises(torch._dynamo.exc.BackendCompilerFailed, match="deliberate backend error"):
        run(torch.tensor(0.2))


def test_missing_openmp_headers_are_actionable_but_other_cpp_errors_remain_visible():
    torch = pytest.importorskip("torch")
    from torch._inductor.exc import CppCompileError

    def no_openmp(graph, inputs):
        raise CppCompileError(["clang++"], "fatal error: 'omp.h' file not found")

    run = oq.compile_compute(lambda x: x.sin(), backend="torch", compiler=no_openmp, fullgraph=True)
    with pytest.raises(oq.ResourceError, match="OMP_PREFIX"):
        run(torch.tensor(0.2))

    def bad_code(graph, inputs):
        raise CppCompileError(["clang++"], "error: unknown generated symbol")

    run = oq.compile_compute(lambda x: x.cos(), backend="torch", compiler=bad_code, fullgraph=True)
    with pytest.raises(torch._dynamo.exc.BackendCompilerFailed, match="unknown generated symbol"):
        run(torch.tensor(0.2))


def test_fullgraph_readout_accepts_fresh_values_without_retracing():
    torch = pytest.importorskip("torch")
    plan = oq.PauliPlan(oq.Circuit(1).ry(oq.Parameter("x"), 0), [oq.PauliString("Z")])
    captures = []

    def capture(graph, inputs):
        captures.append(graph)
        return graph.forward

    run = oq.compile_compute(
        lambda value: plan.evaluate({"x": value}, backend="torch", like=value),
        backend="torch",
        compiler=capture,
        fullgraph=True,
    )
    for angles in ([0.2, -0.4], [0.8, -0.1]):
        actual = run(torch.tensor(angles, dtype=torch.float64))
        np.testing.assert_allclose(actual.numpy()[:, 0], np.cos(angles), atol=2e-12)
    assert len(captures) == 1
