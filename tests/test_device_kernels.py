"""Numerical and code-generation checks for the actual OpenQML kernel sources."""

import os

import pytest


def _runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton", minversion="3.4")
    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip("CPU kernel execution requires the Triton interpreter")
    return torch


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
@pytest.mark.parametrize("wires", [(2,), (4, 0), (2, 0, 3)])
def test_gate_kernel_matches_portable_complex_vjp(wires, precision):
    torch = _runtime()
    from openqml.sim._device_ops import _differentiable_gate
    from openqml.sim.dense import _apply_matrix_portable, _TensorOps

    dtype = getattr(torch, precision)
    dimension = 1 << len(wires)
    state = torch.randn(2, 3, 256, dtype=dtype)[..., ::2].requires_grad_()
    ops = _TensorOps("torch", state.real)
    for shape in ((), (2, 1), (1, 3), (2, 3)):
        gate = torch.randn(*shape, dimension, dimension, dtype=dtype).mH.requires_grad_()
        expected = _apply_matrix_portable(state, gate, wires, 7, ops)
        actual = _differentiable_gate().apply(state, gate, wires, 7)
        cotangent = torch.randn_like(actual)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(
            torch.autograd.grad(actual, (state, gate), cotangent),
            torch.autograd.grad(expected, (state, gate), cotangent),
            strict=True,
        ):
            torch.testing.assert_close(
                a,
                b,
                atol=2e-5 if dtype == torch.complex64 else 1e-12,
                rtol=2e-5 if dtype == torch.complex64 else 1e-12,
            )


@pytest.mark.parametrize("shared", ["", "j", "jk", "jkl"])
@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_strided_contraction_kernel_and_gradient(shared, precision):
    torch = _runtime()
    from openqml.compute._contraction_program import ContractionKernel
    from openqml.sim._device_ops import _differentiable_contraction
    from openqml.sim.dense import _TensorOps

    dtype = getattr(torch, precision)
    left_axes, right_axes = "ac" + shared[::-1] + "bf", "d" + shared + "ze"
    plan = ContractionKernel.compile(left_axes, right_axes, shared)
    left = torch.randn(2, 1, *(2,) * len(left_axes), dtype=dtype).transpose(-1, -2).conj()
    right = torch.randn(1, 3, *(2,) * len(right_axes), dtype=dtype).transpose(-1, -3)
    left.requires_grad_()
    right.requires_grad_()
    ops = _TensorOps("torch", left.real)
    actual = _differentiable_contraction().apply(left, right, plan, ops)
    expected = torch.einsum(f"...{left_axes},...{right_axes}->...acbfdze", left, right)
    torch.testing.assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    for a, b in zip(
        torch.autograd.grad(actual, (left, right), gradient),
        torch.autograd.grad(expected, (left, right), gradient),
        strict=True,
    ):
        tol = 5e-5 if dtype == torch.complex64 else 1e-12
        torch.testing.assert_close(a, b, atol=tol, rtol=tol)


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_offline_cuda_codegen(precision):
    pytest.importorskip("torch")
    pytest.importorskip("triton", minversion="3.4")
    if os.environ.get("TRITON_INTERPRET") == "1":
        pytest.skip("Offline compilation requires the JIT frontend")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile

    from openqml.sim._triton_kernels import _apply_gate, _contract

    def compile_kernel(fn, constants, scalars):
        signature = {name: scalars.get(name, "constexpr") for name in fn.arg_names}
        compiled = compile(
            ASTSource(fn, signature, constants),
            target=GPUTarget("cuda", 80, 32),
            options={"num_warps": 2},
        )
        assert ".target sm_80" in compiled.asm["ptx"] and compiled.asm["cubin"]

    pointers = {name: "*" + precision for name in ("state", "gate", "output")}
    for masks, offsets in [((8,), (0, 8)), ((2, 16), (0, 16, 2, 18)), ((1, 2, 4), tuple(range(8)))]:
        compile_kernel(
            _apply_gate,
            dict(
                GROUPS=128,
                GATE_STRIDE=0,
                MASKS=masks,
                OFFSETS=offsets,
                BLOCK=64,
                BATCH_SHAPE=(3, 2),
                MATRIX_STRIDES=(0, len(offsets) ** 2),
            ),
            pointers,
        )
    compile_kernel(
        _contract,
        dict(
            BATCH=(3, 2),
            LS=(0, 32),
            RS=(16, 0),
            FREE_L=(4, 2, 1),
            FREE_R=(8, 4),
            SUM_L=(0, 8, 16, 24),
            SUM_R=(0, 2, 1, 3),
            CONJ_L=True,
            CONJ_R=False,
            BLOCK=128,
        ),
        {**{n: "*" + precision for n in ("left", "right", "output")}, "count": "i32"},
    )
