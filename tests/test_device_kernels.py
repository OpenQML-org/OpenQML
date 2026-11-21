"""Numerical and code-generation checks for the actual OpenQML kernel sources."""

import os

import pytest


def _runtime():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton", minversion="3.4")
    if os.environ.get("TRITON_INTERPRET") == "1":
        return torch, "cpu"
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Device execution requires SM80 or a configured interpreter.")
    return torch, "cuda"


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
@pytest.mark.parametrize("wires", [(2,), (4, 0), (2, 0, 3)])
def test_gate_kernel_matches_portable_complex_vjp(wires, precision):
    torch, device = _runtime()
    from openqml.sim._device_ops import _differentiable_gate
    from openqml.sim.dense import _apply_matrix_portable, _TensorOps

    dtype = getattr(torch, precision)
    dimension = 1 << len(wires)
    state = torch.randn(2, 3, 256, dtype=dtype, device=device)[..., ::2].requires_grad_()
    ops = _TensorOps("torch", state.real)
    for shape in ((), (2, 1), (1, 3), (2, 3)):
        gate = torch.randn(
            *shape, dimension, dimension, dtype=dtype, device=device
        ).mH.requires_grad_()
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
    torch, device = _runtime()
    from openqml.compute._contraction_program import ContractionKernel
    from openqml.sim._device_ops import _differentiable_contraction
    from openqml.sim.dense import _TensorOps

    dtype = getattr(torch, precision)
    left_axes, right_axes = "ac" + shared[::-1] + "bf", "d" + shared + "ze"
    plan = ContractionKernel.compile(left_axes, right_axes, shared)
    left = (
        torch.randn(2, 1, *(2,) * len(left_axes), dtype=dtype, device=device)
        .transpose(-1, -2)
        .conj()
    )
    right = torch.randn(1, 3, *(2,) * len(right_axes), dtype=dtype, device=device).transpose(-1, -3)
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


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
@pytest.mark.parametrize("matrix_batch", [(), (1, 3), (2, 1), (2, 3)])
def test_gate_cotangent_reduces_strided_amplitudes_and_shared_batches(precision, matrix_batch):
    torch, device = _runtime()
    from openqml.sim._triton_kernels import launch_gate_cotangent
    from openqml.sim.dense import _apply_matrix_portable, _TensorOps

    dtype = getattr(torch, precision)
    wires = (7, 1)
    state = torch.randn(3, 2, 1024, dtype=dtype, device=device).transpose(0, 1)[..., ::2].conj()
    grad = torch.randn(2, 3, 1024, dtype=dtype, device=device)[..., 1::2].conj()
    gate = torch.randn(*matrix_batch, 4, 4, dtype=dtype, device=device, requires_grad=True)
    result = _apply_matrix_portable(state, gate, wires, 9, _TensorOps("torch", state.real))
    (expected,) = torch.autograd.grad(result, gate, grad)
    got = launch_gate_cotangent(state, grad, wires, 9, gate.shape)
    torch.testing.assert_close(
        got, expected, atol=5e-5 if precision == "complex64" else 1e-11, rtol=1e-5
    )


@pytest.mark.parametrize("precision", ["complex64", "complex128"])
def test_paired_pauli_readout_and_pullback_match_independent_local_matrices(precision):
    torch, device = _runtime()
    from openqml.sim._device_ops import _differentiable_pauli
    from openqml.sim.dense import _FIXED, _apply_matrix_portable, _pauli_expectation, _TensorOps

    dtype = getattr(torch, precision)
    width = 13
    raw = torch.randn(3, 2, 2 << width, dtype=dtype, device=device) / (1 << width) ** 0.5
    state = raw.transpose(0, 1)[..., 1::2].conj().requires_grad_()
    weight = state.real.new_tensor([0.5, -0.75, 0.0]).expand(2, 3)
    ops = _TensorOps("torch", state.real)
    words = (
        "IX" + "I" * 10 + "Z",
        "Z" + "I" * 11 + "Y",
        "I" * 5 + "Y" + "I" * 6 + "Y",
        "X" + "I" * 9 + "YYY",
        "X" + "I" * 8 + "YYYY",
    )
    for word in words:
        transformed = state
        for wire, pauli in enumerate(word):
            if pauli != "I":
                matrix = torch.as_tensor(_FIXED[pauli.lower()], device=device, dtype=dtype)
                transformed = _apply_matrix_portable(transformed, matrix, (wire,), width, ops)
        expected = (state.conj() * transformed).sum(-1).real
        actual = (
            _pauli_expectation(state, word, width, ops)
            if device == "cuda"
            else _differentiable_pauli().apply(state, word)
        )
        assert type(actual.grad_fn).__name__ == "PauliExpectationBackward"
        torch.testing.assert_close(actual, expected)
        (grad,) = torch.autograd.grad(actual, state, weight, retain_graph=True)
        (reference,) = torch.autograd.grad(expected, state, weight)
        torch.testing.assert_close(grad, reference)
        if word == words[1]:
            (first,) = torch.autograd.grad(actual, state, weight, create_graph=True)
            torch.testing.assert_close(first, reference)
            (second,) = torch.autograd.grad(first, state, state.detach())
            torch.testing.assert_close(second, reference)


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_offline_cuda_codegen(precision):
    pytest.importorskip("torch")
    pytest.importorskip("triton", minversion="3.4")
    if os.environ.get("TRITON_INTERPRET") == "1":
        pytest.skip("Offline compilation requires the JIT frontend")
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile

    from openqml.sim._triton_kernels import _apply_gate, _contract, _gate_cotangent, _pauli_pair

    def compile_kernel(fn, constants, scalars):
        signature = {name: scalars.get(name, "constexpr") for name in fn.arg_names}
        compiled = compile(
            ASTSource(fn, signature, constants),
            target=GPUTarget("cuda", 80, 32),
            options={"num_warps": 2},
        )
        assert ".target sm_80" in compiled.asm["ptx"] and compiled.asm["cubin"]

    for backward in (False, True):
        compile_kernel(
            _pauli_pair,
            dict(
                PAIRS=4096,
                FLIP=4097,
                SIGN=4099,
                PIVOT=4096,
                PHASE=1,
                SHAPE=(3, 2),
                XS=(32768, 16384),
                WS=(1, 0),
                LAST=2,
                CONJ=True,
                BACKWARD=backward,
                BLOCK=256,
            ),
            {name: "*" + precision for name in ("state", "weight", "output")},
        )
    pointers = {name: "*" + precision for name in ("state", "gate", "output")}
    compile_kernel(
        _gate_cotangent,
        dict(
            GROUPS=128,
            COUNT=384,
            PARTS=3,
            DIM=4,
            MASKS=(2, 128),
            WIRE_BITS=(128, 2),
            SHAPE=(3, 2),
            BROADCAST=(True, False),
            XS=(2048, 1024),
            GS=(1024, 3072),
            X_LAST=2,
            G_LAST=1,
            X_CONJ=True,
            G_CONJ=False,
            BLOCK=128,
        ),
        {name: "*" + precision for name in ("state", "grad", "partial")},
    )
    for masks, offsets in [((8,), (0, 8)), ((2, 16), (0, 16, 2, 18)), ((1, 2, 4), tuple(range(8)))]:
        compile_kernel(
            _apply_gate,
            dict(
                GROUPS=128,
                MASKS=masks,
                OFFSETS=offsets,
                BLOCK=64,
                BATCH_SHAPE=(3, 2),
                MATRIX_STRIDES=(0, len(offsets) ** 2),
                STATE_STRIDES=(1024, 3072),
                STATE_LAST=2,
                GATE_ROW=1,
                GATE_COL=len(offsets),
                STATE_CONJ=True,
                GATE_CONJ=True,
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
