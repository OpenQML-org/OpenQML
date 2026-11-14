"""Regularized SLD solves with implicit derivatives at repeated eigenvalues."""

from ..compute.kernel import _torch_compiling
from ..errors import ConvergenceError


def adjoint(value):
    return value.conj().swapaxes(-1, -2)


def hermitian(value):
    return (value + adjoint(value)) * 0.5


def _spectral(rho, rhs, damping, ops, *, capturing=False):
    values, basis = ops.xp.linalg.eigh(hermitian(rho))
    denominator = (values[..., :, None] + values[..., None, :]) * 0.5 + damping
    valid = ops.xp.isfinite(denominator) & (denominator > 0)
    if capturing:
        ops.xp._assert_async(
            ops.xp.all(valid), "[OpenQML:ConvergenceError] SLD operator is not positive definite."
        )
        denominator = ops.xp.where(valid, denominator, float("nan"))
    else:
        denominator = ops.check(
            denominator,
            valid,
            "SLD damping must keep the density Sylvester operator positive definite.",
            error=ConvergenceError,
        )
    local = ops.matmul(adjoint(basis), ops.matmul(rhs, basis)) / denominator
    return ops.matmul(basis, ops.matmul(local, adjoint(basis)))


def sld_solve(rho, rhs, damping, ops):
    rho, rhs = hermitian(rho), hermitian(rhs)
    if ops.backend == "torch":
        from ._torch_sylvester import solve

        result = solve(rho, rhs, damping, compiling=_torch_compiling())
    elif ops.backend == "jax":
        import jax

        xp = ops.xp

        def pack(value):
            return xp.concatenate((value.real, value.imag), axis=-1)

        def unpack(value):
            real, imaginary = xp.split(value, 2, axis=-1)
            return real + 1j * imaginary

        def product(value):
            value = unpack(value)
            return pack((ops.matmul(rho, value) + ops.matmul(value, rho)) * 0.5 + damping * value)

        result = unpack(
            jax.lax.custom_linear_solve(
                product,
                pack(rhs),
                solve=lambda _, value: pack(_spectral(rho, unpack(value), damping, ops)),
                symmetric=True,
            )
        )
    else:
        result = _spectral(rho, rhs, damping, ops)
    residual = (ops.matmul(rho, result) + ops.matmul(result, rho)) * 0.5 + damping * result - rhs
    tolerance = 2e-10 if rho.dtype == ops.xp.complex128 else 3e-4
    error = ops.norm(residual.reshape(residual.shape[:-2] + (-1,)), axis=-1)
    reference = ops.norm(rhs.reshape(rhs.shape[:-2] + (-1,)), axis=-1)
    return ops.check(
        hermitian(result),
        (error <= tolerance * ops.xp.where(reference > 1, reference, 1))[..., None, None],
        "SLD solve failed its Sylvester residual; increase state damping or precision.",
        error=ConvergenceError,
    )
