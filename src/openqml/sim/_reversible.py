"""Native reverse products for a unitary statevector, using its inverse gate sequence."""

from functools import lru_cache

from ._device_ops import matrix_cotangent
from .dense import _apply_matrix, _evolve_unitary, _gate_matrix


@lru_cache(maxsize=1)
def _torch_reversible():
    import torch

    class Reversible(torch.autograd.Function):
        @staticmethod
        def forward(ctx, forward, reverse, initial, *angles):
            final = forward(initial, *angles)
            ctx.save_for_backward(initial, final, *angles)
            ctx.forward_circuit, ctx.reverse_circuit = forward, reverse
            return final

        @staticmethod
        def backward(ctx, gradient):
            initial, final, *angles = ctx.saved_tensors
            if torch.is_grad_enabled():
                # Differentiable replay supplies higher derivatives.
                proxies = tuple(v.view_as(v) for v in (initial, *angles))
                active = [v for v in proxies if v.requires_grad]
                replay = ctx.forward_circuit(*proxies)
                if not replay.requires_grad:
                    return (None,) * (2 + len(proxies))
                grads = iter(
                    torch.autograd.grad(
                        replay, active, gradient, create_graph=True, allow_unused=True
                    )
                )
                return (None, None, *(next(grads) if v.requires_grad else None for v in proxies))

            def vjp(build, inputs):
                if not inputs:
                    return build(), None
                with torch.enable_grad():
                    leaves = [v.detach().requires_grad_(True) for v in inputs]
                    matrix = build(*leaves)
                if not matrix.requires_grad:
                    return matrix, None

                def apply(dm):
                    return torch.autograd.grad(matrix, leaves, dm, allow_unused=True)

                return matrix.detach(), apply

            return (None, None, *ctx.reverse_circuit(final, gradient, angles, vjp))

    return Reversible


def evolve_reversible(state, operations, bindings, width, ops, fusion, budget):
    gate_names = {key for op in operations for p in op.params for key in p.parameters}
    names = tuple(key for key in bindings if key in gate_names)
    # Pass each binding as its own native input. The framework sums contributions
    # when multiple symbols refer to the same model parameter.
    values = tuple(ops.array(bindings[name], real_value=True) for name in names)

    def forward(initial, *angles):
        return _evolve_unitary(
            initial,
            operations,
            dict(zip(names, angles, strict=True)),
            width,
            ops,
            None,
            fusion=fusion,
            max_memory_mb=budget,
        )

    if not names:
        return forward(state)
    slots = {name: i for i, name in enumerate(names)}

    def reverse(final, gradient, angles, local_vjp):
        gradients = [ops.xp.zeros_like(x) for x in angles]
        for gate in reversed(operations):
            if gate.name == "barrier":
                continue
            keys = tuple(sorted({key for p in gate.params for key in p.parameters}))
            inputs = tuple(angles[slots[key]] for key in keys)

            def build(*args, gate=gate, keys=keys):
                local = dict(zip(keys, args, strict=True))
                return _gate_matrix(
                    gate.name, [p.evaluate(local) for p in gate.params], ops, gate.matrix
                )

            matrix, pullback = local_vjp(build, inputs)
            inverse = ops.conj(matrix).swapaxes(-1, -2)
            before = _apply_matrix(final, inverse, gate.qubits, width, ops)
            if pullback is not None:
                dm = matrix_cotangent(before, gradient, gate.qubits, width, ops, matrix.shape)
                for key, delta in zip(keys, pullback(dm), strict=True):
                    if delta is not None:
                        i = slots[key]
                        gradients[i] = gradients[i] + delta
            gradient = _apply_matrix(gradient, inverse, gate.qubits, width, ops)
            final = before
        return gradient, *gradients

    if ops.backend == "jax":
        import jax

        @jax.custom_vjp
        def evolve(initial, *angles):
            return forward(initial, *angles)

        def fwd(initial, *angles):
            final = forward(initial, *angles)
            return final, (final, angles)

        def bwd(residual, cotangent):
            final, angles = residual

            def vjp(build, inputs):
                if not inputs:
                    return build(), None
                matrix, pull = jax.vjp(build, *inputs)

                def apply(dm):
                    # JAX represents complex cotangents via a bilinear pairing.
                    from jax import numpy as jnp

                    axes = tuple(range(dm.ndim - matrix.ndim)) + tuple(
                        i + dm.ndim - matrix.ndim
                        for i, s in enumerate(matrix.shape)
                        if s == 1 and dm.shape[i + dm.ndim - matrix.ndim] != 1
                    )
                    reduced = jnp.sum(dm, axis=axes, keepdims=True).reshape(matrix.shape)
                    return pull(jnp.conj(reduced))

                return matrix, apply

            dx, *derivatives = reverse(final, ops.conj(cotangent), angles, vjp)
            return (ops.conj(dx), *derivatives)

        evolve.defvjp(fwd, bwd)
        return evolve(state, *values)

    import torch

    from ..compute.kernel import _torch_compiling

    if (
        _torch_compiling()
        or not torch.is_grad_enabled()
        or torch._C._are_functorch_transforms_active()
        or any(
            torch.autograd.forward_ad.unpack_dual(v).tangent is not None for v in (state, *values)
        )
    ):
        return forward(state, *values)

    return _torch_reversible().apply(forward, reverse, state, *values)
