"""Resident quantum geometry and geometry-first spectral steps for Torch/JAX."""

from collections.abc import Mapping
from typing import NamedTuple

from ..compute.kernel import _array_backend, compute_operation, get_compute_kernel
from ..errors import ValidationError
from ..ir.core import integer
from ..ir.parameters import _checked, real
from ..sim.dense import _TensorOps, statevector


class NativeMetricResult(NamedTuple):
    direction: object
    converged: object
    iterations: object
    metric_products: object
    residual_norm: object
    damping: float


class SpectralStepResult(NamedTuple):
    step: object
    natural: NativeMetricResult
    alignment: object
    used_natural_direction: object


def _inputs(plan, bindings, vector, backend, sample_weights=None, sample_chunk_size=None):
    if backend not in {"torch", "jax"} or _array_backend(vector) != backend:
        raise ValidationError("Provide a native Torch/JAX vector and its explicit backend.")
    if vector.shape != (len(plan.parameters),):
        raise ValidationError("Native metric vector must match the selected parameter order.")
    ops = _TensorOps(backend, vector)
    if not isinstance(bindings, Mapping) or set(bindings) != set(plan.circuit.parameters):
        raise ValidationError("Native metric bindings must cover exactly the circuit parameters.")
    local = {}
    for name, value in bindings.items():
        if _array_backend(value) not in {"numpy", backend}:
            raise ValidationError("Native metric inputs must use one framework.")
        value = ops.array(_checked(value), real_value=True)
        if value.ndim > 1:
            raise ValidationError("Native geometry bindings require scalars or one sample axis.")
        local[name] = ops.check(value, ops.xp.isfinite(value), "Metric bindings must be finite.")
    mass = ops.array(1 if sample_weights is None else _checked(sample_weights), real_value=True)
    if mass.ndim > 1:
        raise ValidationError("Metric sample weights require one sample axis.")
    shape = ops.broadcast_shape(mass.shape, *(value.shape for value in local.values()))
    if shape and not shape[0]:
        raise ValidationError("Native geometry requires nonempty samples.")
    mass = ops.xp.broadcast_to(mass, shape)
    mass = ops.check(
        mass,
        ops.xp.isfinite(mass) & (mass >= 0),
        "Metric sample weights must be finite and nonnegative.",
    )
    # Scale before normalization so finite large weights cannot overflow a sum.
    peak = mass.max()
    mass = mass / ops.xp.where(peak > 0, peak, 1)
    total = ops.check(mass.sum(), mass.sum() > 0, "Metric sample mass must be positive.")
    mass = mass / total
    vector = ops.check(vector, ops.xp.isfinite(vector), "Metric vectors must be finite.")
    # Model the largest sequential factor's primal/JVP/VJP state workspace.
    # Framework tapes, compiler workspaces and allocator pools remain separate.
    workspace = max(
        (
            16 * (1 << p.circuit.num_qubits) * (16 + 8 * len(p.circuit.operations))
            for p, _ in plan._plans
        ),
        default=0,
    )
    count = shape[0] if shape else 1
    if sample_chunk_size is not None:
        sample_chunk_size = integer(sample_chunk_size, "sample_chunk_size", 1)
    active = count if sample_chunk_size is None else min(count, sample_chunk_size)
    coordinates = sum(value.shape[0] if value.ndim else 1 for value in local.values())
    get_compute_kernel().admit(
        (active * workspace + 64 * coordinates + 64 * count + 7) // 8, plan.max_memory_mb
    )
    return ops, local, vector, mass


@compute_operation
def native_matvec(
    plan, bindings, vector, *, backend="torch", sample_weights=None, sample_chunk_size=None
):
    """Apply Re(QGT) with native JVP/VJP, without a dense parameter metric.

    Realifying the state gives the same convention in both frameworks. Remove
    its radial and global-phase tangent before the reverse pullback. Independent
    state factors add their metrics, including reused global parameters.
    """
    ops, local, vector, mass = _inputs(
        plan, bindings, vector, backend, sample_weights, sample_chunk_size
    )
    return _apply_metric(plan, local, vector, mass, ops, backend, sample_chunk_size)


def _apply_metric(plan, local, vector, mass, ops, backend, sample_chunk_size=None):
    if sample_chunk_size is None or not mass.ndim or mass.shape[0] <= sample_chunk_size:
        return _metric_product(plan, local, vector, mass, ops, backend)
    size, total = sample_chunk_size, mass.shape[0]
    if backend == "jax":
        import jax

        padding = (-total) % size
        weights = ops.xp.pad(mass, (0, padding))
        columns = {
            name: ops.xp.pad(value, (0, padding), mode="edge")
            if value.ndim and value.shape[0] != 1
            else value
            for name, value in local.items()
        }

        def step(result, start):
            selected = {
                name: jax.lax.dynamic_slice_in_dim(value, start, size)
                if value.ndim and value.shape[0] != 1
                else value
                for name, value in columns.items()
            }
            weight = jax.lax.dynamic_slice_in_dim(weights, start, size)
            return result + _metric_product(plan, selected, vector, weight, ops, backend), None

        value, _ = jax.lax.scan(
            jax.checkpoint(step), vector * 0, ops.xp.arange(0, total + padding, size)
        )
        return ops.check(
            value, ops.xp.isfinite(value), "Native quantum geometry exceeds finite precision."
        )
    result = vector * 0
    for start in range(0, total, size):
        selected = {
            name: value[start : start + size] if value.ndim and value.shape[0] != 1 else value
            for name, value in local.items()
        }
        result = result + _metric_product(
            plan, selected, vector, mass[start : start + size], ops, backend
        )
    return ops.check(
        result, ops.xp.isfinite(result), "Native quantum geometry exceeds finite precision."
    )


def _metric_product(plan, local, vector, mass, ops, backend):
    if not plan.parameters:
        return vector * 0
    xp = ops.xp
    if backend == "torch":
        jvp, vjp = xp.func.jvp, xp.func.vjp
    else:
        import jax

        jvp, vjp = jax.jvp, jax.vjp

    def cat(values):
        return xp.cat(values, dim=-1) if backend == "torch" else xp.concatenate(values, axis=-1)

    def stack(values):
        return xp.stack(values, dim=-1) if backend == "torch" else xp.stack(values, axis=-1)

    def dot(a, b):
        value = a * b
        return (
            value.sum(dim=-1, keepdim=True)
            if backend == "torch"
            else value.sum(axis=-1, keepdims=True)
        )

    result = [vector[i] * 0 for i in range(len(plan.parameters))]
    for child_plan, indices in plan._plans:
        names = tuple(plan.parameters[i] for i in indices)
        per_sample = any(local[name].ndim for name in names)
        coordinates = stack(
            tuple(
                ops.xp.broadcast_to(local[name], mass.shape) if per_sample else local[name]
                for name in names
            )
        )
        tangent = xp.broadcast_to(stack(tuple(vector[i] for i in indices)), coordinates.shape)

        def state(values, names=names, child_circuit=child_plan.circuit):
            selected = {name: values[..., j] for j, name in enumerate(names)}
            parameters = {
                name: selected.get(name, local[name]) for name in child_circuit.parameters
            }
            psi = statevector(
                child_circuit,
                parameters,
                backend=backend,
                like=vector,
                max_memory_mb=plan.max_memory_mb,
            )
            psi = xp.broadcast_to(psi, mass.shape + (psi.shape[-1],))
            return cat((psi.real, psi.imag))

        psi, velocity = jvp(state, (coordinates,), (tangent,))
        _, pullback = vjp(state, coordinates)
        half = psi.shape[-1] // 2
        phase = cat((-psi[..., half:], psi[..., :half]))
        projected = velocity - psi * dot(psi, velocity) - phase * dot(phase, velocity)
        product = pullback(projected * mass[..., None])[0]
        if product.ndim > 1:
            product = product.sum(dim=0) if backend == "torch" else product.sum(axis=0)
        for j, index in enumerate(indices):
            result[index] = result[index] + product[j]
    value = xp.stack(tuple(result))
    return ops.check(value, xp.isfinite(value), "Native quantum geometry exceeds finite precision.")


@compute_operation
def native_natural_gradient(
    plan,
    bindings,
    gradient,
    *,
    backend="torch",
    sample_weights=None,
    sample_chunk_size=None,
    **options,
):
    """Solve (Re(QGT)+damping*I)d=g on the selected native device."""
    ops, local, gradient, mass = _inputs(
        plan, bindings, gradient, backend, sample_weights, sample_chunk_size
    )
    return _natural_solve(
        plan, local, gradient, mass, ops, backend, sample_chunk_size=sample_chunk_size, **options
    )


def _natural_solve(
    plan,
    local,
    gradient,
    mass,
    ops,
    backend,
    *,
    metric_scale=1,
    metric_reducer=None,
    sample_chunk_size=None,
    **options,
):
    from ..qml._native_training import solve_metric

    if not plan.parameters:
        raise ValidationError("A natural-gradient solve needs selected parameters.")

    def product(value):
        result = metric_scale * _apply_metric(
            plan, local, value["coordinates"], mass, ops, backend, sample_chunk_size
        )
        if metric_reducer is not None:
            result = metric_reducer(result)
        if (
            _array_backend(result) != backend
            or result.shape != gradient.shape
            or result.dtype != gradient.dtype
        ):
            raise ValidationError(
                "Metric reducers must preserve the native vector shape and dtype."
            )
        return {"coordinates": result}

    result = solve_metric(
        product,
        {"coordinates": gradient},
        ops,
        **options,
    )
    return NativeMetricResult(result.direction["coordinates"], *result[1:])


def _spectral_shapes(block_shapes, size):
    try:
        shapes = tuple(
            tuple(integer(n, "spectral block dimension", 1) for n in shape)
            for shape in block_shapes
        )
    except TypeError as exc:
        raise ValidationError("Spectral blocks require an iterable of matrix shapes.") from exc
    if any(len(s) != 2 for s in shapes) or sum(a * b for a, b in shapes) != size:
        raise ValidationError(
            "Spectral matrix blocks must partition the selected parameter vector."
        )
    return shapes


@compute_operation
def spectral_step(
    plan,
    bindings,
    gradient,
    *,
    block_shapes,
    mixing=0.1,
    learning_rate=0.01,
    max_step_norm=0.1,
    backend="torch",
    sample_weights=None,
    sample_chunk_size=None,
    **options,
):
    """Mix active polar factors AFTER natural preconditioning, then bound the step.

    Geometry-first polar mixing follows arXiv:2611.10105. This stateless step
    uses exact SVD and an observed descent-alignment safeguard. It does not
    implement the paper's momentum or distance-from-initialization schedule.
    The nested natural result certifies the unmodified CG solve only.
    """
    mixing, rate, radius = real(mixing), real(learning_rate), real(max_step_norm)
    if not 0 <= mixing <= 1 or rate <= 0 or radius <= 0:
        raise ValidationError("Use mixing in [0,1] and positive learning rate/step bound.")
    shapes = _spectral_shapes(block_shapes, len(plan.parameters))
    ops, local, gradient, mass = _inputs(
        plan, bindings, gradient, backend, sample_weights, sample_chunk_size
    )
    get_compute_kernel().admit(
        32 * len(plan.parameters) + 8 * sum(min(s) ** 2 for s in shapes), plan.max_memory_mb
    )
    natural = _natural_solve(
        plan, local, gradient, mass, ops, backend, sample_chunk_size=sample_chunk_size, **options
    )
    xp = ops.xp
    direction = _polar_mix(natural.direction, shapes, mixing, ops, backend)
    alignment = (gradient * direction).sum()
    fallback = ~(xp.isfinite(alignment) & (alignment > 0))
    direction = xp.where(fallback, natural.direction, direction)
    norm = ops.norm(direction)
    length = xp.minimum(rate * ops.norm(natural.direction), ops.array(radius, real_value=True))
    step = (direction / xp.where(norm > 0, norm, 1)) * length
    step = ops.check(step, xp.isfinite(step), "Spectral natural step must be finite.")
    return SpectralStepResult(step, natural, (gradient * step).sum(), fallback | (mixing == 0))


def _polar_mix(vector, shapes, mixing, ops, backend):
    xp, parts, start = ops.xp, [], 0
    for a, b in shapes if mixing else ():
        matrix = vector[start : start + a * b].reshape((a, b))
        u, singular, vh = xp.linalg.svd(matrix, full_matrices=False)
        # The partial polar factor must leave zero singular subspaces inactive.
        eps = 2.220446049250313e-16 if "64" in str(matrix.dtype) else 1.1920928955078125e-7
        active = singular > eps * max(a, b) * singular.max()
        polar = ops.matmul(u * active[None, :], vh)
        rank = ops.array(active.sum(), real_value=True)
        norm = ops.norm(matrix.reshape(-1))
        scaled = polar * norm / xp.sqrt(xp.where(rank > 0, rank, 1))
        parts.append(((1 - mixing) * matrix + mixing * scaled).reshape(-1))
        start += a * b
    return vector if not mixing else _cat_parts(parts, backend, xp)


def _cat_parts(parts, backend, xp):
    return xp.cat(tuple(parts)) if backend == "torch" else xp.concatenate(tuple(parts))
