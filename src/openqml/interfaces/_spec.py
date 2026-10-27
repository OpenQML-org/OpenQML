from collections.abc import Mapping

from ..errors import ValidationError
from ..layers import LayerCheckpoint, LayerSpec, validate_layer


class LayerSpecMixin:
    @classmethod
    def data_reuploading(cls, num_qubits, depth=2, *, entanglement="linear", **kwargs):
        from ..templates import data_reuploading

        spec = data_reuploading(num_qubits, depth, entanglement=entanglement)
        return cls.from_spec(spec, **kwargs)

    @classmethod
    def hardware_efficient(cls, num_qubits, depth=2, **kwargs):
        from ..templates import hardware_efficient

        spec = hardware_efficient(num_qubits, depth)
        return cls(spec.circuit, spec.input_parameters, spec.weight_parameters, **kwargs)

    @classmethod
    def from_program(
        cls,
        program,
        input_parameters,
        weight_parameters=None,
        *,
        framework=None,
        source_options=None,
        **kwargs,
    ):
        if source_options is not None and not isinstance(source_options, Mapping):
            raise ValidationError("source_options must be a mapping.")
        spec = LayerSpec.from_program(
            program,
            input_parameters,
            weight_parameters,
            kwargs.pop("observables", None),
            framework=framework,
            **({} if source_options is None else dict(source_options)),
        )
        return cls.from_spec(spec, **kwargs)

    @classmethod
    def from_spec(cls, spec, **kwargs):
        if not isinstance(spec, LayerSpec):
            raise ValidationError("from_spec requires a LayerSpec.")
        return cls(
            spec.circuit, spec.input_parameters, spec.weight_parameters, spec.observables, **kwargs
        )

    def to_spec(self):
        return LayerSpec(
            getattr(self, "source_circuit", self.circuit),
            self.input_parameters,
            self.weight_parameters,
            self.observables,
        )

    def checkpoint(self, weights=None):
        """Capture concrete weights in source order; call outside JIT/autodiff transforms."""
        import numpy as np

        if weights is None:
            weights = getattr(self, "weights", None)
        if weights is None:
            raise ValidationError("Pass the functional layer's weight vector explicitly.")
        try:
            if type(weights).__module__.startswith("torch"):
                weights = weights.detach().cpu().numpy()
            values = np.asarray(weights)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValidationError(
                "Checkpoint export requires concrete float32/float64 weights."
            ) from exc
        if values.ndim != 1 or values.dtype not in {np.dtype("float32"), np.dtype("float64")}:
            raise ValidationError("Checkpoint export requires a float32/float64 weight vector.")
        return LayerCheckpoint(self.to_spec(), values, values.dtype.name)


def checkpoint_values(checkpoint, spec, dtype):
    import numpy as np

    if not isinstance(checkpoint, LayerCheckpoint):
        raise ValidationError("Expected a LayerCheckpoint.")
    values = checkpoint.weights_for(spec)
    try:
        dtype = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise ValidationError("Checkpoint loading requires float32 or float64.") from exc
    if dtype not in {np.dtype("float32"), np.dtype("float64")}:
        raise ValidationError("Checkpoint loading requires float32 or float64.")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(values, dtype=dtype)
    if not np.isfinite(result).all():
        raise ValidationError("Checkpoint weights exceed the target dtype range.")
    return result


def prepare_layer(circuit, inputs, weights, observables, *, optimize, optimization_options):
    """Validate source ownership before an optional pass can eliminate symbols."""
    if type(optimize) is not bool:
        raise ValidationError("optimize must be a boolean.")
    if optimization_options is not None and (
        not optimize or not isinstance(optimization_options, Mapping)
    ):
        raise ValidationError("optimization_options requires optimize=True and a mapping.")
    names = validate_layer(circuit, inputs, weights, observables)
    report = None
    if optimize:
        from ..optimization import optimize as optimize_circuit

        options = {} if optimization_options is None else dict(optimization_options)
        if set(options) - {"max_operations", "max_depth", "max_passes"}:
            raise ValidationError("Unknown layer optimization option.")
        report = optimize_circuit(circuit, **options)
    return (*names, circuit if report is None else report.circuit, report)


def broadcast_batch_shape(weights, inputs):
    """Broadcast only leading dimensions; input and weight feature widths may differ."""
    import numpy as np

    try:
        return np.broadcast_shapes(weights.shape[:-1], inputs.shape[:-1])
    except ValueError as exc:
        raise ValidationError("Input and weight batch dimensions must be broadcastable.") from exc


def resolve_gradient_method(circuit, backend, method, limits):
    """Keep native graphs by default; adjoint is an explicit NumPy simulator choice."""
    if not isinstance(method, str) or method not in {"auto", "parameter-shift", "adjoint"}:
        raise ValidationError("gradient_method must be auto, parameter-shift or adjoint.")
    native = isinstance(backend, str) and backend in {"torch", "jax"}
    if method == "auto":
        return "backprop" if native else "parameter-shift"
    if native:
        raise ValidationError("Explicit numerical gradient methods require an external backend.")
    if method == "adjoint":
        if not isinstance(backend, str) or backend != "numpy":
            raise ValidationError("Adjoint gradients require backend='numpy'.")
        from ..adjoint import _check_adjoint

        _check_adjoint(circuit, limits["max_qubits"], limits["max_memory_mb"])
    return method


def _sum_to_shape(values, shape):
    """Undo NumPy broadcasting for a first-order numerical pullback."""
    leading = values.ndim - len(shape)
    axes = tuple(range(leading)) + tuple(leading + i for i, size in enumerate(shape) if size == 1)
    return values.sum(axis=axes, keepdims=True).reshape(shape)


def numeric_rows(layer, weights, inputs):
    """Bind only executable symbols while retaining the source-facing interface."""
    import math

    import numpy as np

    weights, inputs = np.asarray(weights), np.asarray(inputs)
    names = layer.input_parameters + layer.weight_parameters
    active = set(layer.circuit.parameters)
    shape = broadcast_batch_shape(weights, inputs)
    count = math.prod(shape)
    flat = np.broadcast_to(inputs, shape + (len(layer.input_parameters),)).reshape(
        count, len(layer.input_parameters)
    )
    flat_weights = np.broadcast_to(weights, shape + (len(layer.weight_parameters),)).reshape(
        count, len(layer.weight_parameters)
    )
    return [
        {name: value for name, value in zip(names, (*row, *w), strict=True) if name in active}
        for row, w in zip(flat, flat_weights, strict=True)
    ]


def numeric_vjp(layer, weights, inputs, cotangents, *, input_grad=True, weight_grad=True):
    """Stream only active shifts, then scatter exact zeros to eliminated columns."""
    import numpy as np

    from ..gradients import parameter_shift_vjp

    weights, inputs = np.asarray(weights), np.asarray(inputs)
    active = set(layer.circuit.parameters)
    selected = tuple(
        name
        for name in (
            (layer.input_parameters if input_grad else ())
            + (layer.weight_parameters if weight_grad else ())
        )
        if name in active
    )
    method, options = (
        parameter_shift_vjp,
        {
            "chunk_size": layer.gradient_chunk_size,
            "backend": layer.execution_backend,
            **layer.limits,
        },
    )
    if layer.gradient_method == "adjoint":
        from ..adjoint import adjoint_vjp

        method, options = (
            adjoint_vjp,
            {name: layer.limits[name] for name in ("max_qubits", "max_memory_mb")},
        )
    values = method(
        layer.circuit,
        layer.observables,
        numeric_rows(layer, weights, inputs),
        np.asarray(cotangents).reshape(-1, len(layer.observables)),
        parameters=selected,
        **options,
    )
    positions = {name: i for i, name in enumerate(selected)}
    shape = broadcast_batch_shape(weights, inputs)
    dx = np.zeros_like(inputs) if input_grad else None
    dw = np.zeros_like(weights) if weight_grad else None
    for i, name in enumerate(layer.input_parameters if input_grad else ()):
        if name in positions:
            dx[..., i] = _sum_to_shape(values[:, positions[name]].reshape(shape), inputs.shape[:-1])
    for i, name in enumerate(layer.weight_parameters if weight_grad else ()):
        if name in positions:
            dw[..., i] = _sum_to_shape(
                values[:, positions[name]].reshape(shape), weights.shape[:-1]
            )
    return dw, dx


def numeric_jvp(layer, weights, inputs, weight_tangent, input_tangent):
    """Shared first-order external directional bridge; never allocate B*M*P."""
    import math

    import numpy as np

    from ..gradients import parameter_shift_jvp

    weights, inputs = np.asarray(weights), np.asarray(inputs)
    shape = broadcast_batch_shape(weights, inputs)
    count = math.prod(shape)
    source_names = layer.input_parameters + layer.weight_parameters
    active = set(layer.circuit.parameters)
    indices = [i for i, name in enumerate(source_names) if name in active]
    names = tuple(source_names[i] for i in indices)
    dx = np.broadcast_to(input_tangent, shape + (len(layer.input_parameters),)).reshape(
        count, len(layer.input_parameters)
    )
    dw = np.broadcast_to(weight_tangent, shape + (len(layer.weight_parameters),)).reshape(
        count, len(layer.weight_parameters)
    )
    method, options = (
        parameter_shift_jvp,
        {
            "chunk_size": layer.gradient_chunk_size,
            "backend": layer.execution_backend,
            **layer.limits,
        },
    )
    if layer.gradient_method == "adjoint":
        from ..adjoint import adjoint_jvp

        method, options = (
            adjoint_jvp,
            {name: layer.limits[name] for name in ("max_qubits", "max_memory_mb")},
        )
    result = method(
        layer.circuit,
        layer.observables,
        numeric_rows(layer, weights, inputs),
        np.concatenate((dx, dw), axis=-1)[:, indices],
        parameters=names,
        **options,
    )
    return result.reshape(shape + (len(layer.observables),)).astype(inputs.dtype)
