from collections.abc import Mapping

from ..errors import ValidationError
from ..layers import LayerCheckpoint, LayerSpec, validate_layer


class LayerSpecMixin:
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


def numeric_rows(layer, weights, inputs):
    """Bind only executable symbols while retaining the source-facing interface."""
    import math

    import numpy as np

    weights, inputs = np.asarray(weights), np.asarray(inputs)
    names = layer.input_parameters + layer.weight_parameters
    active = set(layer.circuit.parameters)
    flat = np.asarray(inputs).reshape(math.prod(inputs.shape[:-1]), len(layer.input_parameters))
    return [
        {name: value for name, value in zip(names, (*row, *weights), strict=True) if name in active}
        for row in flat
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
    values = parameter_shift_vjp(
        layer.circuit,
        layer.observables,
        numeric_rows(layer, weights, inputs),
        np.asarray(cotangents).reshape(-1, len(layer.observables)),
        parameters=selected,
        chunk_size=layer.gradient_chunk_size,
        backend=layer.execution_backend,
        **layer.limits,
    )
    positions = {name: i for i, name in enumerate(selected)}
    dx = np.zeros_like(inputs) if input_grad else None
    dw = np.zeros_like(weights) if weight_grad else None
    for i, name in enumerate(layer.input_parameters if input_grad else ()):
        if name in positions:
            dx[..., i] = values[:, positions[name]].reshape(inputs.shape[:-1])
    for i, name in enumerate(layer.weight_parameters if weight_grad else ()):
        if name in positions:
            dw[i] = values[:, positions[name]].sum()
    return dw, dx


def numeric_jvp(layer, weights, inputs, weight_tangent, input_tangent):
    """Shared first-order external directional bridge; never allocate B*M*P."""
    import math

    import numpy as np

    from ..gradients import parameter_shift_jvp

    weights, inputs = np.asarray(weights), np.asarray(inputs)
    shape = inputs.shape[:-1]
    count = math.prod(shape)
    source_names = layer.input_parameters + layer.weight_parameters
    active = set(layer.circuit.parameters)
    indices = [i for i, name in enumerate(source_names) if name in active]
    names = tuple(source_names[i] for i in indices)
    dx = np.asarray(input_tangent).reshape(count, len(layer.input_parameters))
    dw = np.broadcast_to(weight_tangent, (count, len(layer.weight_parameters)))
    result = parameter_shift_jvp(
        layer.circuit,
        layer.observables,
        numeric_rows(layer, weights, inputs),
        np.concatenate((dx, dw), axis=-1)[:, indices],
        parameters=names,
        chunk_size=layer.gradient_chunk_size,
        backend=layer.execution_backend,
        **layer.limits,
    )
    return result.reshape(shape + (len(layer.observables),)).astype(inputs.dtype)
