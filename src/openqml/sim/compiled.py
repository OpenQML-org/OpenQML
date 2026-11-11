"""Compile a complete native circuit graph behind one checked host boundary."""

from collections.abc import Mapping

import numpy as np

from ..compute import get_compute_kernel
from ..errors import ResourceError, ValidationError
from ..ir import Circuit, integer


def compile_circuit(
    circuit,
    observables=None,
    *,
    backend="jax",
    like=None,
    transform=None,
    max_qubits=None,
    max_memory_mb=256,
    max_branches=256,
    compiler=None,
    **options,
):
    """Compile gate construction, evolution and readout as one Torch/JAX graph.

    The callable accepts a parameter mapping and captures an immutable kernel.
    Python gate dispatch runs at tracing, not on each warmed invocation. Supply
    transform=jax.jacrev or torch.func.jacrev to compile derivatives before the
    checked host boundary. Shapes/dtypes/devices follow the native compiler's
    specialization policy. Compilation and checking have their own overhead.
    """
    from .dense import _resource_check, _TensorOps, expectations, statevector

    if not isinstance(circuit, Circuit) or backend not in {"torch", "jax"}:
        raise ValidationError("Provide a Circuit and a native torch/jax backend.")
    if transform is not None and not callable(transform):
        raise ValidationError("transform must be callable.")
    if observables is not None:
        from .batch import _observables

        try:
            observables = _observables(observables, (circuit,))
        except TypeError as exc:
            raise ValidationError("Provide an iterable of Pauli observables.") from exc
    effectful = circuit.has_effects
    if observables == () or (observables is None and effectful):
        raise ValidationError("Effect circuits require nonempty expectation observables.")
    integer(max_branches, "max_branches", 1)
    _resource_check(circuit, max_qubits, max_memory_mb)
    names, kernel = circuit.parameters, get_compute_kernel()
    budget = kernel._prepare_budget(max_memory_mb)
    capacity = budget.max_bytes >> (circuit.num_qubits + 6)
    records = None
    if effectful:
        from .density import _density_program
        from .density import _resource_check as check_density
        from .planning import _record_plan

        check_density(circuit, max_qubits, max_memory_mb)
        density_rows = budget.max_bytes >> (2 * circuit.num_qubits + 6)
        plan = _record_plan(circuit, max_branches=min(max_branches, density_rows))
        capacity = density_rows // max(1, plan.peak_branches)
        records = _density_program(circuit, plan)
    ops = _TensorOps(backend, like)
    limits = dict(backend=backend, like=like, max_qubits=max_qubits, max_memory_mb=max_memory_mb)

    def evaluate(bindings):
        native_limits = limits
        if like is None and names:
            native_limits = {**limits, "like": bindings[names[0]]}
        if observables is None:
            return statevector(circuit, bindings, **native_limits)
        if effectful:
            from .density import _compiled_density, _expectations_from_density

            native = _TensorOps(backend, native_limits["like"])
            rho = _compiled_density(circuit, records, bindings, native)
            return _expectations_from_density(rho, observables, circuit.num_qubits, native)
        return expectations(
            circuit, observables, bindings, max_branches=max_branches, **native_limits
        )

    function = evaluate if transform is None else transform(evaluate)
    if not callable(function):
        raise ValidationError("transform must return a callable.")
    if backend == "torch":
        options.setdefault("fullgraph", True)
        options.setdefault("dynamic", False)
    compiled = kernel.compile(function, backend=backend, compiler=compiler, **options)

    def call(bindings=None):
        bindings = {} if bindings is None else bindings
        if not isinstance(bindings, Mapping) or set(bindings) != set(names):
            raise ValidationError(f"Expected exactly these parameter bindings: {names}.")
        try:
            shape = np.broadcast_shapes(
                *(
                    tuple(value.shape) if hasattr(value, "shape") else np.shape(value)
                    for value in bindings.values()
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError("Circuit parameter batches cannot be broadcast.") from exc
        batch = max(1, int(np.prod(shape, dtype=object)))
        if batch > capacity:
            raise ResourceError("Compiled circuit batch exceeds its working-memory budget.")
        values = {}
        for name in names:
            value = bindings[name]
            module = type(value).__module__.split(".")[0]
            if (backend == "jax" and module == "torch") or (
                backend == "torch" and module in {"jax", "jaxlib"}
            ):
                raise ValidationError("Pass native parameters or NumPy/Python data.")
            if backend == "torch" and module == "torch":
                if value.dtype == ops.xp.bool or value.is_complex():
                    raise ValidationError("Circuit parameters must be real numeric arrays.")
                if not value.is_floating_point():
                    value = value.to(dtype=ops.real_dtype)
            else:
                dtype = getattr(value, "dtype", None)
                if dtype is None:
                    dtype = np.asarray(value).dtype
                if np.dtype(dtype).kind not in "fiu":
                    raise ValidationError("Circuit parameters must be real numeric arrays.")
                if backend == "torch":
                    if isinstance(value, np.ndarray) and (
                        not value.flags.writeable or any(stride < 0 for stride in value.strides)
                    ):
                        value = value.copy()
                    value = ops.xp.as_tensor(value, dtype=ops.real_dtype, device=ops.device)
                else:
                    value = ops.xp.asarray(value, dtype=ops.real_dtype)
            values[name] = value
        return compiled(values)

    call.parameters, call.backend = names, backend
    call.kernel, call.graph, call.eager = kernel, compiled.graph, evaluate
    return call
