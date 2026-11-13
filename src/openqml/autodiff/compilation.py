"""Native compiled graphs with checked host-call boundaries."""

from functools import wraps

import numpy as np

from ..compute.kernel import _checked_jax_context, compute_context, get_compute_kernel
from ..errors import ConvergenceError, NumericalError, ResourceError, ValidationError

_ERRORS = {
    c.__name__: c for c in (ConvergenceError, NumericalError, ResourceError, ValidationError)
}


def _raise(message):
    for name, error in _ERRORS.items():
        if f"[OpenQML:{name}]" in message:
            raise error(message.split(f"[OpenQML:{name}]", 1)[1].strip())
    raise NumericalError(message)


def _leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _leaves(item)
    elif hasattr(value, "dtype"):
        yield value
    elif isinstance(value, (float, complex)):
        yield np.asarray(value)


def _check_compiler_error(error):
    """Translate only an observed missing compiler, preserving unrelated failures."""
    from torch._inductor.exc import InvalidCxxCompiler

    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, InvalidCxxCompiler):
            raise ResourceError(
                "TorchInductor could not find a working C++ compiler. On Windows, install "
                "Visual Studio Build Tools with Desktop development with C++ and run from "
                "an x64 Native Tools command prompt (cl must be available). On other systems, "
                "configure a supported C++ toolchain. For graph execution without C++ code "
                "generation, explicitly select compiler='aot_eager' or compiler='eager'; "
                "these backends do not provide Inductor kernel fusion or CUDA Graph replay. "
                "For CUDABackend, leave compile_mode=None to use eager GPU execution."
            ) from error
        pending.extend(
            (current.__cause__, current.__context__, getattr(current, "inner_exception", None))
        )


def compile_compute(
    function, *, backend, compiler=None, kernel=None, check_outputs=True, **options
):
    """Compile once under an immutable kernel snapshot; raise at the call boundary.

    Put differentiation/transforms inside function. The returned host callable
    checks one result boundary; its .graph attribute exposes native execution.
    """
    kernel = get_compute_kernel() if kernel is None else kernel
    if backend == "jax":
        import jax
        from jax.experimental import checkify

        def checked(*args, **kwargs):
            with _checked_jax_context():
                return function(*args, **kwargs)

        graph = jax.jit(checkify.checkify(checked, errors=checkify.user_checks), **options)

        @wraps(function)
        def call(*args, **kwargs):
            with compute_context(kernel=kernel):
                error, result = graph(*args, **kwargs)
                message = error.get()
                if message is not None:
                    _raise(message)
                if check_outputs:
                    flags = [jax.numpy.all(jax.numpy.isfinite(v)) for v in _leaves(result)]
                    if flags and not bool(jax.numpy.all(jax.numpy.stack(flags))):
                        raise NumericalError("Compiled computation returned non-finite values.")
                return result
    elif backend == "torch":
        import torch

        def evaluate(*args, **kwargs):
            # Guard the captured kernel identity as well as the user's function. Separate
            # checked callables must not reuse a graph with another policy.
            if get_compute_kernel()._source_id != kernel._source_id:
                raise ValidationError("Compiled execution changed its kernel snapshot.")
            return function(*args, **kwargs)

        # Dynamo's specialization limit is per code object. Independent checked
        # programs must not consume the same wrapper's recompilation budget.
        evaluate.__code__ = evaluate.__code__.replace(co_name=f"openqml_checked_{id(evaluate):x}")
        graph = torch.compile(evaluate, backend=compiler or "inductor", **options)

        @wraps(function)
        def call(*args, **kwargs):
            with compute_context(kernel=kernel):
                try:
                    result = graph(*args, **kwargs)
                except Exception as exc:
                    _check_compiler_error(exc)
                    if isinstance(exc, (RuntimeError, AssertionError)) and "[OpenQML:" in str(exc):
                        _raise(str(exc))
                    raise
                if check_outputs:
                    flags = {}
                    for value in _leaves(result):
                        if isinstance(value, torch.Tensor):
                            valid = torch.isfinite(value).all()
                            flags[value.device] = valid & flags.get(value.device, True)
                        elif not np.isfinite(value).all():
                            raise NumericalError("Compiled computation returned non-finite values.")
                    if any(not bool(valid) for valid in flags.values()):
                        raise NumericalError("Compiled computation returned non-finite values.")
                return result
    else:
        raise ValidationError("Compiled execution requires torch or jax.")
    call.graph, call.kernel = graph, kernel
    return call
