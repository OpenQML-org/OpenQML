"""Keep worker precision choices local, including when workers run in-process."""

from functools import wraps


def scoped_x64(function):
    @wraps(function)
    def call(args, *values, **kwargs):
        enabled = getattr(args, "interface", None) == "jax" or str(
            getattr(args, "case", "")
        ).startswith("jax-")
        if not enabled:
            return function(args, *values, **kwargs)
        import jax

        context = getattr(jax, "enable_x64", None)
        if context is None:
            from jax.experimental import enable_x64

            context = enable_x64
        with context():
            return function(args, *values, **kwargs)

    return call
