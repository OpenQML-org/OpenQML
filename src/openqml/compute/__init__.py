"""Lazy compute namespace with one canonical implementation."""

from .._compat import install_namespace as _install_namespace

_install_namespace(
    __name__,
    "kernel",
    [
        "_cache",
        "_contraction",
        "_contraction_path",
        "_contraction_program",
        "_contraction_reuse",
        "_distributed",
        "_norms",
        "_tensor_batches",
        "_tensor_train",
        "_torch",
        "kernel",
    ],
)
