"""Lazy runtime namespace with one canonical implementation."""

from .._compat import install_namespace as _install_namespace

_install_namespace(
    __name__,
    "scheduler",
    [
        "_dependencies",
        "_scheduling",
        "_validation",
        "backends",
        "calibration",
        "costs",
        "environment",
        "recovery",
        "scheduler",
    ],
)
