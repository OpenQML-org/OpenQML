"""Lazy ir namespace with one canonical implementation."""

from .._compat import install_namespace as _install_namespace

_install_namespace(
    __name__,
    "core",
    [
        "_buffer",
        "_serialization",
        "_structure",
        "compiler",
        "core",
        "observables",
        "parameters",
        "pauli_fusion",
        "templates",
    ],
)
