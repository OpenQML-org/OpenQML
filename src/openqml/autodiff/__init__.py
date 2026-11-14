"""Autodiff operations over the shared OpenQML representation and compute service."""

from .._lazy import resolve as _resolve

_EXPORTS = {
    "QuantumFisherPlan": (".fisher", "QuantumFisherPlan"),
    "quantum_fisher": (".fisher", "quantum_fisher"),
    "compile_compute": (".compilation", "compile_compute"),
    "parameter_shift": (".gradients", "parameter_shift"),
    "parameter_shift_jacobian": (".gradients", "parameter_shift_jacobian"),
    "parameter_shift_jvp": (".gradients", "parameter_shift_jvp"),
    "parameter_shift_vjp": (".gradients", "parameter_shift_vjp"),
    "adjoint_jvp": (".adjoint", "adjoint_jvp"),
    "adjoint_vjp": (".adjoint", "adjoint_vjp"),
    "quantum_metric": (".geometry", "quantum_metric"),
    "DistanceAwareQNM": (".distance_aware", "DistanceAwareQNM"),
    "DistanceAwareState": (".distance_aware", "DistanceAwareState"),
    "DistanceAwareResult": (".distance_aware", "DistanceAwareResult"),
    "hybrid_qnm": (".optimizers", "hybrid_qnm"),
    "optimize": (".optimization", "optimize"),
}
_MODULES = {}
__all__ = tuple(_EXPORTS)


def __getattr__(name):
    return _resolve(name, __name__, _EXPORTS, _MODULES, globals())


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys())
