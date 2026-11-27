"""Qml operations over the shared OpenQML representation and compute service."""

from .._lazy import resolve as _resolve

_EXPORTS = {
    "ClassDecision": ("._decision", "ClassDecision"),
    "LindbladPlan": (".lindblad", "LindbladPlan"),
    "LindbladResult": (".lindblad", "LindbladResult"),
    "LindbladUncertainty": (".lindblad", "LindbladUncertainty"),
    "Model": (".models", "Model"),
    "select_predictor": (".selection", "select_predictor"),
    "PredictorSelection": (".selection", "PredictorSelection"),
    "HybridArchitecture": (".architecture", "HybridArchitecture"),
    "QuantumCellSpec": (".architecture", "QuantumCellSpec"),
    "LayerSpec": (".layers", "LayerSpec"),
    "OperatorSpec": (".operators", "OperatorSpec"),
    "DepthSchedule": (".training", "DepthSchedule"),
    "StagewiseOptimizer": (".training", "StagewiseOptimizer"),
    "DeepQuantumPlan": (".deep", "DeepQuantumPlan"),
    "MultiscalePlan": (".multiscale", "MultiscalePlan"),
}
_MODULES = {}
__all__ = tuple(_EXPORTS)


def __getattr__(name):
    return _resolve(name, __name__, _EXPORTS, _MODULES, globals())


def __dir__():
    return sorted(set(globals()) | _EXPORTS.keys())
