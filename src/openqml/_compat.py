"""Canonical implementations with lazy compatibility for published import paths."""

import sys
from importlib import import_module
from importlib.machinery import ModuleSpec, PathFinder
from types import MappingProxyType, ModuleType

_ALIASES = MappingProxyType(
    {
        "openqml._cache": "openqml.compute._cache",
        "openqml._checkpoint_archive": "openqml.qml._archive",
        "openqml._circuit_compile": "openqml.sim.compiled",
        "openqml._compiled": "openqml.autodiff.compilation",
        "openqml._contraction_path": "openqml.compute._contraction_path",
        "openqml._contraction_program": "openqml.compute._contraction_program",
        "openqml._contraction_reuse": "openqml.compute._contraction_reuse",
        "openqml._cuda_backend": "openqml.sim._cuda",
        "openqml._dependencies": "openqml.runtime._dependencies",
        "openqml._distributed_tensor": "openqml.compute._distributed",
        "openqml._exact_instrument": "openqml.sim._instrument",
        "openqml._gate_fusion": "openqml.sim._fusion",
        "openqml._gradient_common": "openqml.autodiff._common",
        "openqml._hybrid_attention": "openqml.qml._attention",
        "openqml._hybrid_graph": "openqml.qml._graph",
        "openqml._hybrid_operators": "openqml.qml.operators",
        "openqml._hybrid_training": "openqml.qml.training",
        "openqml._model_checkpoint": "openqml.qml._checkpoint",
        "openqml._mps": "openqml.sim._mps",
        "openqml._native_training": "openqml.qml._native_training",
        "openqml._native_verification": "openqml.interop._native_verification",
        "openqml._norms": "openqml.compute._norms",
        "openqml._operation_buffer": "openqml.ir._buffer",
        "openqml._parameter_batches": "openqml.interop._parameter_batches",
        "openqml._qasm": "openqml.interop._qasm",
        "openqml._readouts": "openqml.sim._readouts",
        "openqml._scheduling": "openqml.runtime._scheduling",
        "openqml._serialization": "openqml.ir._serialization",
        "openqml._structure": "openqml.ir._structure",
        "openqml._tensor_batches": "openqml.compute._tensor_batches",
        "openqml._tensor_contraction": "openqml.compute._contraction",
        "openqml._tensor_train": "openqml.compute._tensor_train",
        "openqml._torch_compute": "openqml.compute._torch",
        "openqml._validation": "openqml.runtime._validation",
        "openqml.adapters": "openqml.interop.adapters",
        "openqml.adapters._cirq_control": "openqml.interop.adapters._cirq_control",
        "openqml.adapters._common": "openqml.interop.adapters._common",
        "openqml.adapters._qiskit_switch": "openqml.interop.adapters._qiskit_switch",
        "openqml.adapters.cirq": "openqml.interop.adapters.cirq",
        "openqml.adapters.openqasm": "openqml.interop.adapters.openqasm",
        "openqml.adapters.pennylane": "openqml.interop.adapters.pennylane",
        "openqml.adapters.qiskit": "openqml.interop.adapters.qiskit",
        "openqml.adaptive": "openqml.sim.adaptive",
        "openqml.adaptive_amplitude": "openqml.sim.adaptive_amplitude",
        "openqml.adjoint": "openqml.autodiff.adjoint",
        "openqml.aer": "openqml.interop.aer",
        "openqml.amplitude": "openqml.sim.amplitude",
        "openqml.architecture": "openqml.qml.architecture",
        "openqml.architecture_runtime": "openqml.qml.execution",
        "openqml.autoregressive": "openqml.qml.autoregressive",
        "openqml.backends": "openqml.runtime.backends",
        "openqml.batch": "openqml.sim.batch",
        "openqml.calibration": "openqml.runtime.calibration",
        "openqml.compiler": "openqml.ir.compiler",
        "openqml.components": "openqml.sim.components",
        "openqml.conversion": "openqml.interop.conversion",
        "openqml.costs": "openqml.runtime.costs",
        "openqml.deep": "openqml.qml.deep",
        "openqml.density": "openqml.sim.density",
        "openqml.environment": "openqml.runtime.environment",
        "openqml.execution": "openqml.sim.execution",
        "openqml.frontends": "openqml.interop.frontends",
        "openqml.geometry": "openqml.autodiff.geometry",
        "openqml.gradients": "openqml.autodiff.gradients",
        "openqml.inspection": "openqml.interop.inspection",
        "openqml.interfaces": "openqml.qml.interfaces",
        "openqml.interfaces._channel_execution": "openqml.qml.interfaces._channel_execution",
        "openqml.interfaces._flax_channels": "openqml.qml.interfaces._flax_channels",
        "openqml.interfaces._flax_network": "openqml.qml.interfaces._flax_network",
        "openqml.interfaces._jax_ad": "openqml.qml.interfaces._jax_ad",
        "openqml.interfaces._jax_backend": "openqml.qml.interfaces._jax_backend",
        "openqml.interfaces._jax_channels": "openqml.qml.interfaces._jax_channels",
        "openqml.interfaces._jax_deep": "openqml.qml.interfaces._jax_deep",
        "openqml.interfaces._jax_hybrid": "openqml.qml.interfaces._jax_hybrid",
        "openqml.interfaces._jax_network": "openqml.qml.interfaces._jax_network",
        "openqml.interfaces._jax_training": "openqml.qml.interfaces._jax_training",
        "openqml.interfaces._network_checkpoint": "openqml.qml.interfaces._network_checkpoint",
        "openqml.interfaces._spec": "openqml.qml.interfaces._spec",
        "openqml.interfaces._torch_ad": "openqml.qml.interfaces._torch_ad",
        "openqml.interfaces._torch_channels": "openqml.qml.interfaces._torch_channels",
        "openqml.interfaces._torch_deep": "openqml.qml.interfaces._torch_deep",
        "openqml.interfaces._torch_hybrid": "openqml.qml.interfaces._torch_hybrid",
        "openqml.interfaces._torch_network": "openqml.qml.interfaces._torch_network",
        "openqml.interfaces._torch_training": "openqml.qml.interfaces._torch_training",
        "openqml.interfaces.flax": "openqml.qml.interfaces.flax",
        "openqml.interfaces.jax": "openqml.qml.interfaces.jax",
        "openqml.interfaces.torch": "openqml.qml.interfaces.torch",
        "openqml.iqp": "openqml.qml.iqp",
        "openqml.layers": "openqml.qml.layers",
        "openqml.measurement": "openqml.sim.measurement",
        "openqml.mitigation": "openqml.sim.mitigation",
        "openqml.models": "openqml.qml.models",
        "openqml.multiscale": "openqml.qml.multiscale",
        "openqml.observables": "openqml.ir.observables",
        "openqml.optimization": "openqml.autodiff.optimization",
        "openqml.parameters": "openqml.ir.parameters",
        "openqml.pauli": "openqml.sim.pauli",
        "openqml.pauli_fusion": "openqml.ir.pauli_fusion",
        "openqml.pauli_transfer": "openqml.sim.pauli_transfer",
        "openqml.planning": "openqml.sim.planning",
        "openqml.program": "openqml.interop.program",
        "openqml.program_compilation": "openqml.interop.program_compilation",
        "openqml.providers": "openqml.interop.providers",
        "openqml.quantum_activations": "openqml.qml.activations",
        "openqml.quantum_cells": "openqml.qml.cells",
        "openqml.quantum_channels": "openqml.qml.channels",
        "openqml.quantum_generative": "openqml.qml.generative",
        "openqml.quantum_measurement": "openqml.qml.measurement",
        "openqml.readout": "openqml.sim.readout",
        "openqml.recovery": "openqml.runtime.recovery",
        "openqml.resources": "openqml.sim.resources",
        "openqml.semantics": "openqml.interop.semantics",
        "openqml.shadows": "openqml.sim.shadows",
        "openqml.simulator": "openqml.sim.dense",
        "openqml.sparse_pauli": "openqml.sim.sparse_pauli",
        "openqml.templates": "openqml.ir.templates",
        "openqml.verification": "openqml.interop.verification",
    }
)
_PACKAGES = frozenset(["openqml.adapters", "openqml.interfaces"])


class _AliasLoader:
    def __init__(self, alias, target):
        self.alias, self.target = alias, target

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Import locks serialize canonical initialization. Both paths then name
        # the same module, preserving class identity and monkeypatch behavior.
        sys.modules[self.alias] = import_module(self.target)


class _AliasFinder:
    def find_spec(self, fullname, path=None, target=None):
        canonical = _ALIASES.get(fullname)
        if canonical is None:
            return None
        return ModuleSpec(
            fullname, _AliasLoader(fullname, canonical), is_package=fullname in _PACKAGES
        )


_FINDER = _AliasFinder()


class _Namespace(ModuleType):
    """Light parent package; implementation globals retain one owner."""

    def __getattribute__(self, name):
        storage = super().__getattribute__("__dict__")
        if name in storage.get("_namespace_forwarded", ()):
            implementation = import_module(storage["_namespace_implementation"])
            value = getattr(implementation, name)
            storage[name] = value
            return value
        return super().__getattribute__(name)

    def __getattr__(self, name):
        if name in self.__dict__["_namespace_children"]:
            return import_module(self.__name__ + "." + name)
        implementation = import_module(self.__dict__["_namespace_implementation"])
        if name == "__all__" and not hasattr(implementation, name):
            return tuple(key for key in vars(implementation) if not key.startswith("_"))
        value = getattr(implementation, name)
        self.__dict__[name] = value
        self.__dict__["_namespace_forwarded"].add(name)
        return value

    def __setattr__(self, name, value):
        if (
            name.startswith("__")
            or name.startswith("_namespace_")
            or name in self.__dict__.get("_namespace_children", ())
        ):
            return super().__setattr__(name, value)
        setattr(import_module(self.__dict__["_namespace_implementation"]), name, value)
        self.__dict__[name] = value
        self.__dict__["_namespace_forwarded"].add(name)

    def __delattr__(self, name):
        if name in self.__dict__["_namespace_forwarded"]:
            delattr(import_module(self.__dict__["_namespace_implementation"]), name)
            self.__dict__["_namespace_forwarded"].discard(name)
        if name in self.__dict__:
            return super().__delattr__(name)
        delattr(import_module(self.__dict__["_namespace_implementation"]), name)

    def __dir__(self):
        implementation = import_module(self.__dict__["_namespace_implementation"])
        return sorted(set(self.__dict__) | set(dir(implementation)))


def install_namespace(name, implementation, children):
    namespace = sys.modules[name]
    namespace.__dict__["_namespace_implementation"] = name + "." + implementation
    namespace.__dict__["_namespace_children"] = frozenset(children)
    namespace.__dict__["_namespace_forwarded"] = set()
    namespace.__class__ = _Namespace


def install_aliases():
    # CPython 3.10/3.11 dataclasses can observe a half-initialized typing
    # module when independent submodules first import in different threads.
    import_module("typing")
    if _FINDER not in sys.meta_path:
        # Precede PathFinder so old package children cannot initialize a second
        # copy from the canonical directory. Unrelated imports return None.
        index = next(
            (i for i, finder in enumerate(sys.meta_path) if finder is PathFinder),
            len(sys.meta_path),
        )
        sys.meta_path.insert(index, _FINDER)
