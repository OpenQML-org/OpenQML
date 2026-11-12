"""Context-local admission of JAX's GPU without importing or requiring Torch."""

from collections import OrderedDict
from threading import RLock

import jax
import jax.numpy as jnp
import numpy as np

from ...errors import ValidationError
from ...runtime.backends import BackendCapabilities


class JAXNativeBackend:
    name = "jax-gpu"
    capabilities = BackendCapabilities(
        native_expectations=True,
        exact_expectations=True,
        deterministic_expectations=True,
        density_matrix=True,
        channels=True,
        reset=True,
        classical_control=True,
        mid_circuit_measurements=True,
        measurement_expectations=True,
        device_type="GPU",
    )

    def __init__(self):
        self._admitted = OrderedDict()
        self._lock = RLock()

    def is_available(self):
        return jax.default_backend() == "gpu"

    def admit(self, task, circuits, observables, *, options):
        if task != "expectations":
            raise ValidationError("The JAX native candidate admits expectation workloads.")
        from .jax import QuantumLayer

        with self._lock:
            self._admit(circuits, observables, options, QuantumLayer)

    def _admit(self, circuits, observables, options, layer_type):
        for circuit in circuits:
            key = (
                circuit,
                tuple(observables),
                options.max_qubits,
                options.max_memory_mb,
                options.max_branches,
            )
            if key not in self._admitted:
                # Use the very same resource admission as native differentiable
                # layers, with no parameters initialized and no numeric job.
                layer_type(
                    circuit,
                    (),
                    tuple(sorted(circuit.parameters)),
                    observables,
                    backend="jax",
                    max_qubits=options.max_qubits,
                    max_memory_mb=options.max_memory_mb,
                    max_branches=options.max_branches,
                )
                if circuit.parameters:
                    self._admitted[key] = True
            if key in self._admitted:
                self._admitted.move_to_end(key)
            while len(self._admitted) > 16:
                self._admitted.popitem(last=False)

    def expectations_batch(self, circuits, observables, *, options):
        from ...sim.dense import expectations

        self.admit("expectations", circuits, observables, options=options)
        return np.stack(
            [
                np.asarray(
                    expectations(
                        c,
                        observables,
                        backend="jax",
                        like=jnp.asarray(0.0),
                        max_qubits=options.max_qubits,
                        max_memory_mb=options.max_memory_mb,
                        max_branches=options.max_branches,
                    )
                )
                for c in circuits
            ]
        )
