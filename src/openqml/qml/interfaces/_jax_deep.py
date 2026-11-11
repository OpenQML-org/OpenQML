"""Convenience entry into the shared functional hybrid architecture."""

from ..architecture import HybridArchitecture


def QuantumDeepNetwork(
    input_size, output_size, *, backend=None, active_depth=None, **architecture_options
):
    return HybridArchitecture.deep(input_size, output_size, **architecture_options).build(
        "jax", backend=backend, active_depth=active_depth
    )
