"""Convenience entry into the shared hybrid architecture, without a second model."""

from ..architecture import HybridArchitecture


def QuantumDeepNetwork(
    input_size,
    output_size,
    *,
    dtype=None,
    device=None,
    backend=None,
    active_depth=None,
    **architecture_options,
):
    return HybridArchitecture.deep(input_size, output_size, **architecture_options).build(
        "torch", dtype=dtype, device=device, backend=backend, active_depth=active_depth
    )
