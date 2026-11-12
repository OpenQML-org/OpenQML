"""Resolve a kernel at trace time without putting ContextVar operations in graphs."""

from typing import NamedTuple

import torch

from .kernel import ComputeKernel, _ByteBudget


@torch._dynamo.assume_constant_result
def _policy_snapshot():
    from .kernel import _COMPUTE_CONTEXT, _KERNEL

    kernel = _COMPUTE_CONTEXT.get() or _KERNEL
    policy = kernel.policy
    return (
        policy.max_memory_mb,
        policy.cache_entries,
        policy.cache_bytes,
        policy.validation_atol,
        policy.validation_rtol,
        kernel._memory_bytes,
        kernel._source_id,
    )


@torch._dynamo.assume_constant_result
def _budget_snapshot(requested):
    from .kernel import _COMPUTE_CONTEXT, _KERNEL

    kernel = _COMPUTE_CONTEXT.get() or _KERNEL
    return tuple(kernel._prepare_budget(requested))


class _GraphPolicy(NamedTuple):
    max_memory_mb: float | None
    cache_entries: int
    cache_bytes: int
    validation_atol: float
    validation_rtol: float


class _GraphKernel(ComputeKernel):
    def __init__(self, snapshot):
        # A native graph needs immutable policy, not Python locks or plan caches.
        # Inherited primitives stay identical to the host kernel's operations.
        self._policy = _GraphPolicy(*snapshot[:5])
        self._memory_bytes = snapshot[5]
        self._source_id = snapshot[6]
        self._default_budget = _ByteBudget(
            min(256.0, self._policy.max_memory_mb)
            if self._policy.max_memory_mb is not None
            else 256.0,
            min(256 << 20, self._memory_bytes) if self._memory_bytes is not None else 256 << 20,
        )

    def _prepare_budget(self, requested=256):
        return _ByteBudget(*_budget_snapshot(requested))


def captured_kernel():
    return _GraphKernel(_policy_snapshot())
