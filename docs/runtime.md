# Runtime dispatch

Numeric execution defaults to `backend="auto"`: eligible configured QPU first,
then CUDA GPU, then CPU. Explicit backend names/instances and native tensor
backends pin execution. `get_backend("auto").plan(...)` returns an inspectable
`DispatchPlan` without executing a quantum job.

## Capabilities and configuration

`register_qpu_backend` wraps a caller-initialized execution provider. Availability
is an explicit boolean or caller-owned zero-argument callback. Registration does
not discover credentials, select a cloud account or contact a device. Multiple
QPUs are considered in registration order. Device width, gates, quantum effects,
requested output and resource policy determine eligibility.

A QPU sampler is eligible for counts; an estimator is eligible for declared
expectations. Neither interface supplies a statevector or density matrix.
Finite-shot/provider-precision estimates are permitted by automatic numeric
expectations. `PriorityBackend(require_exact=True)` requires the provider's
explicit `exact_expectations` contract. `deterministic_expectations` separately
controls reuse, and never implies accuracy. Pin `numpy`/`cpu` for local exact
floating-point readouts. Explicit approximation backends remain opt-in.

All selection happens before submission. Unsupported/unavailable devices are
skipped; errors from a selected provider propagate without retrying the quantum
job on another device. Availability callback failures also propagate unless
they explicitly report an unavailable device. A missing QPU configuration
therefore makes the default route GPU, or CPU when CUDA is unavailable.

## Execution and memory

| Device | Implementation |
| --- | --- |
| QPU | Qiskit V2 parameter-vectorized Estimator/Sampler PUBs; one compiled symbolic circuit per topology, layout-aware observables, row-specific counts |
| GPU | Lazy Torch CUDA dependency; shared symbolic Pauli or quantum/classical factor plans, bounded tensor batches, native dense states/densities and terminal sampling |
| CPU | Cached exact symbolic/factor plans, vectorized parameter arithmetic and bounded dense batches; MPS/tableau providers retain their own execution contracts |

GPU work stays in CUDA tensors until readouts/counts are returned. Feedback
sampling currently uses the CPU branch ensemble if no compatible QPU is
configured; exact feedback expectations can use native CUDA density factors.
Highly connected circuits that cannot fit GPU representations can still use
admitted exact CPU MPS/tableau algorithms. Device priority applies to eligible
implementations; a device label cannot bypass representation or memory limits.

Working budgets have no default total-width cap. They bound representation
admission and batch sizes, not process RSS, framework allocator reservations or
autodiff tape retention. Large idle/factored registers do not force a full-state
allocation. Full-state outputs and general entanglement retain exponential cost.

Compiled topology caches are bounded and instance-local. They retain fixed
instructions/channel settings, without measurement results or binding arrays.
Qiskit adapters expose `cache_size` and
`clear_cache()`. Invalidate after changing a pass manager, target or calibration;
those caller-owned objects must remain stable between invalidations. Duplicate
stochastic rows remain separate parameter entries. Sampler counts are extracted
per PUB location rather than summed across parameter bindings.

General quantum layers first select a compatible registered QPU by default.
Without one, they retain native framework differentiation and device placement.
Explicit native backends, compiled readout methods, readout reduction and activation
checkpointing pin that native computation graph and do not submit a QPU job.
PyTorch modules without a device pin prefer available CUDA;
JAX follows its configured device placement. Universal external quantum layers
use batched first-order parameter shifts; native structured layers retain higher
derivatives and compiler transforms.

The provider batching follows the official [Qiskit V2 primitive
broadcasting contract](https://quantum.cloud.ibm.com/docs/en/api/qiskit/primitives)
and [BitArray location semantics](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.BitArray).
