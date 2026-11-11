# Runtime dispatch

Local numerical engines share `ComputeKernel` for native arrays, matrix
products, contractions, linear solves and numerical admission. Use
`configure_compute` to establish workspace, validation and metadata-cache
policies before creating models or compiled graphs. See [Shared compute
kernel](compute.md) for defaults and resource accounting.

Numeric execution defaults to `CostAwareBackend`: capability checks first,
then a Pareto comparison of time, price and estimated readout error. Hardware
priority QPU → GPU → CPU only breaks equal forecast scores. Without profiles,
unconstrained local execution uses GPU then CPU; unquoted QPUs are excluded.
Explicit backends pin execution. `get_backend("auto").plan(...)` returns an
inspectable `DispatchPlan` without submitting jobs or probing credentials.

## Cost-aware training

`configure_runtime(policy=..., workload=..., profiles=..., ledger=...)` installs
a process-local policy. It also returns a scheduler that can be passed as an
explicit `backend`. Profiles map backend names to `ExecutionProfile` objects or
read-only callbacks accepting a `SchedulingRequest`. A callback can use the
request's circuits, observables, options, shots and training workload to quote
the actual transpiled layout, provider grouping and calibration regime.

| Input | Meaning |
| --- | --- |
| `TrainingWorkload.batch_size`, `steps` | Circuit rows per sequential optimizer step and number of steps |
| `gradient_evaluations` | Shifted circuits per row; `parameter_shift(circuit, observables)` counts occurrences and controlled-gate recipes |
| `native_gradient_evaluations` | Separate native forward/backward forecast for framework layers |
| `native_devices` | Platforms supporting native autodiff; other platforms include parameter-shift work |
| `submission_batch_size` | Client chunking limit, combined with provider batch limits |
| `microbatch_size`, `gradient_chunk_size` | Sequential training microbatches and shifted-row submission chunks |
| `classical_seconds_per_step`, `classical_cost_per_step` | Encoder, optimizer and other classical work |
| `ExecutionProfile` | Queue, job latency, circuit/shot time, prices, concurrency, groups, mitigation overhead and uncertainty |
| `SchedulingPolicy` | Cost/deadline/error limits, time/cost/error preferences and uncertainty margin |

Forward and gradient phases are sequential; optimizer steps cannot borrow
parallel capacity from future steps. Queue/latency apply to each job wave.
Billing sums total device use, even when jobs overlap. All quotes use the
policy's currency. QPU quotes require explicit queue, pricing and noise values;
zero must be an intentional declaration. Unknown fields are labelled in plans.
Timestamped quotes expire; static declarations remain estimates, not measurements.

`max_batch_size` limits parameter rows per submission. All readouts for a row
are submitted together; `measurement_groups` multiplies physical circuit and
shot work, rather than parameter rows. The default conservatively counts distinct
nonidentity Pauli words; a quote can declare the provider's compiled grouping.
Numeric execution enforces the quoted row limit and preserves result order.
`concurrency > 1` authorizes concurrent provider calls and requires a thread-safe
provider. Optimizer and microbatch dependencies remain sequential.

For a normalized readout the shot model is `RMSE² = bias² + variance / shots`.
Observable coefficient norms scale it. The scheduler chooses the smallest
admissible shot request when a provider exposes `with_shots`; it rejects a
noise floor that shots cannot repair. Qiskit Estimator V2 receives equivalent
statistical precision. Mitigation/randomization can consume additional physical
shots: quote those overheads explicitly. This is an error forecast, not a
certified noise bound or a guarantee about the trained model's accuracy.

`CalibrationModel` estimates logarithmic gate/readout success and optional
T1/T2 decay from a compiled physical circuit. Missing used calibrations fail.
The model accounts for parallel gate timing and physical wire placement; its
bias estimate is a ranking heuristic and omits unmodelled correlated noise.

Successful numeric calls record wall time. Bounded recent residuals adjust
later forecasts within the same topology/readout/batch class. This does not
submit extra calibration jobs or provide a statistical guarantee under drift.
`BudgetLedger` atomically reserves quoted spending before numeric submissions.
Failures retain reservations and never cause automatic resubmission. This is
forecast accounting; provider invoices and already-running cloud GPU charges
remain external. Integrated hybrid-network trainers also reserve their native
and declared classical forecast before each update; custom native loops own
their accounting and optimizer/device scope.

## Capabilities and configuration

`register_qpu_backend` wraps a caller-initialized execution provider. Availability
is an explicit boolean or caller-owned zero-argument callback. Registration does
not discover credentials, select a cloud account or contact a device.
`profile=` accepts the same quote object/callback. Device width, gates, quantum effects,
requested output and resource policy determine eligibility.

A QPU sampler is eligible for counts; an estimator is eligible for declared
expectations. Neither interface supplies a statevector or density matrix.
Finite-shot/provider-precision estimates are permitted subject to the policy.
`CostAwareBackend(require_exact=True)` requires the provider's
explicit `exact_expectations` contract. `deterministic_expectations` separately
controls reuse, and never implies accuracy. Pin `numpy`/`cpu` for local exact
floating-point readouts. Explicit approximation backends remain opt-in.

All selection happens before submission. Unsupported/unavailable devices are
skipped; errors from a selected provider propagate without retrying the quantum
job on another device. Availability callback failures also propagate unless
they explicitly report an unavailable device. A missing QPU configuration
therefore leaves local execution available. `PriorityBackend()` explicitly
retains the previous first-compatible QPU → GPU → CPU policy.

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

General quantum layers compare configured quotes, including parameter-shift
work, before selecting a QPU. Otherwise they retain native differentiation and device placement.
Explicit native backends, compiled readout methods, readout reduction and activation
checkpointing pin that native computation graph and do not submit a QPU job.
PyTorch modules without a device pin prefer available CUDA;
JAX follows its configured device placement. Universal external quantum layers
use batched first-order parameter shifts; native structured layers retain higher
derivatives and compiler transforms.

The provider batching follows the official [Qiskit V2 primitive
broadcasting contract](https://quantum.cloud.ibm.com/docs/en/api/qiskit/primitives)
and [BitArray location semantics](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.BitArray).

The joint scheduling design draws on [Qonductor](https://arxiv.org/abs/2408.04312)
and [Qurator](https://arxiv.org/abs/2604.05505). OpenQML enumerates its finite
candidate Pareto set directly and uses augmented Chebyshev preferences with
configured cost, latency, noise and queue profiles.

## Scoped configuration and job recovery

`runtime_context(backend=...)` or `runtime_context(policy=..., profiles=...)`
selects the automatic scheduler for one thread/async task without changing the
registry. Worker chunks inherit runtime and compute scopes. Each operation
keeps its selected backend and kernel snapshot.

`RecoveryPolicy` bounds retries of result retrieval from an existing provider
job. Qiskit V2 adapters keep the original handle, job ID and PUB row layout;
they do not re-submit on transient retrieval failures. Native result timeouts
are passed to SDKs that support them; SDK networking controls the blocking call
when that interface exposes no timeout. Retry attempts and backoff are bounded.
Qiskit/IBM Runtime wait timeouts are transient; terminal job errors and execution
time-limit failures are not retried automatically.

`BackendJobError.resume()` retries that retained job. A single-job error exposes
`error.pending.to_dict()`; `recover_job(receipt, loader)` retrieves it by ID and
validates results using the recorded layout. Chunked errors retain completed
chunks and per-chunk failures; resumption combines completed data without
re-executing those chunks. Unknown submission state is never replayed.

Registered QPUs use a concurrent circuit breaker. Repeated transient failures
exclude new submissions until cooldown; one probe is admitted after cooldown.
Already submitted jobs remain recoverable. A later automatic scheduling call
can select another eligible backend. Training calls keep their pinned noise
and gradient contract.
