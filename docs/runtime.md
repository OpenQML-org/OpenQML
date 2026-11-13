# Runtime dispatch

## Native C++/CUDA simulation

`get_backend("aer-gpu")` selects Qiskit Aer's native GPU statevector engine.
The simulator's C++/CUDA implementation performs gate fusion, state evolution
and readout reductions; OpenQML supplies IR compilation, budgets and dispatch.
Readout execution returns observables without exporting the complete state.
This is an integration with Aer/cuQuantum, not a separately implemented
OpenQML C++ statevector simulator.

| Execution path | OpenQML's implementation | External kernel/runtime provider |
| --- | --- | --- |
| `AerBackend(device="GPU")` | IR export, parameter batching, memory admission, dispatch | Qiskit Aer C++/CUDA; requested cuStateVec/cuTensorNet support depends on its build |
| `CUDABackend()` | Circuit/readout planning, register schedules, tensor evolution | PyTorch CUDA tensor kernels |
| `CUDABackend(compile_mode="default")` | Shared readout program, packed parameters, bounded graph cache | PyTorch TorchInductor code generation |

Both adapters expose `execution_info()` without loading an SDK or accessing a
device. This describes configuration and provenance. `requested_accelerator`
on Aer records a request, and `execution_verified=False` explicitly prevents
treating that configuration as evidence that a vendor accelerator ran.
OpenQML currently has no independently implemented C++/CUDA kernel engine.

On supported Linux x86-64 systems, install `openqml[aer-gpu]` with a compatible
NVIDIA driver/runtime. Use an environment containing the GPU Aer distribution;
do not install both `qiskit-aer` and `qiskit-aer-gpu`, which share a Python
namespace. The portable `all` extra retains the CPU Aer distribution.

```python
import openqml as oq

engine = oq.AerBackend(
    name="native-gpu",
    device="GPU",
    precision="double",
    cu_statevec=True,
    target_gpus=(0,),
)
```

`cu_statevec=True` requests NVIDIA cuStateVec from an Aer build containing that
support. `method="tensor_network", device="GPU"` selects Aer's cuTensorNet
implementation. `target_gpus` chooses devices; `blocking_qubits` enables native
statevector/density cache blocking across GPUs or an MPI-enabled Aer build.
The caller provides the native runtime, MPI build and launch configuration.
For GPU shot/parameter batches, `batched_shots_gpu=True` and
`runtime_parameter_binding=True` select Aer's native batching path; cuStateVec
and batched shots are mutually exclusive. GPU-unavailable requests raise an
error. They do not silently execute on a CPU. Initialized GPU backends registered
with OpenQML participate in automatic scheduling; native training candidates
retain their framework/autodiff contract.

There is no fixed 24-qubit ceiling. Dense state memory still grows exponentially,
and tensor-network cost depends on circuit structure. Resource admission and
the vendor's actual memory limits apply. These configuration paths do not
establish a measured large-scale speedup or multi-GPU/MPI execution result.
See [Aer GPU options](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html)
and [NVIDIA cuQuantum integrations](https://docs.nvidia.com/cuda/cuquantum/latest/getting-started/index.html).

## Native tensor GPU execution

`CUDABackend(precision="float32", device="cuda:0")` explicitly selects native
Torch CUDA readouts. Float64 remains the default; single precision trades
accuracy for reduced tensor storage and hardware-dependent arithmetic speed.
Admission remains conservative and validation tolerances do not silently relax.
Select tolerances appropriate to the task through `ComputePolicy`.

Readout and sampling sweeps prepare each IR block once and pack occurrence
parameters into one contiguous matrix. Each GPU chunk transfers one packed
buffer, with device-side column views, instead of transferring every rotation
separately. Batched state output and encoded sampling counts likewise use
combined downloads. Device selection is preserved in concurrent workers.
These changes reduce transfer calls; measured end-to-end speed depends on the
device, workload and native compiler.

For training, use native `backend="torch"` tensors on CUDA or JAX's configured
GPU placement. These paths retain device arrays and autodiff. The numeric
`CUDABackend` adapter returns host readouts and is intended for numeric dispatch.
The manual GPU workflow requires real devices, checks float32/float64 readouts,
sampling, resident gradients and financial factor solves, and optionally checks
native Aer GPU execution and two-GPU NCCL execution. It fails when required
hardware is absent; CPU checks do not establish accelerator performance.

## Compiled CUDA readouts

`CUDABackend(compile_mode="reduce-overhead", device="cuda:0")` compiles admitted
Pauli readout programs through TorchInductor with CUDA Graph support. Use
`compile_mode="default"` for ordinary Inductor compilation or leave it unset for
eager native execution. Compilation is explicit because cold compilation and
graph capture can cost more than a small one-off readout.

The compiled callable accepts one device-resident parameter matrix and transfers
only columns active in the selected readout. Programs are cached by plan, batch
shape, precision, actual device, mode and compute-policy snapshot, with eight
instance-local entries. Changing values submits fresh inputs. Graph replay and
the host result snapshot share a per-entry lock so concurrent readers cannot
observe overwritten graph-owned outputs. `clear_cache()` releases prepared plans
and these compiled callables. Other admitted representations retain eager GPU
execution. Indexed CUDA devices and the caller's current device are preserved
when scheduler workers run concurrently.

CUDA Graph workspaces belong to the native compiler; admission does not cap its
allocator or retained graph pools. This path has compiler/value-isolation tests;
the physical CUDA checks require a real device and exercise eager, default and
reduce-overhead modes. CPU checks do not establish GPU throughput or a universal
speedup. See the official [Torch compiler modes](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
and [CUDA Graph constraints](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).

## Shared dispatch

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

There is no fixed 24-qubit ceiling. A complex128 pure-state vector alone needs
`16 * 2**n` bytes; the dense execution admission model reserves four such
buffers (`64 * 2**n` bytes per batch row). Thus 24 qubits require a 1 GiB
working budget, 25 require 2 GiB and 30 require 64 GiB, before framework reserves
and autodiff tapes. `ExecutionOptions.for_statevector(budget)` derives a width
cap from that policy. A general density matrix instead has `4**n` complex
entries. Large factored or stabilizer-compatible expectation queries can avoid
dense allocation; their width does not establish arbitrary-circuit capacity.

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

## Physical noise and error mitigation

Circuit IR supports `amplitude_damping(p,q)`, `phase_damping(p,q)`,
`depolarizing(p,q)`, `bit_flip(p,q)`, `phase_flip(p,q)` and
`channel(kraus,*qubits)`, as well as reset and classical feedback. Density
execution propagates these channels; it is not a binomial shot-noise model.
Custom Kraus operators must define a trace-preserving completely positive
channel. Channel parameters and local dimensions remain subject to admission
and differentiability contracts. Aer/Cirq integrations also accept their
provider-native noise models.

`ReadoutMitigation(assignment)` accepts `(n,2,2)` matrices in
`A[wire,measured,prepared]` order. Alternatively,
`ReadoutMitigation.from_counts(prepared_zero,prepared_one)` estimates marginal
assignment probabilities from all-zero/all-one preparations. Bit strings follow
OpenQML's q0-first convention. `.estimate(counts,observables)` corrects I/Z
Pauli strings and Hamiltonians directly, with no `2**n` distribution or inverse
assignment matrix. Rotate other measurement bases before collecting counts.

Calibration assumes independent readout and trusted preparation. Two global
preparations cannot identify correlated errors or distinguish readout from
preparation bias. Standard errors include covariance between Hamiltonian terms
measured in the same shots, conditioned on the calibration. They exclude
calibration uncertainty and drift. Singular/ill-conditioned calibrations,
excessive single-shot score amplification and workspace/work overruns fail
explicitly. `sampling_overhead` bounds squared scores relative to the
observable's coefficient L1 bound; it is not a measured device speedup.

`zero_noise_extrapolate(scales,values,order=1,standard_errors=None,covariance=None)`
fits a scaled Chebyshev polynomial and evaluates it at zero noise. `values`
has shape `(scales,...)`; independent errors match the scales or values shape.
Alternatively, pass a shared positive-definite `(scales,scales)` covariance
matrix. SVD conditioning, weight amplification, workspace and work budgets
limit unstable extrapolation. The result includes weights, residuals, condition
numbers and propagated statistical errors; errors are `None` when none were
supplied. `sampling_overhead` is the squared L1 norm of the extrapolation
weights, a unit-variance amplification bound rather than measured shot cost.
It does not infer model bias from fit residuals. Callers supply
physically scaled observations: this function neither folds gates nor submits
extra jobs. Higher polynomial order can amplify variance and model bias.
Neither mitigation method clips estimates to spectral bounds or guarantees an
accuracy improvement. The extrapolation approach follows
[Temme, Bravyi and Gambetta](https://arxiv.org/abs/1612.02058).

For IBM hardware, initialize an SDK primitive and its ISA pass manager before
passing them to OpenQML. Install `openqml[qiskit]` and `qiskit-ibm-runtime`
separately; credentials and backend access belong to your IBM account:

```python
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
import openqml as oq

service = QiskitRuntimeService()  # Uses your saved account configuration.
hardware = service.backend("YOUR_BACKEND_NAME")
primitive = EstimatorV2(mode=hardware, options={"resilience_level": 2})
backend = oq.QiskitEstimatorBackend(
    primitive,
    pass_manager=generate_preset_pass_manager(backend=hardware, optimization_level=1),
    mitigated=True,
)
# Submit deliberately with backend=backend in expectations/expectations_batch.
```

IBM's [noise-management documentation](https://quantum.cloud.ibm.com/docs/en/guides/estimator-noise-management)
describes resilience level 2 as TREX plus ZNE and gate twirling, with additional
sampling cost. `mitigated=True` explicitly allows finite statistical estimates
outside physical spectral bounds while retaining dtype and shape validation;
the default remains bounded. Job receipts preserve that contract during
recovery. This SDK integration is covered with local primitive/job tests,
not a claim of completed live-hardware or multi-QPU validation.

## Scoped configuration and job recovery

`QiskitEstimatorBackend.submit_expectations(bound_circuits, observables)` and
`QiskitSamplerBackend.submit_samples(bound_circuits, shots)` return a pending
handle immediately after one V2 submission. `handle.to_dict()` retains the
receipt before polling; `handle.result()` waits and validates the result. The
ordinary batch methods retain their synchronous behavior. Parameters must
already be bound, and optional `options=ExecutionOptions(...)` applies the same
input and qubit limits before submission.

The supplied primitive may use the caller's IBM Runtime Session or Batch.
The adapter adds no automatic job fan-out. Transpiled circuit caches and mapped
observable caches are instance-local and bounded; `clear_cache()` invalidates
both after a target or pass manager changes. Different parameter rows always
submit new work. Concurrent readers of a pending handle share its completed
decode, without repeating retrieval or submission.

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
