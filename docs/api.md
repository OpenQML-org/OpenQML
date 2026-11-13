# Python API

Use `import openqml as oq`. The root and financial namespaces resolve symbols on first
access and cache them. `dir(oq)` lists available names without importing their
implementations. Type stubs describe the same exports for editors. Frameworks
and SDKs are never installed at runtime.

## Common options

| Option | Meaning |
| --- | --- |
| `framework` | `torch`, `jax` or `flax`; model training integration |
| `backend` | Circuit execution backend or instance; `auto` uses configured cost/capability policies |
| `device` | Native tensor placement; defaults to the framework's available GPU/CPU policy |
| `method` | Algorithm choice where applicable; pricing accepts `auto`, `classical`, `quantum` |
| `max_memory_mb` | Estimated working-memory admission in MiB; excludes framework reserves and tapes |

A QPU executes compatible circuits. Classical projections, portfolio solvers
and optimizer state remain native tensor calculations. Recurrent density carries
require native execution; see the stateful-model guide for exporting unrolls.

## Input and failure contracts

Circuit entry points validate `Circuit` instances before backend dispatch.
Bindings require mappings; batch bindings require an iterable of mappings.
Model task/framework and conversion-format names require strings. Invalid
container types raise `ValidationError` rather than incidental attribute or
unhashable-container errors. Tensor subclasses retain their framework; a
framework's non-tensor Python objects are not treated as arrays.

| Failure | Exception |
| --- | --- |
| Invalid types, shapes, values or option names | `ValidationError` |
| Unsupported conversion semantics or format string | `ConversionError` with diagnostics |
| Optional SDK/framework import failure | `MissingDependencyError` with the integration and install command |
| Working-memory, qubit or arithmetic admission exceeded | `ResourceError` |
| Singular factorization or non-finite numerical primitive | `NumericalError` |
| Solver residual, allocation KKT or projection funding tolerance not met | `ConvergenceError`, a `NumericalError` subclass |

Checked `compile_compute` boundaries retain typed numerical failures. Raw native
JIT calls retain framework error behavior and may expose the internal NaN/error
status used by training checks. See [compute contracts](compute.md#numerical-contracts).

## Circuit conversion

```python
import openqml as oq

result = oq.convert(oq.Circuit(2).h(0).cx(0, 1), target="openqasm3")
print(result.value)
```

`convert(value,target=...,bindings=None)` returns a `ConversionResult`; `.value`
contains the target object or text. Core targets are `ir`, `json`, `openqasm2`
and `openqasm3`. Install the matching extra for SDK targets. Source detection,
compatibility checks and output budgets use the same conversion pipeline.
Use `plan_conversion` for repeated exports or partial binding; see
[interoperability](interoperability.md).

## Input validation

Expression `evaluate`, `bind` and `substitute` methods require mappings, including
for constant expressions. Derivatives require a valid parameter name. Malformed
expression fields and incompatible automatic-readout batch shapes raise
`ValidationError`; broadcast failures retain their original NumPy exception as
the cause.

Referenced expression bindings must contain real numeric scalars or NumPy,
Torch or JAX arrays. Boolean, complex, object and text values are rejected
before arithmetic. Native dtype/shape checks preserve tensors and gradients;
errors raised by a caller's mapping implementation remain caller errors.

## Package organization

The implementation is grouped by responsibility. The root API remains the
shortest entry point; published module import paths resolve lazily to the same
canonical implementations.

| Package | Responsibility |
| --- | --- |
| `ir` | Circuit construction, expressions, observables and compiler passes |
| `compute` | Shared native primitives, contractions, admission and metadata caches |
| `sim` | Dense/structured simulation, measurement, mitigation and execution plans |
| `autodiff` | Differentiation, checked compilation, geometry and optimization |
| `qml` | Hybrid architectures, layers, supervised training and framework interfaces |
| `interop` | Conversion, semantics, SDK adapters and provider primitives |
| `runtime` | Scheduling, backend capabilities, costs, configuration and job recovery |
| `finance` | Financial objectives, allocations, risk and chronological evaluation |

For example, `openqml.simulator` and `openqml.sim.dense` identify the same
module. Compatibility does not create a second set of registries or classes.

## Application scope

The public package covers hybrid AI, quantitative research and circuit
interoperability. RL/PPO and standalone tensor-network layers, compression and
plan APIs have been removed. Tensor contraction, compressed projections and MPS
execution remain private implementations of the shared compute service.
Imports and checkpoints belonging to the removed standalone models are no
longer supported. The optional contraction-search extra is `openqml[compute]`.

## Compiled circuits

`compile_circuit(circuit,observables=None,backend="jax",...)` returns a callable
accepting a parameter mapping. With observables it returns native expectation
arrays; otherwise it returns a pure statevector. Use `backend="torch"` for
TorchInductor or `backend="jax"` for XLA. Gate dispatch, evolution and readout
are traced together. Compilation is explicit and the first invocation includes
tracing/code generation.

Pass `transform=jax.jacrev` or `transform=torch.func.jacrev` to compile a
derivative before the checked result boundary. The returned mapping uses the
input parameter names. The callable captures its compute policy, validates
batch admission before conversion, and raises finite-result failures at the
host boundary. Shapes are specialized by default; shape/dtype/device changes
can trigger additional compilation under the native compiler's rules.
Compiling a Torch transform requires PyTorch 2.4 or later. On PyTorch 2.2/2.3,
compiled circuit values still support ordinary autograd; supplying `transform`
raises `ValidationError` without changing global compiler settings.
See [execution lowering](compute.md#circuit-construction-and-execution-lowering).

## Supervised models

`oq.Model(...)` constructs a configuration without importing a training framework.
`fit(x,y,epochs=...,batch_size=...)` builds and trains a native hybrid network,
returning the same object. `predict(x)` returns NumPy results. Dimensions can be
inferred on first fit or supplied as `input_features` and `output_features`.
Pass Python lists, NumPy arrays, or tensors belonging to the selected framework.
NumPy memory-mapped inputs are validated in bounded chunks. Use
`iter_predict(x,batch_size=256,output="prediction")` to consume NumPy prediction
batches without collecting a complete output array. Classification also accepts
`output="probabilities"`; forecasting accepts `output="quantiles"`.

| Task | Targets | Prediction |
| --- | --- | --- |
| `regression` | `[B]` or output-shaped floating arrays | Scalar/vector predictions |
| `classification` | Integer class IDs on non-class output axes | Class IDs; `predict_proba` returns normalized probabilities |
| `forecast` | Next-return labels `[B,T,N]` for causal features `[B,T,F]` | Asset medians; `predict_quantiles` returns ordered `[B,T,N,Q]` quantiles |

Specify `output_features=1` for scalar sequence regression with targets `[B,T]`.
Forecast medians describe the 0.5 quantile. Construct financial features and
labels causally, fit preprocessing on training observations and supply a
chronologically separate validation set when selecting a model.

Supply observations and Hamiltonians from sources appropriate to the task;
datasets and molecular operators are external to the package. Large-workload
coverage and accuracy require validation on that workload. See
[external data](hybrid-training.md#external-data-and-bounded-prediction).

`num_qubits` and `quantum_depth` set each quantum cell's width and depth. A
forecast cell reserves one environment qubit and uses the remaining qubits as
recurrent memory, so it requires at least two qubits. Execution is admitted
against backend resources; there is no fixed maximum qubit count.

`framework="torch"` is the default. Install `openqml[torch]` for it;
`openqml[jax]` includes JAX and Optax, and `openqml[flax]` adds Flax. Input data
are converted in batches on the selected device. Training uses float32; use the
native interfaces for other precision policies. Prediction materializes NumPy
output; call the native module for differentiable inference.

Repeated `fit` calls preserve the native parameter tree and optimizer state.
`history` records each epoch's mean loss and accepted updates. The same
`microbatch_size` option works across all frameworks; changing it invalidates
only the relevant compiled update. Shuffling is off by default.

`fit(...,validation_data=(x_val,y_val),patience=5,min_delta=1e-5)` evaluates
held-out loss after each epoch. `patience` counts epochs without a reduction
greater than `min_delta`. With `restore_best=True` (the default), early stopping
restores parameters, optimizer, update counters and shuffle/framework RNG to
the best validation epoch, including when the epoch budget ends first.
`restore_best=False` retains the final training state. With validation data and
`patience=None`, training runs every requested epoch and records validation loss.
`history` retains all attempted epochs with optional `val_loss`; `best_epoch`
and `stopped_epoch` identify 1-based positions in that history for the latest fit.
Each fit starts a new monitoring window while preserving optimizer ownership.

`model.evaluate(x,y,batch_size=256)` returns the mean task loss in bounded batches.
It preserves the Torch module's training/evaluation mode and optimizer state.
Early stopping keeps one full training snapshot on the model's native device;
include that additional state in the training memory budget.

`model.save("model.oqm")` writes one atomic archive. Restore it with
`oq.Model.load("model.oqm",device="cpu",backend="auto")`; prediction works
immediately, and subsequent `fit` resumes the optimizer and shuffle sequence.
The archive includes the task, architecture, training state and history. Torch
state is stored as a JSON tree plus NumPy tensors; JAX/Flax retain their native
NPZ format. Loading uses no pickle object deserialization.
`max_bytes` bounds the uncompressed archive size on load (256 MiB by default).
Loading checks unique NPZ member names, bounded NPY headers, object-free dtypes
and declared tensor sizes before constructing the model or allocating tensor
payloads. Forecast configuration must match the admitted saved architecture
before native reconstruction; depth admission precedes operator expansion. JSON metadata rejects duplicate fields and oversized input.
The training framework is retained; device and execution backend can be selected
again. Runtime/provider objects and credentials are not serialized. Custom
optimizers and distributed meshes should use their native checkpoint interfaces.

Supply `architecture=oq.HybridArchitecture(...)` for custom array-based
regression/classification. `model.module` and `model.trainer` expose native
forward calls, custom objectives, checkpointing and stateful execution. There
is no second optimizer or copied trainable parameter owner. Native APIs remain
appropriate for graph input dictionaries, explicit carries and specialized
training procedures.

## Financial calculations

`oq.finance.allocate(returns,...)` accepts historical/scenario arithmetic return
arrays `[...,S,N]` and returns the existing native `AllocationResult`. Common
policy arguments are `risk_aversion`, `turnover`, `ridge` and `tolerance`;
pass an `AllocationPlan` for full policy control. Use one configuration style
per call. A supplied `FactorRisk` replaces return observations.

Python/NumPy constants follow any supplied PyTorch or JAX tensor's device and
dtype, so a Python probability list does not detach a differentiable return or
payoff tensor. Integer probability arrays are promoted to floating point.
PyTorch/JAX mixtures and mixed PyTorch devices require explicit conversion.

`previous` contains economic holdings; `warm_start` is only an optimizer seed.
The latter is projected, used to propose an active face, then checked against
current feasibility and KKT residuals. A valid face skips the iterative primal
search. A failed certificate triggers the ordinary solver. Gradients still come
from current implicit equations, with no derivative through the seed. Eager
NumPy/PyTorch and JIT-compiled JAX support this shortcut; compiled PyTorch retains
the ordinary path. Work budgets conservatively admit the fallback.

`oq.finance.price(payoffs,probabilities=...)` performs exact native weighted
summation, supports arbitrary scenario counts, and returns `PayoffEstimate`.
Its value and probability gradients remain on the input array backend. It does
not load a quantum execution engine or use a sampler.

With coherent `preparation` plus bounded `payoffs`, or an `AmplitudeOracle` plus
`bounds`, `method="auto"` first attempts an admitted exact classical calculation.
It reuses deterministic state-vector or exact expectation backends; a payoff
table is summed under the preparation's distribution without synthesizing a
Grover circuit. If no exact path fits, amplified sampling is considered.
`method="classical"` requires an exact path; `method="quantum"` explicitly
selects sampled estimation, including when using a simulator.

The default quantum path uses `AdaptiveAmplitudeEstimationPlan`, translating
the requested absolute monetary `error` through payoff range and discount.
`strict=True` raises if the budget cannot deliver that bound. `strict=False`
returns the partial result and its amplitude diagnostics. A supplied plan can
configure depth costs, per-batch overhead, contrast, rounds and shot batching.
See [finance](finance.md) for mathematical assumptions and method sources.

Pricing requires an explicitly supplied probability measure. Statistical
intervals cover sampling uncertainty; assess floating-point, discretization and
market-model errors separately. Resource profiles account for oracle loading,
circuit execution and the selected backend's scheduling costs.

## Rolling quantitative research

`oq.finance.walk_forward(factory,x,y,times=...,label_end=...,splitter=...)`
refits a fresh `Model` per purged training fold and aligns test predictions with
the original observation axis. `validation_size` reserves a training-tail
validation set and purges its overlapping training labels. `fit_options` and
`predict_options` configure native fitting and prediction. See
[quantitative research](finance.md#rolling-hybrid-model-selection).

## Serialization and conversion limits

`Circuit.to_json(indent=2,max_bytes=2_000_000)` bounds the serialized UTF-8
artifact. Expression expansion is checked before nested dictionaries are built.
`plan_conversion(...,max_output_bytes=2_000_000)` applies the same output
budget to JSON and OpenQASM targets, retaining it through bindings and sweeps.
The CLI exposes this control as `--max-output-bytes`.

## Parameter-sweep conversion

`plan.bind(mapping,partial=False)` returns a plan with bound source parameters.
`plan.iter_exports(rows,max_bindings=1024)` streams checked artifacts;
`plan.export_many(rows,...)` returns a tuple. Compatible symbolic targets reuse
prepared topology. See [interoperability](interoperability.md).

## Risk-budget projection

`oq.finance.risk_budget_projection(weights,scenarios,probabilities=None,confidence=0.95,`
`budget=0.02,radius=0,cash_return=0,max_memory_mb=256)` returns a
`RiskBudgetResult` named tuple. It scales a funded long-only proposal toward
cash to meet weighted CVaR plus a W1/L2 margin. Native batched radii and budgets
retain PyTorch/JAX derivatives. The input proposal must sum to at most one and
the budget must admit cash. See [risk budgets](finance.md#differentiable-decision-risk-budgets).

Native JAX/Flax network trainers accept
`trainer.load(path,max_bytes=256*2**20)`. Tensor archive headers and expanded sizes
are admitted before NumPy loading; duplicate JSON metadata fields are rejected.
`Model.load(...,max_bytes=...)` forwards the selected budget to its native loader.
