# Python API

Use `import openqml as oq`. Public symbols retain their original identities and
serialization paths. The root and financial namespaces resolve symbols on first
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

## Supervised models

`oq.Model(...)` constructs a configuration without importing a training framework.
`fit(x,y,epochs=...,batch_size=...)` builds and trains a native hybrid network,
returning the same object. `predict(x)` returns NumPy results. Dimensions can be
inferred on first fit or supplied as `input_features` and `output_features`.
Pass Python lists, NumPy arrays, or tensors belonging to the selected framework.

| Task | Targets | Prediction |
| --- | --- | --- |
| `regression` | `[B]` or output-shaped floating arrays | Scalar/vector predictions |
| `classification` | Integer class IDs on non-class output axes | Class IDs; `predict_proba` returns normalized probabilities |
| `forecast` | Next-return labels `[B,T,N]` for causal features `[B,T,F]` | Asset medians; `predict_quantiles` returns ordered `[B,T,N,Q]` quantiles |

Specify `output_features=1` for scalar sequence regression with targets `[B,T]`.
Forecast medians are not mean-return estimates. Construct financial features and
labels causally and fit on training observations only; the interface does not
create a validation split or fit preprocessing on future data.

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

`model.save("model.oqm")` writes one atomic archive. Restore it with
`oq.Model.load("model.oqm",device="cpu",backend="auto")`; prediction works
immediately, and subsequent `fit` resumes the optimizer and shuffle sequence.
The archive includes the task, architecture, training state and history. Torch
state is stored as a JSON tree plus NumPy tensors; JAX/Flax retain their native
NPZ format. Loading uses no pickle object deserialization.
`max_bytes` bounds the uncompressed archive size on load (256 MiB by default).
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

All exact calculations exclude floating-point/model errors from their statistical
intervals. Fair pricing still requires the appropriate supplied probability
measure. Neither numerical convergence nor a small sampling interval establishes
financial alpha or an end-to-end quantum speedup.
