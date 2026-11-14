# Native hybrid quantum-classical training

OpenQML provides ordinary trainable Torch modules, functional JAX parameter
trees and Flax Linen modules. Classical encoders, emitted quantum coordinates,
measurement axes and residual projections remain in one native computation
graph. No pretrained model, dataset or demonstration program is bundled.

## External data and bounded prediction

`Model.fit` and `evaluate` accept NumPy arrays and disk-backed `np.memmap`
views. Finite-value and floating classification-label checks use bounded
temporary buffers (at most 8 MiB, reduced by the shared compute ceiling).
They still scan all observations. `shuffle=False` keeps contiguous batch
slices; `shuffle=True` allocates a sample permutation and gathers each batch.
The model, optimizer, framework allocator and activations have separate costs.
This interface does not make a dense input representation sparse.

`model.iter_predict(inputs,batch_size=256)` yields NumPy batches so callers can
write or reduce predictions incrementally. Use `output="probabilities"` for
classification or `output="quantiles"` for forecasts. A one-dimensional input
keeps a length-one batch axis. Each batch is validated when consumed, so a
later invalid batch can raise after earlier results have been used. Kernel
scopes, Torch training mode and gradient mode are restored before yielding.
Inputs and weights must remain stable while the iterator is active; concurrent
training on the same model is not supported by this interface.

No dataset catalogue, synthetic chemistry substitute or industrial throughput
claim accompanies these APIs. Supply real observations with known provenance,
units and preprocessing. Chronological splits, coverage, dimensionality and
accuracy must be assessed on the intended workload.

| API | Training contract |
| --- | --- |
| `oq.torch.IQPQuantumLayer` / `oq.jax.IQPQuantumLayer` | Trainable local quantum readouts, with leading input/weight dimensions broadcast |
| `oq.torch.HyperQuantumAdapter` / `oq.jax.HyperQuantumAdapter` | Per-input circuit hypernetwork, low-rank encoder and residual output preserving `(..., features)` |
| `oq.flax.IQPQuantumLayer` / `oq.flax.HyperQuantumAdapter` | Linen parameter ownership and standard Optax-compatible variable trees |
| `QuantumLayer.identity_blocks(...)` | Paired forward/inverse initialization with independently trainable RZZ entanglers |
| `HyperQuantumAdapter.metric_product(...)` | Exact prepared-state metric pullback on encoder coordinates, using native JVP/VJP |
| `HyperQuantumAdapter.natural_gradient(...)` | Damped, matrix-free encoder solve with a freshly checked residual |
| Existing `QuantumLayer(..., execution_method="auto")` | Exact factor execution when dense admission fails and the circuit's quantum/classical components fit |

## Circuit hypernetworks and exact sparse readouts

The architectural basis is [Wang et al., *Circuit hypernetworks for
quantum-augmented diffusion language models*](https://arxiv.org/html/2609.24657v1),
a September 2026 preprint. OpenQML independently implements its two-body IQP
readout family and adapts the residual branch to general feature tensors.
The implementation combines exact local readouts, native automatic
differentiation and trainable residual branches over general feature tensors.

The prepared circuit is
`C = H^n exp[-i/2 (sum(alpha_q Z_q) + sum(omega_e Z_i Z_j))] H^n`.
For a local axis `psi=(sin(polar)cos(azimuth), sin(polar)sin(azimuth), cos(polar))`,
the exact zero-state expectation is

`z_q = (psi_z cos(alpha_q) - psi_y sin(alpha_q)) product_{e incident q} cos(omega_e)`.

`IQPGraph` stores a simple graph: duplicate/reversed duplicate edges and self
loops are rejected. The default ring/chord graph has degree at most four.
Degree buckets gather each incidence once, without a padded `n * max_degree`
matrix. Both graph construction and readout work scale with vertices/edges;
state vectors, shot sampling and coefficient truncation are absent. Numerical
roundoff remains. The family excludes noncommuting interiors, higher-body
generators and general output distributions. `graph.circuit()` exports the
prepared state as ordinary IR; readout axes remain an explicit output contract.

The hypernetwork maps an input through a rank-limited code, emits bounded
single-site/coupling angles and spherical measurement axes, evaluates the exact
readouts, then projects them into a learned residual. Residual scaling starts
small and nonzero, allowing encoder/quantum-coordinate gradients on the first
step. A frozen classical backbone is caller controlled. Native Adam/SGD or
Optax can update the complete hybrid model without a second training runtime.

Under the paper's independent uniform-angle and uniform-sphere reference
initialization, coordinate-gradient variance is `1/(3 * 2**degree(q))`.
`graph.to_dict()` labels these assumptions explicitly. This is not a theorem
about an arbitrary task loss, a learned hypernetwork distribution or the whole
training trajectory. Dense connectivity can still suppress gradients.

## Prepared-state geometry without quantum-state buffers

The IQP generators are distinct commuting X/XX Pauli words in the register
basis. In the zero input state, their means and distinct pair covariances vanish.
The real Fubini-Study metric on `(alpha, omega)` is consequently `I/4` for all
coordinates. If the encoder emits these coordinates with Jacobian `J_b` on
input row `b`, its exact pullback is

`G_encoder = sum_b J_b.T J_b / (4 * B)`.

Native JVP followed by VJP applies this matrix without materializing it. Torch
vectors use the names/shapes of `adapter.encoder.named_parameters()`. JAX
vectors use the keys/shapes of `parameters["encoder"]`. Measurement-axis,
residual-projection and scaling parameters are outside this prepared-state
geometry. This is not the Fisher metric of the entire classical task model.
Ordinary joint optimizers remain available independently.

`natural_gradient` solves `(G + damping I) d = gradient`. The damping,
tolerance and iteration budget are static controls. All budgeted products are
executed using convergence masks; JAX uses a static `lax.fori_loop` to avoid
unrolling a separate trace per iteration. There is no per-iteration host/device
synchronization or hidden dense inverse. The returned direction, convergence
flag, logical iteration count, actual product count and recomputed residual are
native tensors/arrays. The convergence flag reports whether the requested
residual tolerance was reached within the iteration budget.
Autodiff retention and parameter/coordinate buffers still consume memory.

The state metric follows [Quantum Natural Gradient, Quantum 4, 269
(2020)](https://arxiv.org/abs/1909.02108). OpenQML specializes it to constant IQP
covariance and a native encoder pullback. Select damping and compare held-out
loss under the same training budget when choosing between this optimizer and
Adam or SGD.

## Initialization and general noisy layers

Paired initialization follows [Grant et al., Quantum 3, 214
(2019)](https://arxiv.org/abs/1903.05076). Every block has separate forward and
reverse parameters initialized to opposite values. Its variational action
therefore starts at identity while encoding remains active. Local RY/RZ and
trainable RZZ gates retain an entangling family after parameter updates.
[Identity-Paired Progressive Depth Training](https://arxiv.org/html/2607.16800v1)
(July 2026) identifies permanent expressivity saturation from canceling fixed
CNOT rings. OpenQML uses independently trainable RZZ blocks and explicit depth
schedules to retain active entanglers as training progresses.
Initialization recipes are applied by the named constructors; checkpoints
preserve actual learned values in the existing source-parameter order.

General `QuantumLayer` simulation retains the existing gate/noise/control IR.
If dense admission fails in automatic mode, native component execution joins
quantum operands, classical writers/readers and both feedback arms, then
executes admitted exact factors. It supports Torch/JAX higher derivatives and
batched bindings. Explicit limits remain enforced. Checkpointed dense execution
does not silently change algorithms. A large connected factor can still exceed
memory; this path does not make arbitrary dense simulation polynomial.

## Framework and verification boundaries

The modules preserve arbitrary leading batch/token dimensions, parameter device
placement and native differentiation. IQP readout indices are static device
buffers in Torch. Low-precision Torch inputs use float32 trigonometric
evaluation; float64 is retained when requested. JAX float64 initialization
requires enabled x64. Chunking bounds the visible per-call readout workspace,
not total process RSS or retained backward activations. Model/optimizer weights
have their own costs.

`tests/test_hybrid_training.py` checks small independent dense references,
gradients/Hessians including zero cosine factors, large structured readouts
without dense allocation, ordinary joint training, native metric products and
damped solves, initialization identities, feedback derivatives, checkpoint
topology, chunk equivalence, Torch compile/func transforms and JAX JIT/vmap/Optax.
CUDA tests execute when hardware is available. Measure throughput on the target
device with the intended architecture, precision, batch size and execution path.

## Joint AdamW and Quantum Natural Muon

`hybrid_qnm` owns a complete Torch module. Classical parameters use AdamW;
active weights of native `QuantumLayer` modules use the stateful
distance-aware quantum Fisher method. Unitary circuits retain pure-state
geometry; noisy or dynamic circuits use regularized final-density SLD geometry.
`fisher_options={"state_damping": 1e-4, "max_branches": 256}` configures that
path. See [density-state geometry](native-geometry.md#density-state-fisher-geometry)
for its marginal-state convention and resource costs. After constructing a model
with a quantum submodule named `quantum` containing four active weights:

```python
import openqml as oq
from openqml.qml.interfaces.torch import HybridTrainer

optimizer = oq.hybrid_qnm(
    model,
    lr=1e-3,
    quantum_lr=0.02,
    block_shapes={"quantum": ((2, 2),)},
    quantum_options={"damping": 0.05, "momentum": 0.9},
)
trainer = HybridTrainer(model, optimizer, microbatch_size=16)


def objective(network, batch):
    features, targets = batch
    return (network(features) - targets).square().mean()


trainer.step((features, targets), objective)
trainer.save("training.pt")
optimizer.close()
```

Only layer calls participating in backward contribute geometry. Unused forward
passes and inference do not enter the empirical metric. Feature snapshots span
all participating calls and microbatches until `step` or `zero_grad`; their
memory is admitted against layer and shared compute budgets. Do not mutate
inputs between forward and backward. This metric uses uniform mass per captured
feature row, independently of loss weighting; repeated uses contribute repeated
rows. Classical layers influence the loss gradient but their full cross-block
Fisher information is not computed.

For `DistributedDataParallel`, construct the optimizer with the wrapped model.
Block-shape keys follow its module names, such as `module.quantum`. Metric-vector
products are reduced with each rank's actual feature-row count. The same global
Fisher system is solved on each rank, without gathering a dense metric.
`HybridTrainer` also weights the objective by global sample count, including
unequal rank sizes. Ranks must execute the same optimizer groups and collectives.
This path uses simulator state derivatives, not finite-shot hardware geometry.

Quantum master weights must use float32 or float64. `HybridTrainer` can use AMP
for the surrounding model, and its existing finite-gradient checks apply before
either optimizer updates. Quantum groups receive no AdamW weight decay; each
group's `lr` supports ordinary schedulers and zero-rate freezing. Spectral shapes
must cover active circuit coordinates in `weight_parameters` order. With no
shape mapping, each quantum vector uses a single-row block, retaining natural
momentum without nontrivial polar redistribution.

`trainer.state_dict()` and its save/load helpers retain classical moments,
quantum momentum, initial coordinates, trajectory distance, counters, scheduler
and random state. Restore into the same circuit and optimizer configuration;
identity is checked before trainer model mutation. Call `optimizer.close()`
when replacing it to remove its hooks. Capturable, differentiable, fused and
maximizing optimizer steps are outside this integration.

## Validation and early stopping

The high-level `Model` interface provides a held-out loss monitor for Torch,
JAX and Flax:

```python
import openqml as oq

x = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]]
y = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
model = oq.Model(hidden_features=8, depth=2)
model.fit(x[:4], y[:4], validation_data=(x[4:], y[4:]), epochs=50, patience=5)
print(model.best_epoch, model.evaluate(x[4:], y[4:]))
```

`min_delta` sets the required loss reduction. `restore_best=True` restores the
complete selected training state when `patience` is set: parameters, optimizer,
counters and random state. All attempted epochs remain in `history`; each
validation entry includes `val_loss`. Checkpoints retain history and selection
positions. Repeated `fit` calls resume the selected optimizer state and start a
new monitoring window. Financial data can use the nested chronological
validation in [`finance.walk_forward`](finance.md#rolling-hybrid-model-selection).

## Wide-range optimization

Training norms and quantum/classical gradient diagnostics retain finite values
without first squaring unscaled gradients. JAX uses native scaled reductions;
CPU Torch float32 leaves use float64 accumulation, and other Torch leaves use
scaled reductions. Torch clipping groups gradient tensors by dtype for native
foreach updates. Finite large gradients can be clipped and applied, while
nonfinite losses or gradients still reject the optimizer step.

Native natural-gradient CG normalizes its right-hand side, retains the existing
absolute/relative stopping contract, and reports the fresh residual in original
gradient units. The method uses matrix products without materializing a dense
metric or moving arrays to the host. The scaled sum-of-squares construction is
consistent with [LAPACK safe scaling](https://www.netlib.org/lapack/explore-html/d8/d76/group__lassq.html).
