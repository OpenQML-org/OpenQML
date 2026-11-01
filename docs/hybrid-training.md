# Native hybrid quantum-classical training

OpenQML provides ordinary trainable Torch modules, functional JAX parameter
trees and Flax Linen modules. Classical encoders, emitted quantum coordinates,
measurement axes and residual projections remain in one native computation
graph. No pretrained model, dataset or demonstration program is bundled.

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
The implemented scope is the exact local readout family and native training
interface; no task-level quantum advantage is established.

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
native tensors/arrays. Exhausting the budget never becomes a convergence claim.
Autodiff retention and parameter/coordinate buffers still consume memory.

The state metric follows [Quantum Natural Gradient, Quantum 4, 269
(2020)](https://arxiv.org/abs/1909.02108); the constant IQP covariance and native
encoder pullback are the specialized implementation here, not an assertion
that the HyperQ paper implements this optimizer. Loss convergence and superiority
over Adam/SGD are not guaranteed.

## Initialization and general noisy layers

Paired initialization follows [Grant et al., Quantum 3, 214
(2019)](https://arxiv.org/abs/1903.05076). Every block has separate forward and
reverse parameters initialized to opposite values. Its variational action
therefore starts at identity while encoding remains active. Local RY/RZ and
trainable RZZ gates retain an entangling family after parameter updates.
[Identity-Paired Progressive Depth Training](https://arxiv.org/html/2607.16800v1)
(July 2026) identifies permanent expressivity saturation from canceling fixed
CNOT rings. OpenQML avoids that construction; it does not claim to implement
the paper's complete progressive training curriculum or eliminate all plateaus.
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
CUDA tests execute only when hardware is available. CPU tests do not establish
GPU throughput or the source papers' downstream language-model results.
