# Stateful and generative models

`HybridArchitecture` composes convolution, state attention, ordered-pair graphs,
quantum KAN, recurrent memory and Born readouts across PyTorch, JAX and Flax.
This is a supported set, not an exhaustive implementation of quantum deep learning.

| Model | Entry point | Quantum computation |
| --- | --- | --- |
| Quantum KAN | `HybridArchitecture.kan(input_features, output_features)` | Univariate reuploading activations with trainable frequencies and phases |
| Recurrent network | `HybridArchitecture.recurrent(input_features, output_features)` | Input-conditioned CPTP evolution of an explicit density carry |
| Conditional Born generator | `HybridArchitecture.generative(input_features, visible_qubits)` | Complete visible-register Born probabilities |
| Autoregressive sequence model | `HybridArchitecture.language_model(vocab_qubits)` | Dissipative memory and trainable multi-outcome POVM |
| Quantum-state diffusion | `QuantumDiffusionPlan` and `ChannelLayer` | Depolarizing forward process and shared time-conditioned reverse channel |

## Train a sequence model

```python
import torch
import openqml as oq

architecture = oq.HybridArchitecture.language_model(
    vocab_qubits=8,
    memory_qubits=2,
    environment_qubits=1,
    hidden_features=16,
    depth=2,
)
model = architecture.build("torch")
device = next(model.parameters()).device
tokens = torch.randint(0, 256, (4, 12), device=device)
features = oq.shifted_tokens(tokens, 256, backend="torch", like=model.values["stem_weight"])
trainer = model.trainer(lr=1e-3, microbatch_size=2)
metrics = trainer.step(
    (features, tokens),
    lambda network, batch: oq.born_nll(network(batch[0]), batch[1], backend="torch"),
)
generated, carry = model.sample_tokens(16, prompt=tokens[:1], seed=7)
```

`vocab_qubits=k` selects `2**k` symbols. The default POVM permits more symbols
than memory basis states; more outcomes do not increase accessible memory
information. `measurement="basis"` requires `memory_qubits >= k`.
Teacher forcing uses zero BOS features followed by previous tokens, without
target leakage. Pass `mask[B,T]` to `shifted_tokens`, network inputs
`{"x": features, "mask": mask}` and `born_nll`. Masked IDs may use `-1`.
`reset_mask[B,T]` resets active segment starts and their BOS features.

JAX/Flax trainer losses must reduce token losses to one value per example:
`born_nll(p, ids, backend="jax", reduction="none").mean(axis=-1)`.
Existing gradient accumulation, optimizer groups, depth schedules and
checkpoints apply. Generative/language constructor `depth` counts prefix
blocks; the allocated architecture adds one terminal probability head. That
head remains active during prefix curricula, including active depth zero.

## Explicit recurrent state

```python
network = oq.HybridArchitecture.recurrent(8, 2, depth=3).build("torch")
output, state = network(features, return_state=True)
next_output, state = network(next_features, state=state, return_state=True)
state = oq.detach_quantum_state(state)  # optional caller-selected BPTT boundary
```

Here features have shape `[B,T,8]` on the model device. State maps block names
to complex `[B,2**m,2**m]` densities. Passing it between chunks preserves
gradients. Independent batches use `state=None`; models store no hidden carry.
Padding holds state and emits zeros; resets cut dependence on previous carry.
Computed states are symmetrized and trace-normalized to remove floating-point
drift, without eigenvalue projection or rank truncation. Invalid incoming
densities are rejected rather than repaired.

Each step appends a fresh pure environment, applies a shared SU(4)-based
input-conditioned circuit, then traces out the environment. Optional gating
uses an input-conditioned convex mixture with the identity channel. This
CPTP construction supports learned writing and forgetting through a quantum
channel. Ensemble readouts preserve the recurrent density carry.

Use `scan="parallel"` on recurrent/language constructors, or `method="parallel"`
on `QuantumChannelPlan.scan`, for associative channel composition. It preserves
values and gradients, including gates, padding and resets. JAX uses
`lax.associative_scan`; Torch uses batched prefix products. Logarithmic
composition depth costs extra work and `O(B*T*16**m)` map storage. `max_work`
and `max_memory_mb` reject oversized maps before allocation. Sequential density
evolution remains the default. Select a scan mode using the target device's
sequence-length, memory and execution-time measurements.

Streaming generation prefills once, then updates each stage once per new token.
JAX uses an explicit key and compiled scan. Supported prefixes are memory,
pointwise and quantum KAN. Stateless convolution/attention require their own
context and cannot substitute a density carry for it.

## Measurements, activations and diffusion

`QuantumPOVMPlan` constructs positive effects summing to identity by regularized
Cholesky whitening of a complex operator frame with complete anchor effects.
It avoids eigenvector derivatives at degenerate frames. Probabilities are
`Tr(E_y*rho)` with optional physical mixing with the uniform distribution.
There is no classical vocabulary projection or softmax after measurement.
Frames are prepared per forward call inside its gradient graph.

Static Born heads recover visible probabilities from the complete ordered Z
basis using a fast Walsh transform. They allocate no unused classical output
layer or residual feed-forward head. KAN evaluates a shared univariate quantum
basis per input coordinate, rather than a circuit for every coordinate pair.
Sharing restricts the independent-edge KAN family. Default single-qubit
RX/RY/RZ activations are efficiently classically simulable and provide a
trainable periodic basis as an inductive bias.

Quantum-state diffusion uses `ChannelLayer(QuantumChannelPlan(...))` as a
shared time-conditioned denoiser. In Torch it implements
`layer(rho, layer.time_controls(t))`; JAX/Flax use explicit parameters and
`plan.time_controls(t, backend="jax", like=weights)`.
`process.loss(denoiser, rho0, integer_times, backend=..., like=weights)` returns
per-example Hilbert-Schmidt error against the preceding forward density state.
`process.sample(denoiser, batch_size, ...)` starts at the maximally mixed state.
The forward process depolarizes; pure environments allow reverse entropy
removal. The objective is an implementation adaptation, without transferring
paper benchmark results. Generated objects are quantum densities, not images.

## Execution and method sources

Explicit density carries use native framework GPU/CPU execution. The planner
excludes QPUs before availability checks and cost pruning because its numeric
QPU interface cannot transfer arbitrary carries. `plan.unroll(weights, controls)`
exports a bound circuit starting from zero with physical environment resets.
This covers the final ungated channel sequence, not incoming density states,
noninvasive intermediate readouts, convex gates or learned POVM export.

Density storage remains exponential. Budgets cover primal buffers/descriptors,
excluding autodiff tapes, allocator reserves and communication. Explicit
carries do not support the distributed tensor mesh adapter. There is no fixed
qubit ceiling; requested resources determine admission. Continuous-variable
models, fault-tolerant constructions and fully coherent quantum attention
require dedicated execution primitives beyond these discrete circuit families.

- [Quantum Variational Activation Functions Empower Kolmogorov-Arnold Networks](https://arxiv.org/abs/2509.14026), 2025 preprint: trainable univariate activations.
- [Trainable Quantum-Native Language Models](https://arxiv.org/abs/2610.12806), 2026 preprint: dissipative memory and direct POVM readout, with explicit capacity and classical-control limitations.
- [Quantum Generative Diffusion Model](https://arxiv.org/abs/2401.07039): depolarizing trajectories and Stinespring reverse channels for mixed-state generation.

Select model families using task-specific held-out metrics and the execution
costs of the chosen representation. Each implementation's parameter ownership,
state carry and resource controls are described above.
