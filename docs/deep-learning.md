# Quantum deep learning

`QuantumDeepNetwork(input_size, output_size, num_qubits=..., depth=...)` combines
a trainable classical encoder, deep quantum residual channels and a readout head.
PyTorch accepts `model(inputs)`; JAX uses `model.init(key)` and
`model(parameters, inputs)`; Flax uses Linen `init`/`apply`. Outputs are logits
or features; supply the task's loss and an ordinary optimizer.

For an existing model, use `QuantumLayer.deep(num_qubits, depth=...)` under
`oq.torch`, `oq.jax` or `oq.flax`. It accepts `(..., num_qubits)` controls and
returns the same number of local quantum readouts. It also works as the feature
layer of the existing quantum actor-critic and PPO interfaces.

## Quantum state and trainability

Each block acts on the persistent quantum data state:

`rho_next = (1-p) * rho + p * U @ rho @ U†`, with `p = sin(alpha/2)²`.

The data register is not measured and re-encoded between blocks. Controlled
RY/RZ/RZZ transformations implement the variational branch. One ancillary qubit
is reset and reused after each block, implementing a trace-preserving mixture
without postselection. A data register of `n` qubits therefore uses `n+1`
physical qubits. QPU execution requires native reset and the lowered controlled
gate capabilities; unsupported hardware fails admission.

Forward/reverse angles are independent trainable parameters initialized as
inverse pairs. Blocks initially preserve their input state; the optimizer can
then learn entangling transformations. Optional recurrent data embeddings
conjugate each block, preserving this identity property. Local Z readouts and
small nonzero initial mixing provide practical initialization choices.
These choices do not prove that arbitrary depths, losses or data distributions
avoid barren plateaus or outperform classical networks.

`DeepQuantumPlan` supplies the portable circuit and initialization metadata.
`plan.grow(checkpoint, add_blocks=..., seed=...)` returns a deeper plan and a
checkpoint: existing named weights are copied, and added blocks start at the
identity. Initial readouts are preserved within floating-point roundoff.
Framework optimizer resizing remains a caller operation; this portable
checkpoint does not silently modify optimizer state.

## Training and resources

Native Torch/JAX execution differentiates through the channel and classical
encoder/readout. Use `HybridTrainer` or `make_train_step` for microbatch gradient
accumulation. Flax parameters use a normal Linen tree. External QPU derivatives
use the existing first-order parameter-shift bridge and the cost scheduler
counts the controlled-gate shift recipes.

Exact mixed-state simulation can still cost exponentially in entangled width.
Depth, circuit construction and memory are admitted separately; raising a
budget does not establish scalable trainability. Neither distributed unitary
contraction nor approximate pure-state MPS is silently substituted for these
mixed-state residual channels. The core installation gains no new dependency.

The channel construction follows [postselection-free quantum residual
networks, 2026](https://arxiv.org/abs/2604.06866). Identity growth uses the
continuation principle studied in [identity-paired progressive depth training,
2026](https://arxiv.org/abs/2607.16800). These are research preprints; their
theoretical assumptions and reported results are not library-wide guarantees.
