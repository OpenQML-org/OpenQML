# Training and execution

## Quantum capacity

`oq.MultiscalePlan(num_qubits, bond_qubits=1, depth=2, num_outputs=3)` builds
a unitary multiscale circuit with nearest-neighbour SU(2)/Cartan rotations,
disentanglers between pooling regions, and repeated input encoding.
Unobserved wires undergo an implicit partial trace; no outcomes are selected.
`to_spec()` exports ordinary circuit IR and Pauli readouts.

`oq.torch.QuantumLayer.multiscale(...)`, `oq.jax.QuantumLayer.multiscale(...)`
and `oq.flax.QuantumLayer.multiscale(...)` construct trainable layers from this
plan. Increase `bond_qubits` to retain a larger coarse block, or `depth` to add
variational layers. `num_outputs` selects distinct nonidentity Pauli readouts
on the retained wires, up to their complete Pauli basis.

With `bond_qubits >= num_qubits`, pooling disappears. The resulting connected
rotation circuit can grow in depth without the binary tree's fixed retained
space. This removes that architectural restriction; it does not guarantee easy
optimization or efficient classical execution for every weight configuration.
MERA-style disentanglers also invalidate independent-node tree evaluation.

For native differentiation, select `backend="torch"` or `backend="jax"` and
`execution_method="tensor-network"`. `tensor_options={"rematerialize": True}`
recomputes contractions during backward instead of retaining all intermediate
activations. PyTorch uses non-reentrant checkpointing; JAX uses `jax.checkpoint`.
This option is deterministic circuit execution, not a replay of sampled QPU jobs.
PyTorch checkpointing follows the framework's restrictions on `torch.func`.

## Controlled approximation

`oq.AdaptiveMPSBackend(initial_bond_dimension=2, max_bond_dimension=64,
max_observable_error=1e-3).evaluate(circuit, observables, bindings)` returns
values, observable-error bounds, attempted ranks and routed operation counts.
Arbitrary one/two-wire unitary gates are routed through temporary adjacent
SWAPs that restore the source wire convention. Supported larger gates are
decomposed first; larger opaque matrices require explicit synthesis.

Each rank retry starts from the original circuit. Canonical Schmidt truncation
produces a normalized state-distance bound; observable bounds use the triangle
inequality and Pauli coefficient norms. Every declared observable must meet
the tolerance. If rank, memory or cumulative-work limits prevent this, execution
raises `ResourceError`. Bounds exclude floating-point roundoff and can be loose.

This is an explicit approximation backend. It is never silently substituted
for exact training. Ordinary parameter shifts do not differentiate adaptive
rank selection or nonlinear truncation and are rejected for this backend.
Exact contraction, exact adjoint execution and compatible QPU execution remain
available. General high-entanglement circuits can still require exponential
classical resources; slicing and rank limits do not remove that complexity.

## Hybrid optimization

| Framework | Training entry point | Loss contract |
| --- | --- | --- |
| PyTorch | `HybridTrainer(model, optimizer, microbatch_size=32).step(batch, loss_function)` | `loss_function(model, microbatch)` returns a scalar mean or per-example vector |
| JAX / Flax | `make_train_step(optimizer, loss_function, microbatch_size=32)` | `loss_function(parameters, microbatch)` returns one loss per example |

Torch batches are nested tensors with a common leading sample dimension. Each
microbatch runs backward before the next forward. Existing DDP wrappers use
`no_sync` for intermediate microbatches and global sample counts for uneven local
batch sizes. All ranks must make coordinated calls with nonempty local batches.
Optional `amp_dtype=torch.bfloat16` or CUDA `torch.float16` applies AMP to
classical computation; quantum layers retain float32/float64 arithmetic.
Nonfinite steps are skipped consistently across DDP ranks. Gradient scaling,
clipping and scheduler updates happen once per accepted optimizer step.
`save(path)` / `load(path)` preserve model, optimizer, scheduler, scaler and
Torch RNG state using atomic checkpoint writes.
With DDP, every rank calls `save`: only rank zero writes, while the checkpoint
keeps each rank's RNG state. CUDA RNG capture touches only the local device.

JAX accumulates with `lax.scan` and an unpadded tail. `has_aux=True` also reduces
scalar or per-example metrics. `axis_name` enables sample-weighted `psum` in
`pmap`/`shard_map`; global `NamedSharding` batches can also enter a jitted step.
On newer JAX, manual-axis parameter types are made varying before local AD,
preventing implicit cotangent reduction followed by a duplicate explicit sum.
Optimizer state is unchanged on a rejected step. Flax can use the same learner
with its variable tree and `apply` function.

Microbatch equivalence requires a sample-separable model/loss. BatchNorm and
other batch-dependent operations retain microbatch semantics. Admission limits
for a simulator or rollout buffer do not bound total process or optimizer memory.

## Complete quantum PPO

The policy contract is `QuantumActorCritic`: observation/action sizes, a discrete
or continuous flag, and policy/value outputs. Supply an environment instance,
a factory, or a sequence of independent factories/instances. Environments follow
the Gymnasium reset/step protocol; Gymnasium itself is optional.

| Integration | Construction |
| --- | --- |
| PyTorch | `oq.torch.PPOTrainer(policy, optimizer, environments, config=oq.PPOConfig())` |
| JAX | `oq.jax.PPOTrainer(policy, parameters, optimizer, environments, config=oq.PPOConfig())` |
| Flax Linen | `oq.flax.PPOTrainer(module, variables, optimizer, environments, config=oq.PPOConfig())` |

Call `train(total_timesteps)` for batched rollout collection, final-observation
bootstrap, GAE, advantage normalization, shuffled PPO minibatches, clipped policy
and value objectives, entropy regularization, gradient clipping and KL stopping.
Discrete action masks come from reset/step `info["action_mask"]`. Continuous
policies use tanh-Normal actions mapped to finite environment bounds. Recorded
pre-tanh actions avoid inverse-tanh saturation; likelihoods include the affine
Jacobian. Environment interaction runs on the host, with batched policy calls.

`evaluate(environment, episodes=...)` runs a separate deterministic evaluation
environment. `save(path)` / `load(path)` preserve policy, optimizer, learner
counters and sampling RNG. JAX checkpoints use NPZ arrays without pickle.
Environment trajectories restart on loading because arbitrary environment state
cannot be serialized by the library. `train` returns up to the latest 100 rollout
reports and can stop through a callback. A final vector step may exceed the
requested count by at most the number of environments minus one.
DDP uses distinct rank seeds, synchronized stopping, and collective checkpoints;
its requested environment-step budget and rollout counters are per rank.

SAC and V-trace remain composable objectives; `PPOTrainer` does not claim to be a
SAC runner. No environment, dataset, trained policy or performance suite is bundled.

## Structured program compilation

`oq.compile_program(value, target="openqasm3")` and
`QuantumProgram.from_callable(factory, ...).compile("openqasm3")` preserve
SDK-supported control-flow blocks rather than flattening them into circuit IR.
Qiskit/OpenQASM 3 routes retain while/for/if structure, lower defined unitary gates
to a portable basis, and emit explicit scope-local `gphase` instructions.
Loop ranges are retained; runtime iterations are neither executed nor guessed.

OpenQASM imports stage supported nonlinear expressions of global float inputs
as symbolic parameters, then restore their expression graphs after SDK import.
Gate-local arguments and shadowed runtime names keep their own scope. Gate
identity is checked by SDK instruction type, not by trusting a user-supplied name.
Unsupported opaque instructions, calibrations, synthesis sizes and resource
overruns fail explicitly. Target syntax is checked after export.

`strategy="ir"` retains the existing strict portable IR route. Whole-program
equivalence is not certified by syntax checking. Host Python runs before circuit
capture; imports, arbitrary runtime code, backend context, unsupported OpenQASM
features and other languages are not automatically translated. SDK import and
export support differ; see the [official feature table](https://quantum.cloud.ibm.com/docs/en/guides/qasm-feature-table).

## Research basis

These components are independent implementations of applicable methods, with
explicit scope. A newer paper is not evidence of superiority on every task.

| Source | Applied method |
| --- | --- |
| Vidal, [Entanglement Renormalization](https://doi.org/10.1103/PhysRevLett.99.220405), PRL (2007) | Multiscale unitary disentanglers and coarse-graining |
| Markov and Shi, [Simulating Quantum Computation by Contracting Tensor Networks](https://doi.org/10.1137/050644756), SIAM J. Comput. (2008) | Contraction structure determines classical cost |
| [TensorCircuit-NG](https://arxiv.org/abs/2602.14167), preprint (2026) | Tensor-native AD, bounded approximate execution and data-parallel training design; its cluster results are not OpenQML measurements |
| [PPO-Q](https://arxiv.org/abs/2501.07085), preprint (2025) | Trainable hybrid preprocessing/readout in a complete PPO pipeline; this implementation uses tanh-Normal rather than that paper's Beta policy |
| [PyTorch AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html), [JAX manual parallelism](https://docs.jax.dev/en/latest/notebooks/shard_map.html) | Correct gradient accumulation, precision control and collective semantics |
