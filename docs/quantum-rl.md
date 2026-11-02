# Quantum reinforcement learning

OpenQML supplies policy modules and native training objectives. Environment
ownership, replay storage, optimizers and distributed workers belong to the
application. Gymnasium, MuJoCo and training datasets are not dependencies.

## Policy modules

`oq.torch.QuantumActorCritic`, `oq.jax.QuantumActorCritic` and
`oq.flax.QuantumActorCritic` implement an observation encoder, trainable input
scales, a quantum feature layer and action readout. An independent classical
critic supplies the value baseline. `ActorCriticOutput` contains `policy`,
`value` and `log_std`; `policy` is categorical logits, or a Gaussian mean when
`continuous=True`. Gaussian log standard deviations are bounded to [-20, 2].
Arbitrary leading observation dimensions are preserved.

The default universal circuit uses identity-paired RY/RZ/RZZ blocks. With
`backend="auto"`, construction uses the cost policy to select and pin a compatible QPU, GPU or CPU
for the primal circuit and its parameter-shift derivatives. Pinning preserves
provider-specific lowering, noise placement and stochastic reuse contracts
through an update. Rebuild the layer after changing devices or their configuration.

`quantum_layer` accepts a matching native module/functional layer. Supplying
`IQPQuantumLayer` gives exact structured local readouts with native higher
derivatives and O(n + edges) arithmetic. This restricted family is efficiently
classically evaluable; it does not imply a quantum speedup. Native PyTorch/JAX
backends preserve accelerator tensors and ordinary optimizer parameter ownership.

## Objectives and trajectory targets

Functions in `oq.rl` are also exported from `openqml`. Tensor types select
NumPy, Torch or JAX without loading optional frameworks on a core-only import.
Native losses support compilation, vectorization and differentiation. Half
precision inputs accumulate in float32; no learner step copies tensors to CPU.
Discount, trace decay and clipping controls are static Python reals; learned
temperature remains a native tensor.

| API | Contract |
| --- | --- |
| `generalized_advantages` | Time-major GAE and value targets; per-transition `next_values` include final observations |
| `vtrace_targets` | Clipped importance-weighted value/policy targets for delayed actor trajectories |
| `categorical_statistics` | Masked log probabilities, entropy and a native validity flag; invalid actions/empty rows yield NaN log probability |
| `ppo_loss` | Clipped policy objective, optional clipped value objective, entropy, approximate KL and clipping fraction |
| `squashed_gaussian` | Caller-owned reparameterization noise; tanh actions and a stable log-Jacobian |
| `sac_loss` | Twin-critic soft targets, actor/critic losses and optional automatic temperature loss |
| `surrogate_fit_loss` | Detached measurement-target MSE, or stable binary-logit cross entropy |

True termination disables bootstrap. Time-limit truncation retains the final
observation's value but stops GAE/V-trace recursion into the reset episode.
Targets and behavior-policy statistics are detached. Affine scans use JAX's
work-efficient associative scan, or logarithmically many native NumPy/Torch
vector steps, without division by episode masks.

For SAC, compute `actor_q1/actor_q2` with critic parameters frozen while retaining
action gradients. The objective detaches target networks, actor temperature and
the policy densities in the temperature update. External action rescaling
requires its corresponding density Jacobian. Optimizer steps remain caller owned.
Record the executed behavior policy's log probabilities for off-policy correction;
V-trace handles policy lag, not hardware bias or arbitrary replay distributions.

## Explicit hardware response surrogates

`oq.torch.QuantumSurrogate` and `oq.jax.QuantumSurrogate` expose three named paths:
`quantum` executes the supplied quantum layer, `surrogate` queries only a fitted
classical response, and `straight_through` retains measured forward outputs with
surrogate input derivatives. The default is `quantum`.

`fit_loss` updates only the surrogate from detached control/measurement pairs.
Actor updates detach surrogate parameters while preserving control gradients.
All trainable circuit controls must therefore be represented in its inputs.
Binary feature surrogates optionally use logits/BCE; Pauli expectation surrogates
use MSE. Measurement aggregation and shot settings must match the deployed path.
Surrogate and straight-through gradients are biased approximations. Response-fit
error alone does not certify policy-gradient error. Hardware inference and
surrogate-assisted training are deliberately separate explicit modes.

## Research basis

- [PPO-Q (2025)](https://arxiv.org/abs/2501.07085): hybrid encoder/circuit/action
  architecture and quantum actor or value functions.
- [Training Hybrid Quantum Policies with Classical Surrogates for Actor-Critic
  Reinforcement Learning, v7 (October 2026)](https://arxiv.org/html/2503.09119v7):
  response fitting and circuit-free actor updates; the paper studies TD3 and
  aggregated bitstrings, while OpenQML exposes general response interfaces.
- [IMPALA (ICML 2018)](https://proceedings.mlr.press/v80/espeholt18a.html): V-trace.
- [Soft Actor-Critic (ICML 2018)](https://proceedings.mlr.press/v80/haarnoja18b.html)
  and [SAC Algorithms and Applications](https://arxiv.org/abs/1812.05905):
  maximum-entropy twin critics and temperature adaptation.
- [PPO](https://arxiv.org/abs/1707.06347) and
  [GAE](https://arxiv.org/abs/1506.02438): clipped objectives and advantage targets.

These are independently implemented components. The cited preprints are not
treated as journal-validated superiority or task-level quantum advantage.
