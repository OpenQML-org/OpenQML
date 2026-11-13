# Native quantum geometry

`QuantumMetricPlan` now supports resident PyTorch/JAX metric products, damped
natural-gradient solves and an optional spectral step. Use the parameter order
declared by the plan. Native bindings may be scalars or share one visible
sample axis. A real floating vector selects a common coordinate displacement.
`sample_weights` optionally defines a normalized nonnegative empirical metric;
otherwise samples contribute their uniform mean. Inputs are staged once per
solve, and batched state evolution avoids a Python circuit call per example.

| Method | Result |
| --- | --- |
| `native_matvec(bindings, vector, backend="torch")` | Exact real Fubini–Study metric times a vector |
| `native_natural_gradient(bindings, gradient, backend="torch", ...)` | Native direction and a true damped-system residual certificate |
| `spectral_step(bindings, gradient, block_shapes=..., ...)` | Bounded spectral-natural displacement and the original CG certificate |
| `DistanceAwareQNM(plan, block_shapes, ...).step(...)` | Stateful quantum Fisher momentum, active polar mixing and a trajectory-dependent displacement bound |

The metric is `G = Re(J* (I - |psi><psi|) J)`, with quantum Fisher information
`F = 4G`. Forward and reverse native differentiation evaluate `Gv` without a
parameter-by-parameter metric matrix. Real and imaginary state coordinates
share the same JVP/VJP convention in Torch and JAX. Projecting the radial and
phase tangents removes global-phase gauge directions. Independent circuit
factors contribute additive metrics, including reused and nonlinear parameters.
This is simulator geometry for unitary pure states; it does not estimate a
hardware QFIM from finite-shot QPU measurements.

Native tensors retain their device and differentiation graph. NumPy conversion
is not used for trainable bindings, gradients, metric products or directions.
The unitary pure-state representation still costs exponential storage in the
largest connected factor. Admission models conservative primal/JVP/VJP state
storage for the visible sample batch; compiler buffers, allocator pools, hidden transform
batch axes and autodiff tapes require separate application budgets.

## Geometry-first spectral steps

```python
import torch
import openqml as oq

a, b = oq.Parameter("a"), oq.Parameter("b")
circuit = oq.Circuit(2).ry(a, 0).cx(0, 1).rz(b, 1)
metric = oq.quantum_metric(circuit, parameters=("a", "b"))
theta = torch.tensor([0.3, 0.4], dtype=torch.float64, requires_grad=True)
bindings = dict(zip(metric.parameters, theta, strict=True))
loss = oq.expectations(circuit, [oq.PauliString("ZI")], bindings, backend="torch").sum()
gradient = torch.autograd.grad(loss, theta)[0]
proposal = metric.spectral_step(
    bindings,
    gradient,
    block_shapes=((1, 2),),
    backend="torch",
    damping=0.01,
    mixing=0.1,
    learning_rate=0.01,
    max_step_norm=0.05,
)
with torch.no_grad():
    theta.sub_(proposal.step)
```

First solve `(G + damping I)d = gradient` using bounded, scale-stable native CG.
Then reshape the direction into caller-declared circuit-semantic matrix blocks,
compute their active partial polar factors and match each polar component's
Frobenius norm to its original block before mixing. Zero singular subspaces
remain inactive. The observed gradient alignment selects the unmodified natural
direction if spectral mixing is non-finite or non-descending. Finally, bound
the Euclidean displacement by `min(learning_rate * ||d||, max_step_norm)`.

`proposal.natural` certifies the original damped solve. It does not certify that
the mixed or rescaled step solves that linear system. Positive local alignment
and a displacement bound do not prove loss reduction for an arbitrary nonlinear
task or learning rate. SVD is piecewise smooth, with singular derivatives at
rank changes or repeated singular values; optimizer updates normally run under
`no_grad`. The native metric products retain ordinary higher derivatives.

Choose blocks from the actual circuit parameterization; arbitrary reshaping has
no claim to improve optimization. The metric describes the quantum state block,
not the Fisher information of an entire classical prediction head. Classical
parameter groups can retain their existing optimizer.

The stateless spectral method implements the geometry-first active-polar mixing mechanism described in
[Quantum Natural Muon (November 2026)](https://arxiv.org/abs/2611.10105), with an
explicit alignment safeguard. It is a stateless variant, excluding that paper's
momentum and trajectory-dependent distance schedule. The metric convention and
matrix-free products follow [Quantum Natural Gradient](https://arxiv.org/abs/1909.02108).
No universal convergence, speedup or quantum-advantage claim follows from these
implementation checks. GPU placement requires a real framework GPU runtime;
physical GPU validation is a separate hardware-runner test.

## Stateful distance-aware Quantum Natural Muon

```python
optimizer = oq.DistanceAwareQNM(
    metric,
    ((1, 2),),
    damping=0.01,
    momentum=0.9,
    initial_radius=0.01,
    distance_growth=0.1,
    max_step_norm=0.05,
)
state = None
# Recompute bindings and loss gradients at the current parameters each step.
result = optimizer.step(bindings, gradient, state=state, backend="torch")
with torch.no_grad():
    theta.sub_(result.step)
state = result.state
```

The stateful method solves `(F + damping I)u = gradient`, with `F=4G`.
This differs from the existing stateless `G` convention: damping must be
selected in the intended convention. Natural-space momentum is
`m = momentum * previous_m + (1-momentum) * u`. Active polar mixing acts on
the declared matrix blocks of `m`. The radius is
`min(learning_rate * ||m||, initial_radius + distance_growth * max_distance,
max_step_norm)`, and the displacement divides the mixed direction by its norm
plus `normalization_epsilon`. `max_distance` is the largest observed distance
from the initial parameter vector. With `alignment_guard=True`, a direction
without positive observed gradient alignment restarts from `u`.

`DistanceAwareState` contains native initial parameters, momentum, maximum
distance, an int32 step counter and a circuit/configuration signature. Its
floating arrays are detached between steps. Save and restore all fields;
configuration, parameter order and circuit identity must match. Selected
trainable coordinates are shared scalar bindings. Auxiliary features may have
one weighted sample axis. The caller owns parameter mutation and state lifetime.
Torch and JAX retain native placement; JAX can JIT the state transition.
`compile_compute` preserves typed invalid-state failures at checked boundaries.

The optional scalar `learning_rate` override supports scheduling, including zero.
`metric_reducer` accepts and returns a native vector of the same shape/dtype;
distributed callers must provide one common positive semidefinite metric to all
solve iterations. The integrated Torch optimizer supplies this reduction.

This implements the momentum and distance schedule in Algorithm 1 of
[Quantum Natural Muon](https://arxiv.org/abs/2611.10105), with the documented
optional alignment safeguard. It does not establish convergence or faster
training for arbitrary tasks. A single-row block has no nontrivial polar
redistribution; choose meaningful multirow blocks from the circuit layout.

## Joint Torch optimization

Use [`hybrid_qnm` with `HybridTrainer`](hybrid-training.md#joint-adamw-and-quantum-natural-muon)
to train classical and quantum parameters, capture the participating feature
microbatches, reduce DDP geometry and resume both optimizer states together.
