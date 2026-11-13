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

This implements the geometry-first active-polar mixing mechanism described in
[Quantum Natural Muon (November 2026)](https://arxiv.org/abs/2611.10105), with an
explicit alignment safeguard. It is a stateless variant, excluding that paper's
momentum and trajectory-dependent distance schedule. The metric convention and
matrix-free products follow [Quantum Natural Gradient](https://arxiv.org/abs/1909.02108).
No universal convergence, speedup or quantum-advantage claim follows from these
implementation checks. GPU placement requires a real framework GPU runtime;
physical GPU validation is a separate hardware-runner test.
