# Submission-aware quantum training

Cloud latency can dominate a small variational circuit. `ShiftPlan.linearize`
packs the current readouts and their occurrence-wise parameter-shift stencil
before reading any result. It supports nonlinear shared angles, controlled
rotations and supported noise-channel derivative recipes.

| Field or operation | Meaning |
| --- | --- |
| `values` | Read-only NumPy `[batch,readouts]` forward expectations |
| `jacobian` | Read-only `[batch,readouts,parameters]` derivatives in `parameters` order |
| `vjp(cotangents)`, `jvp(tangents)` | Products without further circuit execution |
| `circuit_evaluations` | Actually submitted logical circuit rows |
| `submission_batches` | OpenQML backend batch calls |

One stencil uses one backend call when it fits `chunk_size` and provider limits.
A provider can split or bill that call further. Packing does not reduce circuit
rows or guarantee shot-cost savings. Single-occurrence Pauli rotations need
`2P+1` rows per example for `P` selected angles; other recipes count occurrences.

```python
import numpy as np
import openqml as oq

x, w = oq.Parameter("x"), oq.Parameter("w")
circuit = oq.Circuit(1).ry(x * w, 0).rz(w, 0)
plan = oq.ShiftPlan(circuit, [oq.PauliString("Z")], backend="numpy")
linearization = plan.linearize({"x": [0.2, 0.4], "w": 0.3}, chunk_size=64)
targets = np.array([[0.1], [-0.2]])
cotangents = 2 * (linearization.values - targets) / len(targets)
gradient = linearization.vjp(cotangents)
```

## Torch and JAX layers

`gradient_method="batched-shift"` opts an external `QuantumLayer` into this
linearization. Forward includes derivatives for active input and weight
coordinates. First-order backward/JVP operations reuse its Jacobian. Leading
input/weight batches broadcast, and shared gradients reduce to original shapes.
Flax uses the same mode through its JAX quantum layer.

```python
from openqml.qml.interfaces.torch import QuantumLayer


def provider_layer(estimator):
    return QuantumLayer(
        circuit,
        input_parameters=("x",),
        weight_parameters=("w",),
        backend=oq.QiskitEstimatorBackend(estimator),
        gradient_method="batched-shift",
        gradient_chunk_size=256,
        device="cpu",
    )
```

Supply a caller-initialized estimator and use an ordinary Torch/JAX optimizer.
External execution crosses a host boundary; returning a Tensor on a chosen
device does not turn provider execution into a native GPU kernel. Higher quantum
derivatives require a native framework backend and fail explicitly here.

The memory tradeoff is `O(batch * readouts * parameters)`, plus cloned IR rows
and backend buffers. Working-memory admission precedes external submission.
The default `parameter-shift` mode retains streamed products. Batch/microbatch
splitting can create further submissions. A linearization belongs to its
original bindings and lives only in its returned object or current autodiff
residual. Numeric results are never retained in a global cache.

Deterministic backends can reuse equal binding rows. Statistical backends keep
separate rows. Finite-shot derivatives remain statistical estimates; evaluating
nonlinear loss cotangents at noisy forward values can introduce finite-shot bias.
JAX marks nondeterministic provider calls as I/O effects so JIT
cannot erase unused submissions or merge equal calls. Callback errors retain
JAX's runtime error boundary; OpenQML does not replay a failed submission.
Upstream I/O effects propagate through the custom derivative primitive, checked
at the supported JAX floor and current release.

`TrainingWorkload(joint_primal_gradient=True, ...)` models the combined phase.
Queue waves, provider limits, gradient chunks and sequential microbatches still
apply. General layers selecting from configured quotes pass this flag when
batched shifting is requested.

This adopts fixed-stencil job packing from
[Training quantum neural networks one job at a time (November 2026)](https://arxiv.org/abs/2611.16365).
It excludes the paper's sinusoid-search optimizer and does not reproduce its
hardware accuracy or job-count comparisons. Callback behavior follows the
[JAX I/O contract](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.io_callback.html).
