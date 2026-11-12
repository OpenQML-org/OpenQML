# Shared compute kernel

OpenQML uses one process-local `ComputeKernel` across its local execution paths.
Domain modules retain their mathematical algorithms and share native array
conversion, matrix products, tensor contraction metadata, finite-value guards,
linear solves, SVD and resource policies. NumPy delegates dense products to its
BLAS implementation. PyTorch and JAX retain tensors, device placement,
autodifferentiation and native compilation. Fused framework operators, including
PyTorch scaled dot-product attention, continue to use their native kernels.

| Workload | Shared services |
| --- | --- |
| Statevectors, density matrices and quantum channels | Native arrays, local GEMMs, finite domain checks, memory ceiling |
| Tensor networks and tensor trains | Shape-specialized GEMMs, shared structural plans, cached NumPy contraction paths, SVD |
| Hybrid attention, graph, recurrent and generative networks | Native contractions, projections, arrays and workspace admission |
| Financial risk, allocation and pricing | Native factor contractions, linear solves, scaled norms and memory/work admission |
| Cross-framework programming | Unitary/Kraus checks and the same simulation primitives for numerical verification |
| Runtime providers | Common workspace options and state, density and expectation validation |

## Circuit construction and execution lowering

`Circuit.append` validates the new operation and shares immutable prefix chunks.
Appending does bounded work independent of the existing length. The public
`operations` value remains a tuple, materialized on first read; repeatedly
reading every intermediate prefix still requires the corresponding output work.
Forks, dataclass replacement, JSON and pickle retain their value semantics.
Cache accounting visits shared expression nodes once, rather than expanding
their logical trees. Deep parameter graphs do not trigger recursive counting.

Local NumPy, Torch and JAX state evolution groups consecutive gates whose
combined support fits one or two wires, up to 64 gates per group. Fixed matrix
products are prepared once in the bounded metadata cache. Parameter-dependent
products use native differentiable matrix operations on the selected device;
cached plans contain no tensors or parameter values. This reduces full-state
updates and retained state activations for suitable local gate sequences.
Torch graph capture builds immutable instructions without accessing cache locks
or NumPy, leaving constant folding to the native compiler. JAX can stage the
same operations through XLA. Native registers below three qubits retain the
ordinary path; larger registers also retain it when the state is too small
relative to the local matrix work. This is a conservative structural heuristic,
not a hardware autotuner.

Barriers, checkpoint segments and larger gates delimit groups. Fusion falls
back to original gates if its temporary matrices exceed the selected workspace.
Checkpointed backward recomputes the same native fused segments; JVP, VJP,
second derivatives and vectorization retain their framework semantics.
`statevector(...,fusion=False)` selects unfused execution for every local
backend. External providers keep their own optimization policies. See also
NVIDIA's [gate-fusion execution model](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/overview/ex-svupdater.html);
OpenQML's native fusion does not require cuQuantum.

One-wire and adjacent two-wire applications use tensor views and batched GEMM.
Separated pairs use four basic input views and one reusable quarter-state
product buffer. They do not gather indexed state copies. Larger local matrices
retain the bounded tile implementation. These working buffers remain subject
to dense representation admission; general entanglement still costs exponential
memory. Precomputed byte limits remove repeated MiB conversion from the common
admission path. Public validation and symbolic compiled checks remain active.

`compile_circuit` captures an entire native circuit, its declared readouts and
an optional differentiation transform. Python gate selection runs during
tracing. Warm execution calls the compiled graph with new parameter arrays.
Torch uses [TorchInductor](https://docs.pytorch.org/docs/stable/generated/torch.compile.html);
JAX uses [XLA through jit](https://docs.jax.dev/en/latest/jit-compilation.html).
Put transforms inside this boundary with `transform=...`; the returned callable
checks admission and result errors at its host boundary. Native shape/dtype/
device specialization can trigger additional compilation.
Classical branch bookkeeping for density execution is resolved into a static
register program before tracing. Compiled Torch transforms require PyTorch
2.4 or later; ordinary autograd through compiled circuit values works on the
base Torch version. OpenQML does not change process-global compiler settings.

Compilation has an initial cost, and small circuits can remain faster eagerly.
TorchInductor can delegate unsupported complex operators to native kernels.
Fusion benefit depends on locality, gate mix and reuse. These mechanisms do not
establish a universal acceleration factor or measured multi-GPU/QPU throughput.

## Configuration

```python
import openqml as oq

kernel = oq.configure_compute(
    policy=oq.ComputePolicy(
        max_memory_mb=512,
        cache_entries=256,
        cache_bytes=8 * 2**20,
        validation_atol=1e-6,
        validation_rtol=1e-7,
    )
)
print(kernel.cache_info())
```

`get_compute_kernel()` returns the same service used by local quantum,
hybrid and financial array bridges. `configure_compute()` atomically
installs a new service with an empty cache. Global updates apply to new operations. Existing operations and native-array
bridges retain their kernel snapshot. Use `compute_context(policy=...)` or
`compute_context(kernel=...)` to isolate configuration in a thread or async task;
nested scopes restore their predecessor, including after exceptions. Runtime
worker submissions propagate these scopes. Kernel policies are read-only.

`compile_compute(function, backend="jax")` uses checkify to carry checks through
the compiled graph and raises their typed errors at the host call boundary.
`backend="torch"` uses native graph assertions and checks returned values;
`compiler="eager"` selects Torch's eager graph engine for debugging. Put gradient
and batching transforms inside `function`. Each compiled callable pins its
kernel; `.graph` exposes lower-level execution (JAX returns error status and
values). Raw third-party JIT calls follow their native compiler error semantics.
Internal JAX checkpoints carry error status as data before rematerialization;
Torch traces immutable policy snapshots without reading task contexts in graphs.

`max_memory_mb=None`, the default, preserves callers' operation-specific
budgets. A configured ceiling is combined with each requested budget using the
smaller value. For example, a global 512 MiB ceiling preserves a caller's
64 MiB limit and reduces a 1024 MiB request to 512 MiB. Primitive output
admission includes broadcast dimensions, so small inputs cannot bypass the
ceiling by creating a large outer product.

## Numerical contracts

| Check | Default behavior |
| --- | --- |
| Provider state normalization | L2 norm within `validation_atol / 2` of one; scaled sum-of-squares avoids overflow |
| Provider density matrices | Hermitian and unit trace within `validation_atol`; minimum eigenvalue at least `-validation_atol` |
| Provider expectation values | Magnitude at most the observable coefficient bound plus `validation_atol + validation_rtol * bound` |
| Linear solve and SVD | Finite inputs and results; eager numerical factorization failures raise `NumericalError`, a `BackendError` subclass |
| Financial and channel domains | Eager and checked compiled calls raise typed validation errors |
| Resource admission | Invalid budgets raise `ValidationError`; rejected workspace or work estimates raise `ResourceError` before the admitted allocation |

These are validation tolerances. Solver convergence criteria, truncation
budgets, sampling uncertainty and each algorithm's error reports retain their
own contracts. Select precision and solver tolerances for the conditioning of
the actual problem. `max_memory_mb` describes modeled primal workspaces;
process RSS, input ingestion, autodiff tapes, allocator pools, library imports
and vendor workspaces have additional costs. Distributed plans apply their
workspace accounting per rank. The metadata cache has its own explicit budget.

Selected provider failures propagate. A quantum job is submitted once; the
runtime chooses eligible devices before submission using the existing cost
and capability policy.

## Bounded reuse

For explicit NumPy `expectations_batch(circuit, observables, rows, backend="numpy")`,
one circuit topology is prepared for all scalar binding rows. Shared symbolic
angles are bound once per row with the same scalar summation and constant-folding
semantics as `Circuit.bind`; the resulting columns execute in admitted chunks.
Fixed gates remain fixed so gate fusion can reuse their products. The bounded
cache retains symbolic metadata only. New calls evaluate their own bindings.
Effects, external providers and heterogeneous circuit lists retain their existing
execution paths. State and expectation checks apply to both paths.

Torch Gaussian attention prepares its augmented keys, padded values and
empty-mask sentinel once per forward. Query chunks share those tensors and
their gradient graph while native SDPA selects its available kernels. This
reduces repeated preparation; exact Gaussian attention still has quadratic
arithmetic in the token count. The separate positive-feature attention mode
retains its linear sufficient-statistic calculation.

Small-factor financial solves reuse one native pivoted LU for the primal solve
and its true-residual correction. Torch and JAX keep that factorization local
to a single solve; numerical arrays never enter the metadata cache. Implicit
derivatives still solve the original factor operator. NumPy retains its checked
LAPACK solve without adding SciPy to the core dependencies.

The shared LRU cache defaults to 256 entries and 8 MiB of conservatively counted
Python metadata. NumPy einsum paths are keyed by equation, shapes and memory
policy. Shape-specialized tensor contraction descriptors are interned across
plans. Binary NumPy contractions lower to prepared batched GEMMs; local tensor
programs execute preallocated slots without rebuilding node dictionaries.
Expanded broadcast/reshape copies are admitted before GEMM. If those copies
exceed the budget, a direct contraction traverses the original broadcast strides.
A per-key future prevents duplicate compilation while unrelated keys build
concurrently. Cache publication is locked; clearing a cache invalidates pending
publication from its previous generation. Oversized entries bypass retention; either zero cache limit
disables retention. `clear_cache()` releases metadata and resets counters.

Cached objects are immutable metadata. Parameter values, activation arrays,
device tensors, autograd graphs and provider results stay outside this cache.
Tensor networks can separately reuse intermediate tensors within a single
forward evaluation under their existing local reuse budget.

Native PyTorch/JAX contractions bypass the Python path cache so their graph
compilers retain control over fusion, contraction optimization and gradients.
The core requires no additional dependencies.

## Validation

`tests/test_compute_kernel.py` exercises shared dispatch, cross-module ceilings,
allocation rejection, concurrent cache publication, eviction, invalid policies,
overflow-resistant norms, provider tolerances, native gradients, Torch
`fullgraph` compilation and JAX `jit`. The existing domain suites check
independent reference results, conversions, training and distributed execution.

Implementation references:

- [NumPy contraction planning](https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html)
- [PyTorch tensor contractions](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html)
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
- [JAX functional error checks](https://docs.jax.dev/en/latest/debugging/checkify_guide.html)

Iterative factor and natural-gradient solvers recompute the true system residual.
A missed tolerance raises `ConvergenceError`, including through checked compiled
calls. Native metric solves can return explicit nonconvergence diagnostics with
`require_convergence=False`. Converged eager/native iterations stop issuing
additional metric products.
