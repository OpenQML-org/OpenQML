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
| Reinforcement learning | Native array placement and the same compute bridge as quantum and financial objectives |
| Financial risk, allocation and pricing | Native factor contractions, linear solves, scaled norms and memory/work admission |
| Cross-framework programming | Unitary/Kraus checks and the same simulation primitives for numerical verification |
| Runtime providers | Common workspace options and state, density and expectation validation |

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
reinforcement and financial array bridges. `configure_compute()` atomically
installs a new service with an empty cache. Configure before creating models,
starting worker threads or tracing native graphs. Retrace compiled Torch/JAX
functions after changing the policy: Python configuration is captured when
their graphs are built.

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
| Financial and channel domains | Eager invalid inputs raise `ValidationError`; compiled invalid lanes produce non-finite values |
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

The shared LRU cache defaults to 256 entries and 8 MiB of conservatively counted
Python metadata. NumPy einsum paths are keyed by equation, shapes and memory
policy. Shape-specialized tensor contraction descriptors are interned across
plans. A lock serializes cache publication and prevents duplicate compilation
of the same key. Oversized entries bypass retention; either zero cache limit
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
