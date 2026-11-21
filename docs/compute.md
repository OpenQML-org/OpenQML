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

The eager fusion planner builds wire dependencies within windows of 64 gates.
It selects ready gates that fit the current one- or two-wire operator and
preserves the order of gates sharing a wire. Barriers and wider gates end a
window. This lets interleaved, independent gate sequences share statevector
traversals. Torch graph capture uses source-order grouping. Fixed matrix
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
backend. External providers keep their own optimization policies.

The gate dependency schedule draws on [Li, Wang and Fan, IEEE Transactions on Quantum Engineering (2026)](https://doi.org/10.1109/TQE.2026.3705783).

## Reversible automatic differentiation

`checkpoint_interval="adjoint"` selects inverse-gate reconstruction for pure
statevectors, expectations and dense quantum layers. The first reverse pass
reconstructs each input state with the conjugate-transposed gate. Contracting
the incoming cotangent with that input gives the local matrix derivative.
The framework differentiates the gate's parameter expression and accumulates
reused symbols and broadcast axes. The global phase participates in complex
state derivatives. Full-register activation storage is independent of circuit
depth. Gate metadata, bindings, result tensors and framework workspaces also
occupy memory. [Jones and Gacon](https://arxiv.org/abs/2009.02823)

PyTorch eager backward runs the inverse sweep. Higher derivatives replay the
framework graph, and forward AD, function transforms and Torch graph capture
use ordinary native evolution. JAX supports the inverse sweep under `jit`,
`vjp` and reverse-mode `grad`. Use the default mode or integer checkpoint
intervals for JAX forward-mode products. Inverse reconstruction accumulates
floating-point roundoff with depth.

## Matrix-product readouts

`openqml.sim.MPSBackend().readout(circuit, observables)` evaluates bound one- and two-wire
unitaries from the zero product state. The default keeps the full bond at each
split. When the left side fits on the bond, an identity factor and a reshape
suffice. A taller left side uses a
reduced QR factorization. The readout therefore needs no singular values in
exact mode. Routing brings nonadjacent gates together with nearest-neighbor
SWAPs and tracks the logical order through measurement. A logical SWAP only
updates the wire map. Admission includes the cost of routing.

`MPSBackend(layout="auto")` compares source order with a chain assembled from
frequent two-wire interactions. It prefers an admitted layout, then compares
estimated contraction and factorization work, routed swaps and peak workspace. The low-level backend
defaults to source order. The certified readout and gradient entry points
use automatic placement when they create their own backend. Observables keep
their original logical wire order under either layout.

For multiple Pauli words, the simulator compares independent full sweeps with
a forward sweep and reusable right identity environments. It stores only the
boundaries needed by the words and contracts each interior exactly. The shared
path requires fewer one-site transfers and must fit the workspace.
`MPSReadout.readout_transfers` gives the number of one-site transfers actually
executed during measurement. For all single-wire $Z$ readouts on an $n$-site
chain ($n>2$), the shared path uses $3n-2$ transfers.

Set `max_discarded_weight` to enable Schmidt truncation. A thin QR moves the
orthogonality center to the active bond before splitting. Each split keeps
enough leading values to meet the local relative discarded-weight target,
subject to `max_bond_dimension`. A tighter bond cap can discard more than
the target. The reported bound uses the actual discarded weight at every
split, including routing operations. If those weights are $w_j$, the
trace-distance bound is $D=\sin(\min(\pi/2,\sum_j\arcsin\sqrt{w_j}))$.
For a Pauli expansion of a Hermitian observable the absolute error is bounded
by twice $D$ times the sum of absolute coefficients of nonidentity terms.
`openqml.sim.MPSReadout.absolute_error_bounds` reports these truncation-only bounds per
observable. Floating-point roundoff is additional. [Zhou, Stoudenmire and Waintal, Physical Review X](https://doi.org/10.1103/PhysRevX.10.041038)

### Choosing bond ranks from the requested output accuracy

`openqml.sim.certified_readout(program, observables, absolute_error=...)`
accepts an absolute tolerance per output, or one tolerance for all outputs.
`plan_readout(circuit, observables).evaluate(bindings, absolute_error=...)`
reuses the unitary causal cone across bindings. The returned `simulated_wires`
maps the compact register back to the source. Its state-distance bound describes
that compact circuit. Each observable bound applies to the corresponding full-circuit output.

At a split with discarded squared Schmidt weight $w$, each reduced density
matrix on one side is a mixture of the retained and discarded components on
orthogonal subspaces. Its trace distance from the normalized retained component
is $w$. The full pure states have trace distance $\sqrt w$. OpenQML traces each
Pauli term backwards through later gates and routed swaps. Clifford gates map
known Pauli factors exactly. Commuting rotations preserve them. Where a gate
can mix components, the simulator keeps the affected support and tracks the
remaining known factors independently. This uses the same Clifford tables as
the Pauli compiler. [Rudolph et al., PRX Quantum](https://doi.org/10.1103/6vd7-l9bn)

For one observable at split $j$, let $A_j$ and $B_j$ be the sums of absolute coefficients
of nonidentity terms that respectively stay on one side or cross the cut.
Telescoping across the actual truncated prefixes with exact suffixes gives

$$
|\Delta\langle O\rangle| \leq 2\sum_j (A_j w_j + B_j\sqrt{w_j}).
$$

The simulator divides the remaining allowance over the remaining splits and
selects a rank from the measured singular spectrum. It reports the tighter of
this bound and the state-distance bound. A hard bond cap can exhaust the
requested accuracy. The default raises `ResourceError` in that case.
`strict=False` returns the achieved bounds. Tolerance zero requests zero
discarded contribution to the declared outputs. These deterministic bounds
cover Schmidt truncation in exact arithmetic. [Schollwöck, Reviews of Modern Physics](https://doi.org/10.1103/RevModPhys.77.259)

## Tensor contraction planning

In the `portfolio` planner, each fixed binary contraction tree has two
possible child orders at every join. OpenQML tracks the peak live elements
and net change in storage for each subtree. A bottom-up pass selects the
minimum estimated peak over its depth-first child orders. This preserves
that tree's arithmetic count. The search over different trees remains bounded
by `search_trials` and `max_work`.

Under memory pressure, the planner scores candidate binary slices against the
whole live schedule. It accounts for both assignments and prefers the lowest
arithmetic cost among candidates that meet the limit, or the lowest estimated
peak while the limit is still exceeded. All assignments contribute to the
result. These are structural memory and arithmetic estimates.
[Pfeifer, Haegeman and Verstraete, Physical Review E](https://doi.org/10.1103/PhysRevE.90.033315)

## CPU statevector kernels

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

Torch 2.2's Dynamo requires Python 3.10 or 3.11; use a newer Torch release for
compilation on Python 3.12. Expression dtype and broadcast checks use static
type/shape guards so full-graph tracing retains real-domain validation on the
supported legacy versions, including native readouts and checkpointed circuits.

TorchInductor also requires a working C++ toolchain for generated CPU code.
On Windows, install Visual Studio Build Tools with **Desktop development with
C++**, then start Python in an **x64 Native Tools Command Prompt** where `cl`
is available. See [PyTorch's Windows setup](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html).
An observed missing compiler raises `ResourceError` with setup instructions;
unrelated compiler failures retain their original exceptions. OpenQML does not
silently switch compilation backends. Explicitly pass `compiler="aot_eager"`
to `compile_compute` or `compile_circuit` for graph execution/autodiff without
C++ code generation, or `compiler="eager"` to debug Dynamo capture. These modes
do not provide Inductor kernel fusion or CUDA Graph replay.
`environment_report()["cxx_compiler"]` reports executable discovery only;
it does not certify a working compiler, SDK, linker or accelerator.

On macOS, generated CPU kernels also need OpenMP headers and libraries.
Install `libomp` and set `OMP_PREFIX` to its installation prefix, for example
the path printed by `brew --prefix libomp`. An observed missing `omp.h` raises
an actionable `ResourceError`; other C++ compilation failures retain their
native exceptions. Portable macOS CI installs and configures this prerequisite.

On legacy Dynamo, resolve lazy module/function references before constructing
a compiled callback, for example `norm = oq.finance.transport_dual_norm`.
Use that reference in the callback so first-use namespace discovery stays
outside the graph capture.

Compilation has an initial cost, and small circuits can remain faster eagerly.
TorchInductor can delegate unsupported complex operators to native kernels.
Fusion benefit depends on locality, gate mix and reuse. These mechanisms do not
establish a universal acceleration factor or measured multi-GPU/QPU throughput.

## Readout working storage

In dense NumPy statevectors, a non-diagonal Pauli word pairs basis indices
that differ by its X mask. Above twelve qubits, the readout takes one member
of each pair and accumulates its real contribution in tiles. The NumPy
adjoint also groups Hamiltonian terms with the same X mask. Each group
reads the source amplitudes once per tile and combines the Z-dependent signs
before multiplying by the amplitudes. [Grouped bit-flip representations in BinSim](https://arxiv.org/abs/2606.28952)

For a gate with several active angles, the adjoint contracts the state and
costate over the spectator wires once. It pairs the resulting small matrix
with each angle derivative. A forward directional product combines the local
derivative matrices before applying them to the state.

Real-arithmetic readout graphs use immutable last-use schedules and reuse
register slots after their final consumer. A slot assignment replaces a Python
reference; tensor data is never overwritten. Shared inputs, repeated operands
and requested roots remain live for all consumers. Pauli and factored expectation
plans report live-register counts and charge primal workspaces against that
schedule. Framework autodiff may retain its own saved tensors; those tapes,
compiler workspaces and allocator reserves remain outside this primal model.

TorchInductor/XLA can fuse the resulting native arithmetic when a surrounding
model or circuit is compiled. Numeric CUDA readouts also expose an explicit
instance-local compiled path described in [runtime dispatch](runtime.md).

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
