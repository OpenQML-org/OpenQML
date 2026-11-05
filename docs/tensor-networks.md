# Quantum and classical tensor networks

OpenQML provides three complementary representations. General circuit networks
retain unitary circuit expressivity; quantum trees define an efficient physical
subfamily; matrix-product operators compress classical projections. Their
combination forms an ordinary trainable residual layer.

## Core representations

| API | Role | Exactness |
| --- | --- | --- |
| `TensorNetworkPlan` | Compile declared Pauli/Hamiltonian readouts into ket/bra contraction graphs | Sum every sliced index assignment; no tensor-rank truncation |
| `QuantumTreePlan` | Balanced forests of two-qubit unitary pooling channels | Exact local Pauli-transfer arithmetic, followed by a partial trace |
| `TensorTrainPlan` | Apply an MPO without constructing its dense matrix | Exact arithmetic for the selected cores/ranks |
| `TensorTrainMatrix.from_dense` | Offline TT-SVD import of a classical matrix | Full ranks by default; explicit rank/error controls return a truncation bound |

Floating-point roundoff remains in all numerical paths. Work and memory limits
are configurable representation-admission models, not process RSS limits or
autodiff-tape bounds. No global qubit-count ceiling is introduced.

## Differentiable circuit contraction

### Compiled paths and bounded tree reconfiguration

`optimizer="portfolio"` compares deterministic size-, arithmetic- and
separator-oriented greedy trees, then applies bounded three-way subtree
rotations. It selects a memory-admitted path by estimated complex arithmetic;
when slicing is necessary, it first selects a smaller separator and repeats
the search on the sliced graph. `search_trials` bounds the expensive joins
considered for reconfiguration and `max_work` accounts for candidate generation
and complete path profiling. `greedy` remains the low-planning-cost option;
`cotengra` enables the optional external hyper-optimizer.

Every chosen binary join compiles its axis permutations and GEMM dimensions
once. Forward execution uses the compiled kernels with native batched matrix
multiplication. The optimized tree retains all input tensors, index assignments
and observable coefficients. This preserves Torch/JAX differentiation and
explicit rematerialization.

Native `HybridArchitecture(...,native_execution="tensor-network")` models use
the portfolio planner and shared-subtree execution in Torch, JAX and Flax.
Direct `TensorNetworkPlan` and `QuantumLayer` users can select these controls:

```python
import openqml as oq

angle = oq.Parameter("angle")
circuit = oq.Circuit(3).ry(angle, 0).cx(0, 1).ry(0.3, 2).cx(1, 2)
observables = (oq.PauliString("ZZZ"), oq.PauliString("ZII"))
plan = oq.TensorNetworkPlan(
    circuit,
    observables,
    optimizer="portfolio",
    reuse_intermediates=True,
    max_memory_mb=64,
)
values, report = plan.evaluate({"angle": [0.2, 0.4]}, return_report=True)
print(values)
print(report)
```

`return_report=True` returns `(values,report)` with executed contraction count,
subtree cache hits, evaluated slices, cache high-water bytes and its budget.
These are forward-execution counters. Rematerialized backward execution creates
its own numerical cache and preserves the returned report snapshot. Compare
planning cost, forward counters and complete training time separately.

### Stream broadcast parameter grids

`plan.iter_evaluate(bindings, batch_chunk_size=128, backend="torch", like=x)`
yields native `[chunk_rows, readouts]` arrays in flat row-major order. The plan
reuses its compiled contraction topology and starts a fresh numerical cache for
each chunk. Parameter shapes such as `[B,1]` and `[1,T]` are gathered directly
from their source arrays; the full `[B,T]` parameter grid is never materialized.
If no chunk size is given, the memory model chooses the admitted row count.

```python
import numpy as np
import openqml as oq

a, b = oq.Parameter("a"), oq.Parameter("b")
circuit = oq.Circuit(2).ry(a, 0).rxx(b, 0, 1)
plan = oq.TensorNetworkPlan(circuit, (oq.PauliString("ZZ"),), max_memory_mb=16)
grid = {"a": np.linspace(-1, 1, 100)[:, None], "b": np.linspace(0, 1, 200)[None]}
total, count = 0.0, 0
for chunk in plan.iter_evaluate(grid, batch_chunk_size=128):
    total += chunk.sum()
    count += chunk.shape[0]
print(total / count)
```

`evaluate(...,batch_chunk_size=128)` collects chunks and restores the original
broadcast shape. Collection reserves memory for retained readouts and their
concatenation; use the iterator for larger outputs. The admission includes
quantum workspace, compiled metadata, subtree cache, parameter gathers and
indexing arrays. Supplied source arrays, framework allocators and retained
autodiff tapes are managed by the caller. Backpropagate chunk objectives or use
the trainer's microbatch accumulation to bound training graph retention.

Torch and JAX gathers remain in their native computation graphs, including
broadcast adjoints and higher derivatives. Chunk geometry is static for JAX JIT;
choose a practical chunk size to control the unrolled compiled graph. With
`return_report=True`, the iterator yields `(chunk, report)` and collection sums
forward work counters, reports the maximum cache high-water mark and adds
`chunks`. Empty grids return correctly shaped empty readouts.

### Demand-driven shared subtrees

`TensorNetworkPlan(...,reuse_intermediates=True)` reserves a bounded part of its
memory budget for common contraction subexpressions across readouts and exact
slices. Symbolic tensor identities, compiled joins and slice dependencies belong
to the plan. Execution checks a subtree's cache entry before visiting its
children, so a hit skips the full repeated traversal. Slice keys include only
the cut assignments on which that subtree depends; distinct sliced graphs use
separate cut namespaces while compatible unsliced branches can share work.

Shared native autodiff nodes accumulate their adjoints. The bounded LRU cache
accounts for tensor bytes and entry overhead; eviction repeats exact work.
Leaf slice views remain outside the value cache, keeping their backing gate
storage under the gate/working-memory admission. Numerical values are discarded
after every evaluation, including each rematerialized replay. Distributed
placement uses its own executor. This adapts shared-work DAG ideas from
[Jet](https://arxiv.org/abs/2107.09793); effectiveness depends on overlap between
chosen contraction trees. Use `tensor_options={"reuse_intermediates":True}` on a
native tensor-network `QuantumLayer` to enable it in training.

`QuantumLayer(..., execution_method="tensor-network")` works in Torch, JAX and
Flax. `tensor_options` controls `optimizer`, `search_trials`, `max_work`,
`max_operations`, `max_slices` and `max_flops`. The pure-Python bounded greedy
planner needs no additional dependency. Optional `openqml[tensor]` enables bounded
cotengra random-greedy hyperparameter search. Search and planning happen outside
the native forward/backward graph.

Each observable first receives an exact backward causal reduction. Gate tensors,
conjugate gate tensors, zero-state boundaries and Pauli boundaries form a scalar
network. A compiled binary path lowers contractions to tensor permutations and
batched matrix multiplication. Native tensors retain the device, computation
graph and higher derivatives. Repeated terms share compiled readout graphs.
Global phase is absent from expectations; its derivatives remain zero.

When a separator is too large, exact index slicing trades additional work for
smaller live tensors. Every binary assignment is evaluated and summed. Budgets
can reject paths that require excessive slicing or arithmetic. Profiles report
tensor counts, planned intermediates, slicing and estimated complex arithmetic.
`TensorNetworkPlan` streams slices locally. `DistributedTensorPlan` distributes
intermediate tensors across a Torch DeviceMesh; see [distributed execution](distributed-execution.md).

The automatic CPU planner considers circuit contraction when other structured
engines cannot serve the query, and compares its working model with dense
execution. CUDA and native training admission can use it when symbolic Pauli or
independent-factor execution is unsuitable. Automatic quantum execution compares
feasible cost/quality quotes. Explicit native contraction requests
preserve their framework differentiation contract.

Current circuit graphs cover unitary IR with zero-state inputs and declared
Hermitian readouts. Noise, reset and feedback keep the existing component/density
engines. Contraction cost can grow exponentially with network width; arbitrary
strong entanglement does not acquire a polynomial classical runtime.

## Physical quantum trees

`oq.torch.TreeQuantumLayer`, `oq.jax.TreeQuantumLayer` and
`oq.flax.TreeQuantumLayer` map `(..., num_qubits)` to `(..., num_outputs)`.
Each output owns a tree. Leaves encode input angles with RY; every internal node
applies a two-qubit unitary and traces out its second output.

The node uses local ZYZ rotations, a Cartan XX/YY/ZZ block and a local rotation on
the retained output. A local rotation on the discarded output cancels under the
trace, removing three irrelevant SU(4) parameters. Twelve node angles therefore
cover this unitary pooling family. Learned spherical axes define bounded root
readouts. The complete model can still have parameter-coordinate redundancies.

All node contractions use three real Bloch coordinates per live wire. Pauli
transfer is bilinear in child coordinates, retaining quantum coherence and the
effects of unitary-generated entanglement. It does not replace quantum pooling
with postselection or unconstrained classical tensor normalization. Tree depth
is `ceil(log2(n))`; real arithmetic and storage scale with batch × trees × sites.
The virtual quantum edge dimension is two, with operator dimension four. More
trees add readout capacity; they do not turn this family into a universal circuit
simulator. Initial Cartan angles near pi/4 provide balanced partial-SWAP pooling.

`to_spec()` exports an ordinary quantum forest in portable IR. A Torch/JAX tree
checkpoint can be loaded by the existing generic `QuantumLayer` for compatible
QPU execution or native circuit contraction. Export uses n × trees physical
wires and has its own IR/observable memory admission. Native tree evaluation
directly contracts this structured family and retains higher derivatives.

## MPO projections and joint training

`TensorTrainLinear` implements the usual `x @ W.T + bias` through cores shaped
`(left_rank, input_mode, output_mode, right_rank)`. Parameter count is
`sum(r[k] * i[k] * o[k] * r[k+1])`. Input/output modes can be specified; automatic
modes use powers of two, zero-pad inputs and crop outputs in a fixed order.
The planner compares both sweep directions and bounds native batch chunks.
Runtime depends on contraction intermediates as well as parameter count.

`from_dense` performs an explicit offline NumPy TT-SVD. Torch returns a module;
JAX returns `(layer, parameters)` and Flax returns `(module, variables)`. Keeping all
singular values introduces no intentional rank truncation. With a rank cap or requested
relative error, the retained cores have Frobenius error bounded by
`sqrt(sum_k discarded_singular_values_k_squared)`. This also bounds output error
by `bound * ||input||_2`. An insufficient rank cap rejects a requested tolerance.
The bound excludes numerical roundoff; rank compression need not help an
unstructured matrix. Import reports describe initial factors, not parameters
after subsequent learning. Rank/coordinate changes require new optimizer state.

`TensorNetworkAdapter` composes an MPO encoder, a quantum tree and an MPO
projection in a small nonzero learned residual. It preserves arbitrary leading
dimensions, including token sequences. Torch owns ordinary module parameters;
JAX exposes ordinary PyTrees; Flax owns Linen parameters. Adam/SGD/Optax update
the entire hybrid model. There is no SVD in their forward/backward computation.
Quantum actor-critic modules infer the tree's declared readout width, allowing a
compact quantum representation independent of input qubit count.

## Research basis and implemented scope

- [Structured Unitary Tensor Network Representations for Circuit-Efficient
  Quantum Data Encoding, v2, October 2026](https://arxiv.org/html/2602.16266v2):
  unitary-aware tensor/core construction and executable quantum operators.
  OpenQML uses an independently implemented qubit pooling architecture.
- [Entanglement is Half the Story: Post-Selection vs. Partial Traces,
  May 2026](https://arxiv.org/abs/2605.02385): the distinction between classical
  tensor contraction, postselection and deterministic physical quantum maps.
- [TensorCircuit-NG, 2026](https://arxiv.org/abs/2602.14167): tensor-native
  differentiable execution and contraction slicing. Distributed simulation,
  analog dynamics and its reported model-level results are outside this scope.
- [Hyper-optimized tensor network contraction, Quantum 5, 410
  (2021)](https://quantum-journal.org/papers/q-2021-03-15-410/): optional cotengra
  contraction search, with current APIs checked against its official documentation.
- [cuTensorNet 26.09 contraction optimization](https://docs.nvidia.com/cuda/cuquantum/26.09.0/cutensornet/overview.html):
  memory-constrained paths, interleaved slicing/reconfiguration and constant-input
  reuse. OpenQML's portable executor implements bounded local tree rotations,
  compiled native GEMMs and forward-local subtree caching.
- [cotengra tree surgery](https://cotengra.readthedocs.io/en/main/trees.html):
  contraction-tree topology, depth-first paths, slicing and subtree modification.
- [Oseledets, Tensor-Train Decomposition, SIAM Journal on Scientific Computing
  33(5), 2295–2317 (2011)](https://epubs.siam.org/doi/10.1137/090752286): TT-SVD
  factorization and truncation error control.

