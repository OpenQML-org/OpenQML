# Structured quantum–classical models

External `QuantumLayer` instances support `gradient_method="batched-shift"`
to pack forward readouts and their derivative stencil before the first result.
See [submission-aware training](submission-training.md) for the Jacobian-memory
tradeoff, Torch/JAX transforms and QPU submission accounting.

For KAN, recurrent memory, Born generators, autoregressive sequence models and
quantum-state diffusion, see [Stateful and generative models](frontier-models.md).

The model families use `HybridArchitecture` and share parameter names, depth
curricula, optimizers, checkpoints and execution planning across PyTorch, JAX
and Flax. Constructors return architecture specifications; call
`architecture.build("torch" | "jax" | "flax")` to select tensor ownership.

| Model | Constructor | Input | Default output |
| --- | --- | --- | --- |
| Image convolution | `HybridArchitecture.convolutional(C, D)` | `[B,H,W,C]` | `[B,D]` |
| Temporal convolution | `convolutional(C,D,spatial_dims=1)` | `[B,T,C]` | `[B,D]` |
| Quantum Transformer | `HybridArchitecture.transformer(C,D)` | `[B,T,C]` | `[B,T,D]` |
| Quantum graph network | `HybridArchitecture.graph_network(C,D)` | graph tensor dictionary | `[B,D]` |

Convolution and graph constructors use masked mean readout. Set
`readout_pooling="none"` for per-position/node predictions; `sum` and `first`
are also available. Hidden width stays fixed inside the residual stack.
`depth` controls hybrid depth and `QuantumCellSpec.depth` controls circuit depth.

## Convolution

The same trainable quantum filter is applied to every spatial window. 1D/2D
windows support dilation and same/circular padding; causal padding is available
for temporal models. Residual blocks preserve spatial resolution. A window
projection can be dense or MPO according to the architecture policy.

The default `qcnn` cell alternates shared SU(4) convolution blocks with
controlled pooling. Discarded wires do not participate in subsequent gates or
readouts, implementing a partial trace without postselection. Retained width is
controlled by `bond_qubits`. `hardware-efficient`, `multiscale` and
`residual-channel` cells can also be selected. `QuantumLayer.qcnn(...)` and
`QuantumLayer.feature_circuit(...)` expose the circuit primitives separately.

Window gathering is chunked; it does not unfold the entire image into one large
patch tensor. `OperatorSpec.convolution(chunk_size=...)` controls this workspace.
For causal sequence prediction use `readout_pooling="none"`; a mean readout
summarizes the entire sequence.

## Attention

Independent quantum query, key and value branches share circuit topology, with
separate encoders and variational parameters. Multi-head Gaussian attention
uses `softmax(-||q-k||²/sqrt(d))`. PyTorch expresses this as augmented Q/K/V
tensors and delegates to SDPA, enabling fused kernels when the device and dtype
support them. Queries are chunked to bound forward score workspace.

JAX models admit their own `jax-gpu` candidate from JAX's active platform,
independently of Torch installation. Configure its time/price profile under
`profiles["jax-gpu"]`; this keeps framework-specific observations separate.
Native JAX GPU execution stays in the differentiable JAX graph.

`attention="linear"` uses positive quantum feature maps with
`S=sum(phi(k) outer v)`, `z=sum(phi(k))`, and `y=phi(q) S/(phi(q) z+epsilon)`.
It avoids an `N x N` score matrix. Causal mode uses chunked prefix sums;
noncausal mode uses global sufficient statistics. Linear and Gaussian modes
define different kernels. They are not interchangeable accuracy guarantees.

`causal=True` prevents future-token access. Pass
`{"x": features, "mask": valid_tokens}` for padding (`True` means valid).
All-masked token outputs are zero. Sinusoidal positions are enabled by default;
set `positional=False` when permutation equivariance is required.

## Graph messages

Graph inputs are dictionaries of framework tensors:

| Field | Shape | Meaning |
| --- | --- | --- |
| `nodes` | `[B,N,C]` | Node features |
| `edge_index` | `[B,E,2]` | Signed integer `(source,destination)` indices |
| `edge_features` | `[B,E,F]` | Optional, with `edge_features=F` declared |
| `node_mask` | `[B,N]` | Optional valid nodes |
| `edge_mask` | `[B,E]` | Optional valid edges; required for `-1` padding |
| `positions` | `[B,N,d]` | Required with `geometric=True` |

A shared quantum circuit processes source/destination features, optional edge
features and squared inter-node distance. Sparse sum/mean scatter aggregation
is equivariant to node relabeling; graph readout is permutation invariant.
Distance-only geometry is invariant to translation, rotation and reflection.
The model outputs scalar node features, not updated geometric coordinates.
Duplicate edges contribute separately. Empty edge sets are permitted.

Node and edge counts may change without changing parameter shapes. Graph
batches use padding and masks; no dense adjacency matrix is constructed. Invalid
active indices raise eagerly; inside JAX compilation they produce nonfinite
graph outputs, which the integrated trainer rejects without an optimizer update.

## Shared features and higher-order structure

`HybridArchitecture.transformer(C,D,attention="state")` uses one trainable quantum
encoder per token for similarities and values. `OperatorSpec.state_attention`
controls heads, causality, positions, chunk size and depolarization. The default
QCNN exposes a complete local X/Y/Z readout. Custom cells must expose complete,
unit-coefficient local X/Y/Z triples; several measured qubits are supported.

Each head learns a CPTP Pauli channel through four softmax probabilities, followed
by fixed depolarization. Its kernel is the mean local Hilbert–Schmidt overlap
`k_h(x,y)=mean_q Tr(E_h(rho_q(x)) E_h(rho_q(y)))`. The feature representation
`[1,lambda_x*r_x,lambda_y*r_y,lambda_z*r_z]/sqrt(2Q)` gives this kernel exactly.
Sufficient statistics compute normalized attention without pairwise overlap
circuits or an N×N score matrix. Causal mode uses prefix statistics. Quantum row
evaluations fall from three independent branches to one shared bank, independent
of head count. This changes model capacity and parameterization; it does not
preserve arbitrary independent Q/K/V circuits.

Measured Bloch vectors are projected to the physical unit ball, introducing bias
for noisy estimates. The kernel is positive semidefinite and nonnegative; fixed
depolarization keeps nonempty exact-state denominators positive. Computation
remains hybrid; each head's kernel is defined by local reduced states and
classical sufficient statistics.

`HybridArchitecture.graph_network(C,D,order=2)` lifts sparse graph inputs to
persistent ordered-pair states `[B,N,N,H]`. Initial features include both nodes,
summed edge multiplicity/features, diagonal equality and optional squared distance.
No node identifiers are added. Each block evaluates one circuit per pair and
contracts learned factors as `M_ij,c=sum_k L_ik,c R_kj,c`. The quantum feature bank
feeds both factors and the local branch, avoiding a circuit per `(i,k,j)` triple.
This is a higher-order update, rather than ordinary local edge aggregation.

The default invariant readout concatenates mean pair and mean diagonal features.
`readout_pooling="none"` returns pair predictions `[B,N,N,D]`. Padding, repeated
edges, geometric invariance and parameter ownership follow the shared model.
`input_lift="graph-pairs"` permits pair-graph and pointwise blocks; the first block
defines input edge/geometry fields. `order=1` keeps sparse O(E) messages. Pair
states need O(B N² H) storage; classical contraction costs O(B N³ H). Admission
checks occur before quadratic allocation, with no fixed node-count limit. The
finite-width factorized model does not inherit a full 3-WL theorem. A functional
regression separates C6 from two triangles with identical node features, which
ordinary 1-WL message passing cannot distinguish.

`native_execution="pauli-transfer"` selects exact compiled sparse Pauli transfers
for PyTorch/JAX/Flax cells. All observables share propagation frontiers; topology
is compiled once per execution view and bindings are evaluated in batches.
Compilation and batch memory have explicit limits. Exponentially growing
frontiers are rejected, never silently truncated. QPU routes retain source
circuits and their numerical gradient contract.

Numeric bridge forwards and shifted pullbacks merge identical rows only when the
backend declares `deterministic_expectations=True`. Gradients preserve distinct
input cotangents and accumulate shared weights. Reuse is bounded to each call;
no results survive parameter updates. Finite-shot QPUs keep independent estimates.
Forecasts conservatively budget unmerged rows: duplication is input dependent.

## Composition and extensions

Pass a tuple of `OperatorSpec` objects to `HybridArchitecture(operators=...)`
to mix pointwise, convolution, attention and graph blocks. Operators preserve
hidden shape and use the existing gated residual/normalized SwiGLU program.
Graph–attention composition should use `positional=False` to preserve symmetry.

Subclass `NetworkOperator` and register it with
`register_network_operator("organization.operator", implementation)`.
Implement `validate`, `input_width`, `roles`, `declare`, `apply`,
`validate_metadata` and `row_chunks` as needed. The context exposes `kernel`,
`project`, array helpers and block parameters. Quantum roles get independently
owned encoders/weights. Extra parameters use the declaration callback and join
the same optimizer groups and progressive state. `list_network_operators()` lists
registered names. Names cannot replace existing registrations.

`QuantumCellSpec.from_spec(layer_spec)` accepts a portable `LayerSpec`, including
circuits imported with `LayerSpec.from_program(...)` from supported quantum SDKs.
Fixed quantum filters may have an empty variational parameter list. Custom
cells serialize circuit IR and observable ownership as JSON. Operator code is
never stored in architecture metadata; register the same name/version before
loading a custom operator. Default 0.9 architecture fingerprints are preserved.

## Training and planning

All families use the [shared trainers](deep-learning.md). Data dictionaries and
labels must share batch dimension `B`. The default input selector recognizes
`nodes`, `x`, `inputs` or the first feature tensor; provide `input_selector` for
custom loss batches. JAX/Flax losses return one scalar per batch example.

For explicit planning, pass `metadata=architecture.input_metadata(inputs)` and
`microbatch_samples=...` to `architecture.schedule(...)`. Integrated trainers
do this before each update. Plans count windows, tokens, edges, independent
Q/K/V calls, tail chunks, provider limits and gradient submissions. Native
checkpoint replay is included in the default gradient work estimate. A routing
plan constructed without metadata assumes a bootstrap workload, not a measured
whole-dataset forecast. Forward workspace checks exclude autodiff tapes and
framework allocator reservations.
