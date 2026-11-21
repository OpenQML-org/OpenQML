# Changes

## 0.36.0

- Dense statevector readouts visit each Hermitian amplitude pair once for wide
  off-diagonal Pauli words. Eligible CUDA tensors use a strided Triton reduction
  and a direct PyTorch pullback in complex64 or complex128.
- Exact MPS updates use an identity left factor or a reduced QR. The planner
  reports `none_full_bond_retained`, estimates split work and reserves the
  Schmidt SVD for truncation and its existing error certificates.
- NumPy adjoints group Hamiltonian terms by their bit-flip masks. Each gate
  shares one local cotangent across active angles in reverse mode and applies
  one combined directional derivative in forward mode.
- Consolidate optimizer quickstart, namespace and framework setup checks.
  Rewrite the README around installation, simulation, training and allocation.

## 0.35.1

- Hardware-efficient and data-reuploading layers accept a logical wire
  permutation for their CX gates through the core, PyTorch, JAX and Flax
  constructors. Feature, weight and output coordinates stay in logical order.
- A six-qubit circuit checks SDK conversions, hybrid gradients and discounted
  payoff readout against an independent 64-amplitude state.

## 0.35.0

- CUDA gate application reads strided state and matrix views directly. A fused
  first-order gate-gradient kernel reduces spectator amplitudes and shared
  parameter batches for ordinary and reversible PyTorch backward passes.
- MPS simulation can choose a wire layout from the circuit interactions and
  reuse exact left and right environments across local Pauli measurements.
  Readout reports expose the number of site transfers.
- The contraction portfolio minimizes modeled peak storage over depth-first
  orders of each candidate tree and scores slicing decisions over the full path.
  Cached, flat and distributed executors share the compiled slice selectors.
- CPU and native natural gradients share one residual-checked solver. An
  optional rank-bounded Nyström preconditioner also serves quantum Fisher
  optimization. Its metric product count includes setup.

## 0.34.0

- The MPS simulator propagates Pauli supports across routed bonds, allocates
  Schmidt loss against each requested readout tolerance and reports the achieved
  absolute errors. Prepared causal cones retain the mapping to source wires.
- Hybrid layers and saved weights expose pairwise class margins and
  occurrence-wise parameter-shift gradients with truncation bounds. Shared
  Pauli terms cancel before simulation.
- Marked-oracle pricing converts a discounted price tolerance into a local
  readout tolerance. Walk-forward fitting admits output storage before
  scanning the data and validates noncontiguous inputs in bounded chunks.
- Certified program execution preserves final classical assignments, terminal
  dephasing, Cirq measurement maps and analytic PennyLane Pauli returns.
- Layer specifications and checkpoints share bounded JSON I/O. Portable dense
  simulation and native device gradients share their local matrix layout.

## 0.33.0

- `sim/_triton_kernels.py` implements CUDA statevector updates on one to
  three ordered wires and direct complex contractions over short binary bonds.
  The optional `gpu` extra installs the compiler. The Qiskit simulator
  adapter retains its CPU methods.
- `sim/_fusion.py` schedules ready gates by wire dependencies before local
  fusion. `sim/_reversible.py` adds on-device inverse-state reverse products
  for batched PyTorch and JAX circuits, nonlinear angles and global phase.
- `sim/_mps.py` routes nonadjacent gates, tracks logical SWAPs and supports
  canonical Schmidt truncation through `openqml.sim.MPSBackend`, with
  per-observable truncation bounds. Exact
  readouts retain full singular columns. `finance.price` propagates the
  truncation bound to discounted direct-oracle payoff intervals.
- `runtime/environment.py` reports the optional kernel compiler and preserves
  detection of the CPU simulator. GPU selection uses `backend="gpu"`.

## 0.32.0

- `sim/pauli_transfer.py` compiles five local noise channels, reset and
  discarded measurements into trainable sparse Pauli transfers. Batched
  NumPy, PyTorch and JAX evaluations share the same prepared frontiers.
- `finance/entropic.py` adds finite-support EVaR, least-favourable scenario
  probabilities and implicit sensitivities. Scenario models report asset-level
  risk and deviation contributions. Execution budgets and conditional trees
  accept `risk_measure="evar"`, including fees and quadratic impact in funded trades.
- `ir/phase_folding.py` merges symbolic diagonal phases on affine parities,
  preserving complemented-wire and branch-local phases. Enable it through
  conversion, circuit optimization or native layer optimization.
- `autodiff/optimization.py` tracks classical stores when folding later guards.
  The conversion interface also accepts the existing Pauli-fusion options.


## 0.31.0

- `qml/lindblad.py`: add controlled open-system dynamics with exact local
  completely positive maps, symmetric splitting, a trace-norm discretization
  bound, streaming state and masked time series. The implementation applies
  local operators without building a dense Liouvillian.
  Commuting supported generators use exact composition without repeated
  splitting, including mutually commuting Pauli dissipators and disjoint decay.
- `qml/interfaces`: add trainable PyTorch and Flax Lindblad layers, positive
  dissipation rates, checkpointed time evolution and native higher derivatives.
- `qml/lindblad.py`: propagate absolute generator-error envelopes into final
  trace-norm and observable bounds, with initial-state and numerical errors.
- `finance/tail.py`, `_tail_linear.py`, `_torch_tail.py`: jointly optimize
  multi-period softplus CVaR, Wasserstein robustness and rebalancing using
  matrix-free Newton/CG and implicit first/second derivatives. Report exact
  empirical tail risk and checked stationarity. Connect learned path weights
  through `ScenarioPlan.tail_decision`.
- `ir/core.py`, `sim/planning.py`, `sim/density.py`: introduce version-5
  classical assignments, preserve version-1–4 readers, and merge overwritten
  storage in eager and compiled density programs without additional qubits.
- `interop/adapters`: lower typed Qiskit variables, stores and finite-loop
  exits with shared assignment snapshots and reusable loop flags. Reject
  undefined inputs and uninitialized reads. Preserve stores through Qiskit,
  JSON and OpenQASM 3 and diagnose unsupported targets.
  Keep the existing loop-note hook signature while reporting added storage
  separately, preserving importer-budget integrations.
- `docs`: describe all three interfaces, their input domains, approximation
  bounds, resource limits and method sources. Remove an obsolete store path
  from the immutable-continuation compiler.
- `MANIFEST.in`, `README.md`: include the change history in source distributions
  and link it from the package description.
- `tests/test_lindblad.py`: check analytic decay, independent matrix-exponential
  evolution, physical density constraints, discretization bounds, second
  derivatives, streaming and compiled PyTorch execution.

The dynamical learning interface is informed by Xie et al., *Physical Review
Letters* **136**, 010402 (2026),
[doi:10.1103/7lsx-ssjl](https://doi.org/10.1103/7lsx-ssjl). The current generator
family comprises Pauli Hamiltonians, Pauli jumps and local amplitude damping;
it does not implement that paper's full environment model. Numerical error
bounds exclude floating-point roundoff and errors in the supplied model.

Generator uncertainty follows the modeling direction of Kraft et al., *PRX*
**16**, 031037 (2026), [doi:10.1103/s96t-n8tx](https://doi.org/10.1103/s96t-n8tx).
Bounds are conditional on caller-supplied error envelopes. The financial layer
uses the decision-focused optimization direction of
[ICML 2024](https://proceedings.mlr.press/v235/ma24j.html) and
[ICML 2026](https://proceedings.mlr.press/v306/jeon26a.html), with a continuous
path-tail objective rather than either paper's discrete selection model.
Classical lowering uses assignment-by-copy semantics from
[OpenQASM](https://openqasm.com/language/classical.html) and bounded control/data
separation as discussed in [QuCtrl-BELL](https://arxiv.org/abs/2605.22433).
