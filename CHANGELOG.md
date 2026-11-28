# Changes

## 0.41.0

- Train Born classifiers with the likelihood of the measured class and return
  their native probabilities from eager and streamed prediction. Accept single
  images and sequences for convolution, attention and quantum-memory inference.
- Keep masked attention and zero-probability losses finite in native training.
  Preserve mixed-precision tensor placement, initial-state tangents and
  scheduling indices after Pareto pruning.
- Compute the exact scalar scenario score from cumulative masses in linear
  space. Preserve close joint atoms, rare marked payoffs, and capped portfolio
  budgets without cancellation. Fix Torch expectiles and turnover at zero equity.
- Restore interoperability exports and loop dependencies. Preserve inverted
  Cirq records and Qiskit sampler classical stores. Check register addresses and
  nested import budgets, and compile Qiskit register broadcasts from Python source.
- Keep unsupported CUDA probes on the CPU route and sample large statevectors
  with a cumulative distribution. Retain empty-batch shapes and conserve
  measurement shot budgets. Preserve provider failure counts and release a
  reservation when the backend confirms that it submitted no job.

## 0.40.0

- Check dynamic OpenQASM 3 addresses and Qiskit indexed reads and stores at
  execution when a static range proof is unavailable. IR v7 carries the error
  branch and continues to read v1–v6 circuits.
- Exchange measurement-based Cirq repeat-until loops and resolve deterministic
  runtime control before exporting PennyLane tapes or OpenQASM 2. A portable
  program retains dynamic behavior for execution through OpenQML.
- Sum terminating feedback paths over an absorbing quantum instrument. NumPy
  verifies the instrument on a complete operator basis. Torch and JAX compute
  native density outputs and gradients and sample the final classical record.
- Compile bounded Python circuit builders with explicit entry arguments.
  Circuit factories can also be captured through the conversion interface.

## 0.39.1

- Fix exact density simulation and conversion verification for finite
  measurement-controlled loops that revisit a classical record with a
  different quantum state.
- Verify that automatic OpenQASM import preserves and executes runtime while
  loops, restoring the PyPI release check.

## 0.39.0

- Preserve runtime while loops and nested break/continue in Qiskit, OpenQASM 3
  and IR v6. Existing circuits without runtime control still serialize as v5.
- Lower range-checked runtime indices on OpenQASM gates, measurements and
  classical stores. Qiskit indexed writes resolve the address before mutation.
  Grouped switch cases support unsigned selectors and a default arm.
- Sample measurement-dependent loops locally and check finitely resolved
  conversion round trips over a complete operator basis. Recurrent paths and
  exhausted budgets produce explicit errors.
- Correct the README quickstart check that blocked the previous release.

## 0.38.0

- Simulate named Clifford circuits with a packed CPU stabilizer engine in the core installation. Sample terminal measurements directly, including projected and overwritten classical bits, and evaluate weighted Pauli readouts.
- Fuse runs of diagonal gates across up to eight wires in the NumPy statevector simulator. Parameter sweeps reuse the same structural plan.
- Use the same Clifford conjugation rules for exact conversion proofs and simulation. Skip redundant lowering when the backend accepts the full gate set.
- Accept Pauli words such as `"ZZ"` in expectation, batch and planning APIs. The quickstart now runs with a core install.

## 0.37.0

- Add paired, budget-aware selection for fitted hybrid and classical predictors.
  Deployment uses the best eligible classical predictor until held-out losses
  support the requested improvement against each registered baseline.
- Compute exact two-wire Z correlations in sparse two-body IQP models with
  NumPy, PyTorch or JAX. Prepared readouts share index groups across batches.
- Accept single-asset forecast labels with shape `[batch, time]` and individual
  sequences at inference. Apply the configured observation gap at the inner
  validation boundary during walk-forward fitting.
- Add a portfolio decision method that keeps the classical incumbent on a tie,
  a worse sampled objective or a sample with no feasible holdings. Exact
  fixed-cardinality search scores holdings or omissions in bounded batches.
- Import symbolic Cirq FSim, fractional iSWAP and phased-X gates through
  differentiable IR rotations. Include controlled-gate phase corrections in
  the import operation budget.

## 0.36.1

- Require a NumPy 2-compatible PyTorch version in the installation extras.
  On Intel macOS, keep NumPy 1.x for PyTorch and allow NumPy 2 wheels for the
  core installation on Python 3.13.
- Accept NumPy bindings in PyTorch simulation and preserve gradients of native
  bindings in the same circuit. Keep empty JAX circuits on the requested device.
- Broadcast explicit batch shapes in automatic expectation readouts and report
  missing PyTorch with the installation command during circuit compilation.
- Fix model input placement and distributed finite-gradient rejection on JAX
  0.4.30. Infer a single regression output when targets match the sample and
  sequence axes. Mark optional checks so the core-only suite can run on its own.

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
