# Research methods across OpenQML

This release reviews the library by computational responsibility. It applies
research methods where the IR, numerical representation and validation support
them. Newer publication dates alone do not establish a better replacement.
No benchmark here establishes overall state-of-the-art performance or an
exclusive technical moat.

| Library responsibility | Implemented method / decision | Evidence and limits |
| --- | --- | --- |
| Framework frontends, IR, symbolic expressions and serialization | Explicit circuit semantics, preserved register/parameter ownership and bounded parsing; exact conversion checks | Circuit objects are translated; arbitrary surrounding Python, pulse calibrations and external hardware context remain outside the IR. Replacing parsers with a quantum algorithm would add no scientific benefit. |
| Compiler, simplification and control-flow optimization | Algebraic phase polynomials, exact word rewrites, Clifford conjugation, exact operator and instrument kernels; new `optimize_verified` checks optimized IR before returning it | [Semantic kernel research record](semantic-kernels.md), including Irene (2026). Hard identities can remain inconclusive; bounded failure never becomes a proof. |
| Hybrid model/layer definitions and numerical gradients | Existing adjoint/JVP/VJP engines plus new `QuantumMetricPlan`: tangent forward and projected adjoint reverse products, damped matrix-free CG natural gradients | Pure states, unitary circuits, CPU state vectors. Shared/nonlinear angle chain rules and global-phase gauge cancellation are checked independently. `LayerSpec.metric_plan()` selects ordered trainable weights. |
| Finite-shot readouts and sampler providers | Shared commuting groups and covariance; new independent-pilot covariance allocation with exploration floors | Pilot shots count toward the total budget and are never reused in production means. Sum-of-output variance is the allocation objective. Conditional unbiasedness assumes the sampler executes the intended ideal measurement; hardware bias is not removed. |
| General Pauli/Hamiltonian measurement reuse | New correlated one/two-site locally optimal dual-frame shadows, pilot tomography, positive regularization and greedy mutual-information pairing | June 2026 local-dual research, below. The dual reconstruction identity is checked; production data can estimate additional readouts without another quantum call. Pair matching is bounded and greedy, not the paper's complete graph partitioner. |
| Dense state-vector/density simulation, noise and measurement branches | Existing exact numerical engines, matrix-unit verification, branch liveness and explicit resource checks | General entangled dense states remain exponential. New sparse-tensor/distributed research is assessed below, not advertised as implemented. |
| Tensor-network simulation | Existing full-SVD MPS plus new opt-in canonical Schmidt truncation with rank cap and accumulated state/observable error bounds | Nearest-neighbor one/two-wire unitary circuits. Canonical QR sweeps precede truncation. Bounds exclude floating-point error; exact execution remains the default. Standard parameter-shift gradients of nonlinear truncation are rejected. |
| Pauli/stabilizer/readout engines | Existing bounded Heisenberg/Pauli propagation, shared expression graphs, Clifford tableaux and causal readout reduction | Structure-specific exact methods. Large structured widths do not imply general large-qubit dense simulation. |
| Execution selection, scheduling, batch reuse and memory planning | Existing admission-aware method selection and reuse; approximate MPS must be selected explicitly | Deterministic execution, approximate numeric execution and noisy sampling keep different contracts. Memory admission is not a hard process-RSS guarantee. |
| Checkpoints, interfaces, CLI, package and platform infrastructure | Preserve portable model/parameter ownership, keep heavyweight SDKs optional, test core/full installations independently | These are engineering contracts rather than opportunities to insert unrelated quantum research algorithms. |

## Quantum geometry

[Quantum Natural Gradient](https://arxiv.org/abs/1909.02108v3), published in
Quantum 4, 269 (2020), defines the real quantum geometric/Fubini--Study metric.
[Efficient classical calculation of the Quantum Natural Gradient](https://arxiv.org/abs/2011.02991v1)
develops constant-state-buffer reverse recurrences. OpenQML combines directional
state derivatives with the projected adjoint contraction
`Re[J† (I - |psi><psi|) J v]`, then solves `(G + damping I) d = gradient` with
linear conjugate gradients. It does not form a parameter-by-parameter metric
inside the solver. Matrix construction remains an explicitly bounded diagnostic.

The solver reports iterations, product count, convergence and a recomputed
residual. A damped solve is not a guarantee of loss convergence or freedom from
barren plateaus. This is the real Fubini--Study metric, not four times that metric
(the pure-state quantum Fisher matrix).

[Modified Conjugate Quantum Natural Gradient](https://link.springer.com/article/10.1140/epjqt/s40507-025-00432-4)
(2025) uses nonlinear search-direction/step adaptation. It is distinct from
the linear CG system solve implemented here and is not claimed as implemented.
Recent mixed-state natural-gradient work also has different state preparation
and metric requirements; it is not silently substituted for the pure-state API.

## Correlated shadows and adaptive shots

[Improving shadow estimation with locally-optimal dual frames](https://arxiv.org/html/2511.02555v2)
(v2, June 2026) motivates correlated block reconstruction under local measurements.
OpenQML independently implements bounded one/two-qubit blocks. Pilot Pauli
tomography is projected to a positive state and regularized; mutual information
selects disjoint pairs. With local POVM frame `A` and positive reference outcome
probabilities `p`, reconstruction coefficients solve
`D = diag(1/p) A (A.T diag(1/p) A)^-1`.
The implementation checks `A.T D = I`; the identity column is a constant
estimator. Tensor products of valid block duals reconstruct Pauli readouts even
when the reference state differs from the measured state.

Pilot and production streams are independent. Production records alone determine
means and sample covariance, conditional on the fitted model. `reestimate()`
reuses these records for additional Pauli/Hamiltonian observables. The code does
not copy the paper's implementation or claim its 40-qubit/molecular results.
Correlations spanning selected blocks can limit variance improvements. Resource
limits bound tomography, unique measurement settings, records and postprocessing.

[AQUIRE](https://arxiv.org/html/2605.00682v1) (May 2026) studies Bayesian adaptive
observable estimation including qudits and device errors. OpenQML's separate
`estimate_adaptive` uses a simpler covariance-based Neyman allocation with an
independent pilot and positive exploration floor. It is not the full AQUIRE
protocol and does not claim hardware error mitigation.

## Controlled MPS approximation

The Schmidt/MPS approach traces to
[Vidal, Phys. Rev. Lett. 91, 147902](https://arxiv.org/abs/quant-ph/0301063v2).
Contemporary emulator implementations likewise distinguish rank caps and
[norm-distance truncation control](https://pasqal-io.github.io/emulators/latest/emu_mps/advanced/mps/).
OpenQML first makes the environment outside a two-site gate orthonormal, then
retains selected Schmidt values and renormalizes. For discarded relative weight
`w`, the normalized step distance is
`sqrt(2*w/(1 + sqrt(1-w)))`. Triangle inequalities accumulate a state-distance
bound `e`; a Hermitian readout's truncation error is bounded by
`min(2, 2*e) * ||O||`. Pauli coefficient L1 norms give a conservative operator
norm bound. QR/SVD and floating-point roundoff are excluded explicitly.

Hard rank and state-error limits can reject execution. The default exact MPS
and adaptive planner never silently introduce this approximation. The new
backend's ordinary parameter-shift gradient bridge is blocked: the truncated
map is nonlinear and need not obey the original circuit's shift identity.

## Assessed methods requiring a different implementation

[TensorCircuit-NG](https://arxiv.org/html/2602.14167v1) (2026) describes distributed
tensor-network slicing and broader analog/qudit/fermionic capabilities.
[Approximate simulation using sparse tensors](https://arxiv.org/html/2602.04011v1)
(2026) introduces a different sparse-tensor representation and contraction
pipeline. Neither is a drop-in improvement to a small dense NumPy kernel;
adoption needs new contraction, differentiation and approximation contracts plus
relevant hardware/benchmarks. This release does not claim either implementation.
Likewise, compiler SMT/e-graph completeness, error-corrected hardware and
device-calibrated mitigation are not inferred from bibliographic citations.

## Reproducible local evidence

Run `python benchmarks/research_methods.py`. The committed
[CPU report](../benchmarks/results/research_methods_cpu.json) records matrix-free
versus explicit metric agreement, finite-shot allocation at a fixed total budget,
exact dual-frame variance on a specified reference case, independent pilot
acquisition, and approximate-MPS readouts versus dense simulation. It separates
known-reference variance calculations from pilot-fitted acquisitions.

`tests/test_research_methods.py` independently verifies derivative geometry,
linear-system solves, gauge/null directions, sampler budgets, dual reconstruction
for all two-site Pauli operators, post-hoc data reuse, approximation bounds,
resource rejection and verified optimization. Results apply to their recorded
cases; they do not establish overall superiority, exclusive rights or quantum
advantage. Differentiation comes from the combined implementation, its explicit
semantic/error contracts and an accumulating compiler/scientific regression set.
