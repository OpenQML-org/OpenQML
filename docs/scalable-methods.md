# Sparse registers and resource-driven computation

OpenQML has no fixed total circuit-width ceiling. `Circuit(n)` stores a declared
width, not an array of all wires. Dense vector/density execution, backend bridges,
native verification and semantic certification default to `max_qubits=None`.
Optional user width caps are still enforced. Memory/work, operation, expression,
branch and matrix-payload budgets remain independent resource controls. An
external SDK/device can impose its own capacity.

Removing a width cap does not make a full entangled state inexpensive. A full
complex128 vector has `16 * 2**n` output bytes, and a full density matrix has
`16 * 4**n`. Resource rejection happens before state allocation. Automatic plans
never disguise a shot estimate or intentional truncation as exact execution.

## Sparse observables and noisy Heisenberg propagation

`SparsePauliString` stores only named sites; `SparseHamiltonian` combines such
terms. Their storage follows the number of specified sites, not the declared
register size. Numeric propagation compacts active wire indices before creating
binary masks, so a gate on wire `10**12 - 1` does not create a trillion-bit mask.


The Heisenberg engine supports named unitary gates, reset, non-selective
measurement, named local noise, and budget-admitted matrix/Kraus transfer maps.
Classical feedback is excluded; use component execution or the density engine.
It returns zero-state expectation values, not samples or a state vector. Global
phase is unobservable. It accepts one scalar binding at a time; ordinary native
autodiff APIs remain separate. All named channel probabilities are checked even
when the observable is identity or outside that channel's cone.

Approximation is requested explicitly through `max_error`, `max_terms` and
`coefficient_cutoff`.


Equal Pauli words are merged before pruning. At each pruning event the sum of
absolute discarded coefficients is charged to that output's error budget.
For Hermitian discarded operator `D = sum(c_P P)`, `||D|| <= sum(abs(c_P))`.
The adjoint of a CPTP channel contracts Hermitian operator norm. Consequently,
triangle inequalities give a deterministic absolute readout truncation bound
equal to the accumulated discarded L1 mass. Non-unital channels can increase
coefficient L1 norms; this does not invalidate the operator-norm argument. No
random-circuit hypothesis is used. Bounds exclude floating-point error and the
tolerance used to accept supplied unitary/Kraus payloads. They may be loose.
The engine rejects a frontier that cannot fit both the resource and error
budgets. It never silently relaxes `max_error`.

The algorithm and per-truncation bound follow
[Rudolph et al., *Pauli Propagation: A Computational Framework for Simulating
Quantum Systems*, Theory Box 3](https://arxiv.org/html/2505.21606v1).
[Angrisani et al., local-noise Pauli propagation](https://arxiv.org/abs/2501.13101)
motivates non-unital channel support. Their polynomial average-case results
require additional ensemble assumptions; OpenQML does not claim those results
for arbitrary circuits.
[Dissipation accelerates quantum and classical simulation of open-system
dynamics](https://arxiv.org/html/2609.40174v1) (September 2026) further motivates
noise-aware compression. Its Lindblad discretization, extrapolation and
high-probability theorems are not implemented by this circuit API.

## Exact quantum/classical component execution

`oq.plan_components(circuit, observables)` decomposes independent factors of a
zero-initialized computation. The dependency graph includes quantum operands,
measurement writes, condition reads, overwritten shared storage and both arms
of every atomic conditional. A shared classical bit joins factors even when
they have no shared quantum wire. Barriers do not create state dependencies.

Each Pauli readout factors into local readouts; Hamiltonian coefficients and
output ordering are retained. Idle X/Y sites give zero, while idle I/Z sites
give one. Repeated local readouts are evaluated once. Unqueried trace-preserving
factors can be omitted. All source parameter names remain binding-owned,
including parameters removed from the readout. Torch/JAX scalar and batched
evaluation retains native higher derivatives. Different initial entangled
inputs and full classical record distributions are outside this contract.

`oq.plan_execution(..., method="components")` selects the representation
explicitly. Automatic execution considers it before allocating a global dense
state when existing structured methods have not been selected. Every queried
component's local state/branch requirements are admitted against the memory
budget. Memory models exclude autodiff tape retention and total process RSS.

## Exact factorized quantum geometry

For a pure product state, `psi = tensor_k psi_k`, its Fubini--Study metric is
`G = sum_k pullback(G_k)`. Shared parameter coordinates add contributions across
factors; the phase cross terms cancel under the Fubini--Study projection.
`oq.quantum_metric` now prepares only differentiated connected factors and
performs tangent/adjoint metric products sequentially. It never constructs a
global dense state or dense metric inside the natural-gradient solver.

This is an exact factorization of the existing
[quantum natural gradient](https://arxiv.org/abs/1909.02108) and
[reverse metric recurrence](https://arxiv.org/abs/2011.02991), not a diagonal or
block-diagonal approximation. Gates connecting two quantum components merge
them. Highly entangled connected factors remain exponentially expensive.
`plan.to_dict()` reports the actual differentiated component sizes and working
model. Global-phase-only parameters correctly have zero metric rows.

## Semantic proofs and import paths

Proofs now compact the union of active quantum and classical resources of both
programs before algebraic normalization, preserving original fingerprints and
register-dimension checks. Idle output quantum wires retain the same identity
map; unused classical output bits retain zero initialization. Total width and
connected-component width have no default cap. `max_component_qubits` remains
an optional user restriction. Exact matrix-unit/column counts are checked
against term/work budgets before exponential allocations.

The built-in OpenQASM parser likewise accepts large literal register declarations
without expanding them. Whole-register gate/guard expansion is bounded before
constructing wire tuples. `max_register_bits` is optional. Dense matrix fallback
and externally supplied matrix payloads keep their separate local memory/entry
controls. An arbitrary Python program still requires a supported circuit IR;
unrepresented hardware context is not converted automatically.

## Evidence and assessed alternatives

`tests/test_sparse_scaling.py` compares all Pauli operators against small full
density references, nonlocal custom Kraus/unitary maps, deterministic truncation
bounds, shared classical storage, second derivatives and full-state finite
differences of the factorized metric. Resource exhaustion is tested separately
from numerical agreement.

[Generalized m-body mean-field projections](https://arxiv.org/abs/2610.22069)
(October 2026) is assessed but not substituted for the exact/default engine:
state-adapted projection changes the approximation contract and requires
covariance/Krylov machinery and independent validation. Distributed tensor
slicing, device-calibrated error mitigation, qudit/analog IR and hardware-only
protocols likewise need representations or hardware absent here. Newer dates
alone establish neither suitability nor superiority.
