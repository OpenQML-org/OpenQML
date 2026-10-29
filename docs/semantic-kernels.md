# Exact hybrid conversion validation

OpenQML now validates more conversions by comparing quantum instruments, rather
than requiring measurement/control syntax to match. This is a bounded compiler
component, not a universal translator of Python or a claim of a new theorem.

## Semantics and algorithm

For every input matrix unit `|i><j|`, the kernel computes its coefficient at
every quantum output matrix unit and final classical record. Matrix units span
the full input operator space, so equality covers arbitrary mixed inputs and
inputs entangled with a reference. Classical registers initially contain zero.
Every final classical bit remains observable. Measurement histories that were
overwritten or otherwise become internal are added, preserving their induced
quantum channel rather than comparing trajectories. Conditional bodies are
chosen once, including when the body overwrites its own guard.

The verifier first builds a dependency graph over the union of both programs'
quantum wires, measurement destinations and classical guards. Disconnected
components are independent factors. Each factor is evaluated separately using
sparse maps; idle wires need no matrix allocation. A new cross-wire interaction
or feedback dependency on either side merges the affected factors.

Coefficients are sparse exponential polynomials. Their scalar field is
`Q(zeta_8)`, with `zeta_8 = exp(i*pi/4)` and `zeta_8^4 = -1`, represented by four
exact rational coordinates. Named Clifford+T gates therefore introduce no
floating-point constants. Symbolic rotations use formal exponential exponents;
stochastic channels use polynomials in real expression atoms. Expressions whose
identities are not implemented remain opaque. Damping coherence factors retain
real `sqrt(1-p)` atoms, with exact rational simplification for perfect-square
constants. Arbitrary matrix/Kraus entries retain their stored binary rationals.
The kernel never promotes a floating approximation of pi to mathematical pi.

Sparse entries with the same input/output/record key are added after each step.
The complete resulting maps are compared; hashes identify certificates but do
not decide equivalence. No tolerances, parameter samples, dense numerical
simulation or optional SDKs enter the proof decision.

## API and boundaries

```python
import openqml as oq

reset = oq.Circuit(1).reset(0).measure(0, 0)
feedback = (
    oq.Circuit(1)
    .measure(0, 0)
    .if_else(oq.ClassicalCondition((0,), 1), oq.Circuit(1).x(0))
    .measure(0, 0)
)
proof = oq.prove_equivalence(reset, feedback, max_component_qubits=1)
assert proof.proven and proof.check(reset, feedback)
```

`compile_verified` and `ConversionPlan.certify` use this stage automatically for
effectful programs. Defaults are six quantum wires per connected component,
100,000 polynomial terms per sparse map/local transition table (and combined
comparison maps), 2,000,000 charged work units, and 4096 bits per rational
coefficient or real-atom exponent. The component width can be raised to at most
eight. Budgets are admission limits, not hard process RSS limits: current maps,
transition maps, branch maps and the compared result may coexist.

When the existing unitary normal forms are inconclusive, the same scalar
algebra also compares exact operator columns per independent component. This
supports the universal gate set of Hadamard and arbitrary symbolic rotations
within the component budgets. Strict mode compares operator amplitudes and
global phase; projective mode compares the induced channel. Cross-component
phase compensation and opaque symbolic identities may remain inconclusive.

Symbolic equality holds where the original IR expressions are defined and all
channel probabilities are valid. Circuit and conditional-branch global phases
cancel for instruments. Pure unitary strict proofs still retain global phase.
Unsupported symbolic identities are inconclusive. Budget exhaustion raises
`ResourceError`; neither outcome is a proof of inequivalence. A certificate can
be rechecked against the original source and target fingerprints.

The trusted base includes the IR, adapters, exact gate/channel rules, expression
interpretation and arithmetic implementation. This is not a proof in a theorem
prover. Native SDK/runtime execution remains a separate validation layer, and
the existing native checker still excludes recorded measurement and feedback.
Pulse calibration, external hardware/noise context and arbitrary Python control
flow are not converted. Highly connected programs retain exponential worst-case
cost. Independent-component benchmarks do not establish general thousand-qubit
verification or simulation.

## Research basis and differentiation

The design draws on current research without copying implementations or
claiming its authors' benchmark results:

| Source | Idea used here | Scope of this implementation |
| --- | --- | --- |
| Ke, Li and Li, [Irene: Equivalence Checking of Hybrid Quantum Programs via Structure-Preserving Symbolic Reduction](https://arxiv.org/abs/2609.36065), September 2026 preprint | Stage structural reasoning before observable density-kernel comparison; compare coherent input/output maps and hidden histories | Exact sparse, component-factored kernels; no Irene HPS graph-isomorphism engine, SMT solver or reproduced Irene performance claims |
| Ricciardi et al., [Quantum Circuit Equivalence Checking: A Tractable Bridge From Unitary to Hybrid Circuits](https://arxiv.org/abs/2511.22523), 2025 preprint | Explicit distinction between unitary, hybrid and discarded-output semantics | Direct instrument semantics; no general deferred-measurement lifting or SQbricks implementation |
| Hong et al., [Equivalence Checking of Parameterised Quantum Circuits](https://arxiv.org/abs/2404.18456) | Symbolic parameters require an all-bindings guarantee, separate from testing at selected bindings | Conservative exponential-polynomial identities; no claim to implement the paper's full algorithm |
| Vilmart, [Rewriting and Completeness of Sum-Over-Paths in Dyadic Fragments of Quantum Computing](https://lmcs.episciences.org/13200), Logical Methods in Computer Science 20(1:20), 2024 | Mathematical background for exact dyadic-phase reasoning rather than floating comparison | The present scalar field handles the Clifford+T fragment; this is not Vilmart's complete rewrite system for all dyadic fragments |

These techniques are published and reproducible. Using them alone is not an
exclusive technical moat. OpenQML's concrete differentiation is their integration
into cross-framework conversion, a dependency-light proof kernel, explicit
semantic contracts, recheckable fingerprints and a growing set of adversarial
compiler regressions. Defensibility requires measured coverage on real compiler
outputs and sustained improvements; it cannot be established by this commit.

## Reproducible evidence

`tests/test_exact_instrument.py` checks every named unitary against an independent
numeric matrix implementation, randomized hybrid maps over all input matrix
units, hidden histories, classical-record mismatches, coherence, guard writes,
symbolic noise composition, Kraus maps, wire ordering and resource admission.
`python benchmarks/semantic_kernels.py` writes a timed proof/mutation audit.
Results are local evidence for the recorded cases and machine, not universal
coverage or an industry-leading performance claim.
