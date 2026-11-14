# Changes

## 0.31.0

- `qml/lindblad.py`: add controlled open-system dynamics with exact local
  completely positive maps, symmetric splitting, a trace-norm discretization
  bound, streaming state and masked time series. The implementation applies
  local operators without building a dense Liouvillian.
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
- `docs`: describe all three interfaces, their input domains, approximation
  bounds, resource limits and method sources. Remove an obsolete store path
  from the immutable-continuation compiler.
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
