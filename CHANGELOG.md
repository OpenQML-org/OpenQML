# Changes

## 0.31.0

- `qml/lindblad.py`: add controlled open-system dynamics with exact local
  completely positive maps, symmetric splitting, a trace-norm discretization
  bound, streaming state and masked time series. The implementation applies
  local operators without building a dense Liouvillian.
- `qml/interfaces`: add trainable PyTorch and Flax Lindblad layers, positive
  dissipation rates, checkpointed time evolution and native higher derivatives.
- `tests/test_lindblad.py`: check analytic decay, independent matrix-exponential
  evolution, physical density constraints, discretization bounds, second
  derivatives, streaming and compiled PyTorch execution.

The dynamical learning interface is informed by Xie et al., *Physical Review
Letters* **136**, 010402 (2026),
[doi:10.1103/7lsx-ssjl](https://doi.org/10.1103/7lsx-ssjl). The current generator
family comprises Pauli Hamiltonians, Pauli jumps and local amplitude damping;
it does not implement that paper's full environment model. Numerical error
bounds exclude floating-point roundoff and errors in the supplied model.
