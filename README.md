# OpenQML

Differentiable quantum layers for hybrid AI, with a shared circuit IR connecting Qiskit, PennyLane, Cirq and OpenQASM.

## Install

NumPy is the only required dependency; integrations are optional.

```bash
python -m pip install 'openqml[torch]'
# Other extras: jax, flax, qiskit, pennylane, cirq, aer, qasm
```

For a source checkout, use `python -m pip install '.[all,dev]'`.

## Hybrid learning

Use one optimizer for classical parameters and quantum weights, with broadcast batches, input gradients and portable checkpoints.

Compose blocks with `spec.compose(...)` and `checkpoint.compose(...)`. Reparameterize symbols with `spec.reparameterize(...)`; embed constants with `spec.bind(...)` and `checkpoint.freeze(...)`.

```python
import torch
from openqml.interfaces.torch import QuantumLayer

model = torch.nn.Sequential(
    torch.nn.Linear(8, 4),
    QuantumLayer.data_reuploading(4, depth=2, batch_chunk_size=16),
    torch.nn.Linear(4, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
model(torch.randn(32, 8)).square().mean().backward()
optimizer.step()
```

JAX exposes `layer(weights, inputs)`; Flax uses standard parameter trees and Optax. Generated weights accept broadcastable `(..., n_weights)` arrays through PyTorch's `layer.functional(weights, inputs)`, JAX's call, or Flax's `weights=`. Inputs and weights remain differentiable.

`batch_chunk_size` bounds forward/adjoint batches; `gradient_chunk_size` bounds parameter-shift jobs, including under `vmap`. External bridges support standard PyTorch/JAX JVPs, VJPs, Jacobians and nested `vmap`; products stream without a full Jacobian. JAX `linearize`/`linear_transpose` builds fixed-point normal operators for hybrid models; see `examples/hybrid_linearization.py`. Nonlinear higher derivatives require native tensor execution. For native unitary circuits, `checkpoint_interval` trades recomputation for fewer retained training tensors.

For unitary simulation, `backend="numpy", gradient_method="adjoint"` reuses bounded batches. `AdjointPlan` supports VJPs and directional JVPs with state-vector storage independent of gate count.

`execution_method="pauli"` compiles supported unitary expectations into real arithmetic without state vectors. `PauliPlan` also evaluates NumPy arrays. `pauli_options` bounds symbolic growth; exhaustion raises diagnostics without approximate truncation. Growth can be exponential.

`execution_method="planned"` factors readouts and selects exact Pauli or dense engines. Repeated dense functions execute once; matching blocks pack parameter lanes into native batches. Readouts run only on requesting lanes. `ExpectationPlan` exposes provenance, costs, rejection reasons, `dense_batches` and `ReadoutRoute` destinations. `planning_options` bounds search, groups, dense width and `max_dense_batch`; packing adapts to working memory. Native PyTorch/JAX/Flax retain higher derivatives and source checkpoints. Memory estimates exclude autodiff retention; selection uses a static heuristic. `plan.verify(...)` compares expectations against bounded full-source NumPy, Qiskit, Cirq or PennyLane execution.

## Circuit interoperability

Import native programs with `convert(program, target="ir")`. Inspect compatibility before loading a target SDK, then export the prepared plan:

```python
from openqml import Circuit, Parameter, plan_conversion

circuit = Circuit(2).ry(Parameter("theta"), 0).cx(0, 1)
plan = plan_conversion(circuit, target="openqasm3")
print(plan.to_json())
result = plan.export()  # Raises ConversionError for reported incompatibilities.
print(result.value)
```

Plans report resources, required extras and deferred synthesis. Conversion, `QuantumProgram.convert(...)` and the CLI expose operation, nesting and output-statement limits, enforced before built-in import expansion. Single-qubit matrices lower with NumPy while retaining phase; OpenQASM 2 phase loss is rejected.

`ConversionPlan.verify(bindings)` checks export/reimport over a complete input basis, including unitary phase, channels and final classical records. `mode="native"` independently executes the exported SDK artifact; OpenQASM uses Qiskit's frontend. Native checks cover unitary programs and channels/reset through spanning physical inputs; recorded measurement and feedback require round-trip mode. Checks are bounded and pointwise in parameters. `ConversionCheck.raise_for_errors()` rejects numerical mismatches.

IR readers check byte, operation, nesting, expression and matrix budgets before constructing circuits. Import settings belong in `source_options`; portable layer and checkpoint readers accept matching matrix limits.

IR binding and renaming reuse immutable matrix payloads. `reduce_readout=True` compacts causal cones; `plan_readout(...)` exposes wire/operation maps. Compiled expectation methods preserve declared readouts from the all-zero state, retain source ownership and reject effects. Full-program export retains the source circuit.

The IR preserves parameter expressions, wire order, supported channels, measurements and classical branches. General runtime classical computation, pulse schedules and some native control flow remain outside the portable subset. Simulator execution has explicit resource budgets.

## Development

```bash
python -m pytest -q
ruff check . && ruff format --check .
python -m build && python -m twine check dist/*
```

Examples and benchmarks are in `examples/` and `benchmarks/`. Alpha software, licensed under Apache-2.0.
