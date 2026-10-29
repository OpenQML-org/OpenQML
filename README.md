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

Compose blocks and weights with `spec.compose(...)` and `checkpoint.compose(...)`. `spec.reparameterize(...)` ties symbols or builds nonlinear encoders; `spec.bind(...)` and `checkpoint.freeze(...)` embed constants.

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

`batch_chunk_size` bounds each simulator batch. For native unitary circuits, `checkpoint_interval` trades recomputation for fewer retained training tensors. Native tensor backends support higher derivatives; external numerical bridges support first-order differentiation.

For unitary simulation, `backend="numpy", gradient_method="adjoint"` reuses bounded batches. VJPs use reverse sweeps; JVPs propagate one tangent state. `AdjointPlan` accepts named array columns and retains state vectors independent of gate count.

`execution_method="pauli"` compiles supported unitary expectations into real arithmetic without state vectors. `PauliPlan` also evaluates NumPy arrays. `pauli_options` bounds symbolic growth; exhaustion raises diagnostics without approximate truncation. Growth can be exponential.

`execution_method="planned"` factors each readout, shares local queries and parameter graphs, and selects exact Pauli or dense engines per block. `ExpectationPlan` exposes source provenance, candidate costs and rejection reasons. `plan.verify(...)` checks a concrete binding against bounded full-source execution through NumPy, Qiskit, Cirq or PennyLane. `planning_options` bounds total search, groups and dense width. Native PyTorch/JAX/Flax retain higher derivatives and source checkpoints; working-memory limits exclude autodiff retention. Selection uses a static cost heuristic.

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
