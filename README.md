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

Use one optimizer for classical parameters and quantum weights. Layers support arbitrary leading batch dimensions, input gradients, portable checkpoints and configurable execution backends.

Compose blocks and trained values with `spec.compose(...)` and `checkpoint.compose(...)`, using explicit sharing maps. `spec.reparameterize(...)` ties symbols or builds nonlinear encoders; `spec.bind(...)` and `checkpoint.freeze(...)` embed constants for transfer learning.

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

JAX exposes `layer(weights, inputs)`; Flax uses standard parameter trees and Optax. For classical weight generators and ensembles, pass broadcastable `(..., n_weights)` arrays via PyTorch's `layer.functional(weights, inputs)`, JAX's call, or Flax's `weights=` argument. Both inputs and generated weights receive gradients.

`batch_chunk_size` bounds each simulator batch. For native unitary circuits, `checkpoint_interval` trades recomputation for fewer retained training tensors. Native tensor backends support higher derivatives; external numerical bridges support first-order differentiation.

For unitary simulation, `backend="numpy", gradient_method="adjoint"` reuses prepared batches within memory limits. VJPs use reverse sweeps; JVPs propagate one tangent state across readouts. `AdjointPlan` accepts named array columns. Retained state vectors stay independent of gate count. Effects use the existing differentiation paths.

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

Plans report source and prepared resources, required extras and deferred SDK synthesis. Conversion and `QuantumProgram.convert(...)` accept `max_operations`, `max_depth` and `max_output_statements`; the CLI exposes matching flags. Global limits constrain built-in importers before expansion. Single-qubit matrices lower with NumPy and retain global phase; OpenQASM 2 phase loss is rejected.

IR readers check byte, operation, nesting, expression and matrix budgets before constructing circuits. Import settings belong in `source_options`; portable layer and checkpoint readers accept matching matrix limits.

IR binding and renaming reuse validated immutable matrix payloads.

For sparse unitary readouts, `reduce_readout=True` on PyTorch, JAX and Flax layers removes gates outside the causal cone and compacts simulator wires. Shapes and checkpoints keep source order; eliminated columns receive zero gradients. Native tensor and NumPy backends support this explicit pass. `plan_readout(circuit, observables)` reports resources, wire/operation maps and remapped readouts for conversion. Equivalence covers declared expectations from the all-zero initial state; full-program conversion keeps the source. Channels, measurements and feedback are rejected.

The IR preserves parameter expressions, wire order, supported channels, measurements and classical branches. General runtime classical computation, pulse schedules and some native control flow remain outside the portable subset. Simulator execution has explicit resource budgets.

## Development

```bash
python -m pytest -q
ruff check . && ruff format --check .
python -m build && python -m twine check dist/*
```

Examples and benchmarks are in `examples/` and `benchmarks/`. Alpha software, licensed under Apache-2.0.
