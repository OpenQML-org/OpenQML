# OpenQML

OpenQML is a Python library for training quantum–classical models, working with
portfolio scenarios and moving quantum circuits between frameworks. You can
build and simulate a circuit with NumPy, then use the same circuit in a PyTorch
or JAX model.

[PyPI](https://pypi.org/project/openqml/) · [API reference](https://github.com/OpenQML-org/OpenQML/blob/main/docs/api.md) · [Changelog](https://github.com/OpenQML-org/OpenQML/blob/main/CHANGELOG.md)

## Installation

OpenQML requires Python 3.10 or later.

```bash
python -m pip install openqml
```

The core includes NumPy simulation, portfolio allocation and OpenQASM export.
Install an extra when you need a training framework or a quantum SDK. For
example, `python -m pip install "openqml[torch]"` adds PyTorch.
Other choices include `openqml[jax]`, `openqml[flax]` and `openqml[qiskit,cirq]`.

## Simulate and export a circuit

This six-qubit circuit entangles the first and last wires. Measuring either
wire alone gives a random bit, but the two results always agree. Their
`ZIIIIZ` correlation is therefore 1, up to floating-point rounding.

```python
import openqml as oq

circuit = oq.Circuit(6).h(0)
for control, target in zip(range(5), range(1, 6)):
    circuit = circuit.cx(control, target)

correlation = oq.expectation(circuit, oq.PauliString("ZIIIIZ"), backend="numpy")
print(correlation)
print(oq.convert(circuit, target="openqasm3").value)
```

`convert` also works with supported Qiskit circuits, Cirq circuits and
PennyLane tapes. For a parameter sweep, `plan_conversion` lets you prepare the
conversion once and reuse it across bindings. [Circuit formats and supported operations](https://github.com/OpenQML-org/OpenQML/blob/main/docs/interoperability.md)

## Train a hybrid model

With `openqml[torch]` installed, you can fit a small classifier directly from
Python lists. The two inputs below are the bits of an XOR problem.

```python
import openqml as oq

x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 8
y = [0, 1, 1, 0] * 8
model = oq.Model(task="classification", hidden_features=8, depth=2)
model.fit(x, y, epochs=10)
print(model.predict_proba(x[:2]))
```

Use `Model` for fitting, validation and checkpoints. If you already have a
training loop, add a quantum layer to your existing PyTorch, JAX or Flax model.
The layer exposes its parameters to the framework optimizer.
[Models](https://github.com/OpenQML-org/OpenQML/blob/main/docs/models.md) · [Training and gradients](https://github.com/OpenQML-org/OpenQML/blob/main/docs/hybrid-training.md)

## Allocate a portfolio

Each row of `returns` is one observation and each column is one asset.
Here the allocator caps either asset at 70% of the portfolio.

```python
import openqml as oq

returns = [[0.01, -0.02], [-0.01, 0.03], [0.02, 0.01]]
portfolio = oq.finance.allocate(returns, caps=0.7, fees=0.001)
print(portfolio.weights)
```

For multi-period research, scenario plans model how allocations, cash and fees
change along each path. Walk-forward evaluation fits on past observations and
scores the next period. [Portfolio construction, risk and pricing](https://github.com/OpenQML-org/OpenQML/blob/main/docs/finance.md)

## Execution options

A dense statevector stores one complex amplitude per basis state. If you only
need expectation values and the circuit has modest entanglement, the
matrix-product-state (MPS) backend can use much less memory. Exact mode keeps the full bond at each step.
If you allow Schmidt truncation, the readout reports an error bound for each
observable.

On Linux x86-64, `openqml[gpu]` provides native Triton kernels for supported
NVIDIA GPUs. Use CUDA binding tensors with `backend="torch"` to keep both the
state and its gradients on the device.
[Execution backends](https://github.com/OpenQML-org/OpenQML/blob/main/docs/runtime.md) · [Memory budgets and readout accuracy](https://github.com/OpenQML-org/OpenQML/blob/main/docs/compute.md)

[Apache-2.0 license](https://github.com/OpenQML-org/OpenQML/blob/main/LICENSE) · [Contributing](https://github.com/OpenQML-org/OpenQML/blob/main/CONTRIBUTING.md) · [Governance](https://github.com/OpenQML-org/OpenQML/blob/main/GOVERNANCE.md) · [Security](https://github.com/OpenQML-org/OpenQML/blob/main/SECURITY.md)
