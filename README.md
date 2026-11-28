# OpenQML

OpenQML is a Python library for training quantum–classical models, working with
portfolio scenarios and moving quantum circuits between frameworks. You can
build and simulate a circuit with NumPy, then use the same circuit in a PyTorch
or JAX model.

[PyPI](https://pypi.org/project/openqml/) · [API reference](https://github.com/OpenQML-org/OpenQML/blob/main/docs/api.md) · [Changelog](https://github.com/OpenQML-org/OpenQML/blob/main/CHANGELOG.md)

## Run your first circuit

Install the core package on Python 3.10 or later.

```bash
python -m pip install openqml
```

```python
import openqml as oq

bell = oq.Circuit(2).h(0).cx(0, 1)
print(oq.sample(bell, shots=1000, seed=7))
print(oq.expectation(bell, "ZZ"))
```

The first line of output counts `00` and `11`. The second gives their exact
Z correlation, `1.0`. Bit strings and Pauli words both start with qubit 0.
OpenQML selects an available backend automatically.

Export the same circuit with `oq.convert(bell, target="openqasm3").value`.
OpenQASM export and local simulation are included in the core package.
Qiskit and OpenQASM 3 can exchange measurement-controlled loops and bounded
dynamic indices through the shared circuit representation.
[Runtime loops and addressed wires](https://github.com/OpenQML-org/OpenQML/blob/main/docs/interoperability.md#runtime-loops-and-addressed-wires) · [Circuit formats and supported operations](https://github.com/OpenQML-org/OpenQML/blob/main/docs/interoperability.md)

Install `openqml[torch]`, `openqml[jax]` or `openqml[flax]` for training.
SDK integrations can also be combined, for example `openqml[qiskit,cirq]`.
[Platform and backend details](https://github.com/OpenQML-org/OpenQML/blob/main/docs/runtime.md)

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

To choose a predictor for deployment, compare the fitted model with your
strongest classical alternatives on a separate calibration set.
`model.select_predictor(...)` returns a callable predictor and the paired loss
evidence under a common cost ceiling. [Selecting a predictor](https://github.com/OpenQML-org/OpenQML/blob/main/docs/hybrid-training.md#selecting-a-predictor)

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

The CPU simulator samples Clifford circuits directly and evaluates Pauli words
without building a statevector. For general circuits, NumPy fuses local gates
and consecutive diagonal rotations. Use `oq.statevector(circuit)` when you need
every amplitude. The MPS backend also supports circuits with modest
entanglement and reports a readout error bound when truncation is enabled.

On Linux x86-64, `openqml[gpu]` provides native Triton kernels for supported
NVIDIA GPUs. Use CUDA binding tensors with `backend="torch"` to keep both the
state and its gradients on the device.
[Execution backends](https://github.com/OpenQML-org/OpenQML/blob/main/docs/runtime.md) · [Memory budgets and readout accuracy](https://github.com/OpenQML-org/OpenQML/blob/main/docs/compute.md)

[Apache-2.0 license](https://github.com/OpenQML-org/OpenQML/blob/main/LICENSE) · [Contributing](https://github.com/OpenQML-org/OpenQML/blob/main/CONTRIBUTING.md) · [Governance](https://github.com/OpenQML-org/OpenQML/blob/main/GOVERNANCE.md) · [Security](https://github.com/OpenQML-org/OpenQML/blob/main/SECURITY.md)
