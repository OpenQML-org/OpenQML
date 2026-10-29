# OpenQML

OpenQML adds trainable quantum layers to PyTorch and JAX models and converts circuits between Qiskit, PennyLane, Cirq and OpenQASM through a shared intermediate representation.

## Install

```bash
python -m pip install 'openqml[torch]'
```

Use `openqml[jax]` for JAX, or `openqml` for NumPy-only execution.

## Train a hybrid model

```python
import torch
from openqml.interfaces.torch import QuantumLayer

model = torch.nn.Sequential(
    torch.nn.Linear(4, 2),
    QuantumLayer.data_reuploading(2, depth=1),
    torch.nn.Linear(2, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss = model(torch.randn(8, 4)).square().mean()
loss.backward()
optimizer.step()
```

## Convert circuits

```python
from openqml import Circuit, Parameter, convert

circuit = Circuit(2).ry(Parameter("theta"), 0).cx(0, 1)
qasm = convert(circuit, target="openqasm3").value
restored = convert(qasm, source="openqasm3", target="ir").value
print(qasm)
```

Import SDK circuits with `convert(program, target="ir")`; install the matching `qiskit`, `pennylane` or `cirq` extra.

[More examples](examples/) · [Apache license](LICENSE)
