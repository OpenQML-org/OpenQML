# OpenQML

OpenQML adds trainable quantum layers to PyTorch and JAX models and converts circuits between Qiskit, PennyLane, Cirq and OpenQASM through a shared intermediate representation.

## Install

```bash
python -m pip install openqml
```

NumPy is the only required dependency. Install the extra for your workflow, for example `python -m pip install 'openqml[torch]'`. Other extras include `jax`, `flax`, `qiskit`, `pennylane`, `cirq`, `aer` and `qasm`; `all` installs every integration.

## One OpenQML import

```python
import openqml as oq

circuit = oq.Circuit(2).ry(oq.Parameter("theta"), 0).cx(0, 1)
value = oq.expectation(circuit, oq.PauliString("ZI"), {"theta": 0.4})
qasm = oq.convert(circuit, target="openqasm3").value
restored = oq.convert(qasm, source="openqasm3", target="ir").value
```

Core operations use the `oq` namespace. Framework integrations load only when accessed:

| Namespace | Use |
| --- | --- |
| `oq.torch` | PyTorch modules and functional quantum layers |
| `oq.jax` | Functional layers with JIT, gradients and mapped execution |
| `oq.flax` | Linen modules with standard parameter trees |

## Hybrid training

```python
import torch
import openqml as oq

model = torch.nn.Sequential(
    torch.nn.Linear(4, 2),
    oq.torch.QuantumLayer.data_reuploading(2, depth=1),
    torch.nn.Linear(2, 1),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss = model(torch.randn(8, 4)).square().mean()
loss.backward()
optimizer.step()
```

One optimizer updates the classical and quantum parameters. For JAX, construct `oq.jax.QuantumLayer`, initialize weights with `layer.init(key)`, then call `layer(weights, inputs)`. Native tensor backends support higher derivatives; numerical bridges provide first-order JVPs and VJPs.

## Circuit interoperability

Import SDK program objects with `oq.convert(program, target="ir")`. Use `oq.plan_conversion(...)` to inspect compatibility and resource limits before exporting. Plans diagnose unsupported features and conversions that cannot preserve semantics. Layers retain source ownership and support portable checkpoints.

## Architecture

![OpenQML project architecture](https://raw.githubusercontent.com/OpenQML-org/OpenQML/main/docs/architecture.svg)

[Apache license](LICENSE)
