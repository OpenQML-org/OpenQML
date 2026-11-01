# OpenQML

[PyPI](https://pypi.org/project/openqml/) · [CI](https://github.com/OpenQML-org/OpenQML/actions/workflows/ci.yml) · [Documentation](https://github.com/OpenQML-org/OpenQML/tree/main/docs)

Trainable quantum circuits and tensor-network layers for PyTorch, JAX and Flax.

## Installation

Requires Python 3.10 or later. Install the PyTorch integration to run the example below:

```bash
python -m pip install --upgrade "openqml[torch]"
```

Other options: `openqml[jax]`, `openqml[flax]`, or `openqml` for the NumPy core.
The core depends on NumPy, packaging and psutil; training frameworks and quantum SDKs are optional.

## Train a PyTorch model

```python
import torch
from torch import nn
import openqml as oq

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(
    oq.torch.TensorNetworkAdapter(16, num_qubits=8, num_outputs=4, device=device),
    nn.Linear(16, 2, device=device),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

x = torch.randn(32, 16, device=device)
target = torch.randint(0, 2, (32,), device=device)
optimizer.zero_grad()
logits = model(x)
loss = nn.functional.cross_entropy(logits, target)
loss.backward()
optimizer.step()
print(logits.shape)  # torch.Size([32, 2])
```

`TensorNetworkAdapter(16, ...)` preserves the last dimension: `(..., 16) → (..., 16)`.
Its MPO projections and quantum tree train together through ordinary autograd.

## Choose an interface

| Task | Interface |
| --- | --- |
| Add quantum features | `TreeQuantumLayer` |
| Adjust quantum capacity | `QuantumLayer.multiscale(bond_qubits=..., depth=...)` |
| Train an existing circuit | `QuantumLayer`; set `execution_method="tensor-network"` for native contraction |
| Factor a dense projection | `TensorTrainLinear.from_dense(weight, bias)` |
| Accumulate training gradients | `oq.torch.HybridTrainer` / `oq.jax.make_train_step` |
| Train a quantum policy | `QuantumActorCritic` and `PPOTrainer` |

These layers are available under `oq.torch`, `oq.jax` and `oq.flax`.
JAX uses `layer.init(key)` and `layer(parameters, inputs)`; Flax uses Linen `init`/`apply`.
Dense import returns a module in PyTorch, `(layer, parameters)` in JAX, and `(module, variables)` in Flax.

## Execution and documentation

Automatic circuit execution prefers a configured compatible QPU, then GPU, then CPU.
Native tensor layers run on their selected framework device. There is no fixed
circuit-width ceiling; memory, contraction width and work budgets determine capacity.
`AdaptiveMPSBackend` offers explicit approximation with checked observable-error bounds.
`compile_program` retains SDK-supported dynamic loops; `convert` uses portable circuit IR.

[Training and execution](https://github.com/OpenQML-org/OpenQML/blob/main/docs/training-and-execution.md) · [Tensor networks](https://github.com/OpenQML-org/OpenQML/blob/main/docs/tensor-networks.md) · [Hybrid training](https://github.com/OpenQML-org/OpenQML/blob/main/docs/hybrid-training.md) · [Quantum RL](https://github.com/OpenQML-org/OpenQML/blob/main/docs/quantum-rl.md) · [Circuit conversion](https://github.com/OpenQML-org/OpenQML/blob/main/docs/semantic-kernels.md) · [QPU configuration](https://github.com/OpenQML-org/OpenQML/blob/main/docs/runtime.md)

## License

[Apache-2.0](https://github.com/OpenQML-org/OpenQML/blob/main/LICENSE).
