# OpenQML

[PyPI](https://pypi.org/project/openqml/) · [CI](https://github.com/OpenQML-org/OpenQML/actions/workflows/ci.yml) · [Documentation](https://github.com/OpenQML-org/OpenQML/tree/main/docs)

Build and train quantum–classical neural networks in PyTorch, JAX and Flax.

## Installation

Requires Python 3.10 or later. Install the integration you use:

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
architecture = oq.HybridArchitecture(
    16,
    2,
    hidden_features=32,
    depth=4,
    quantum=oq.QuantumCellSpec(num_qubits=4, depth=1),
)
model = architecture.build("torch", device=device)
trainer = model.trainer(lr=1e-3, quantum_lr=3e-4, microbatch_size=8)

x = torch.randn(32, 16, device=device)
target = torch.randint(0, 2, (32,), device=device)
metrics = trainer.step(
    (x, target),
    lambda network, batch: nn.functional.cross_entropy(network(batch[0]), batch[1]),
)
print(model(x).shape)  # torch.Size([32, 2])
```

The architecture owns residual blocks, quantum cells, projections and one stable
parameter layout. `DepthSchedule` grows active depth while preserving optimizer state.

## Choose an interface

| Task | Interface |
| --- | --- |
| Build a hybrid network | `HybridArchitecture` and `QuantumCellSpec` |
| Image and temporal convolutions | `HybridArchitecture.convolutional(...)` |
| Reduced-state quantum attention | `HybridArchitecture.transformer(..., attention="state")` |
| Higher-order graph networks | `HybridArchitecture.graph_network(..., order=2)` |
| Quantum KAN activations | `HybridArchitecture.kan(...)` |
| Recurrent quantum memory | `HybridArchitecture.recurrent(...)` |
| Born probability generators | `HybridArchitecture.generative(...)` |
| Autoregressive quantum sequence models | `HybridArchitecture.language_model(...)` |
| Mixed-state quantum diffusion | `QuantumDiffusionPlan` and `ChannelLayer` |
| Compose or extend network blocks | `OperatorSpec` and `register_network_operator(...)` |
| Use a custom quantum circuit | `QuantumCellSpec.from_spec(...)` |
| Adjust quantum capacity | `QuantumLayer.multiscale(bond_qubits=..., depth=...)` |
| Train an existing circuit | `QuantumLayer`; set `execution_method="tensor-network"` for native contraction |
| Factor a dense projection | `TensorTrainLinear.from_dense(weight, bias)` |
| Train with accumulation and checkpoints | `model.trainer(...)` |
| Train a quantum policy | `QuantumActorCritic` and `PPOTrainer` |
| Add a quantum residual cell | `QuantumLayer.deep` |
| Schedule time, price and precision | `CostAwareBackend` and `TrainingWorkload` |
| Distribute circuit contractions | `DistributedTensorPlan` with a Torch `DeviceMesh` |

Use `native_execution="pauli-transfer"` on an architecture to compile restricted
circuits into batched exact Pauli transfers. State attention shares one quantum
feature bank across heads; pair graphs retain quadratic node-pair states.
See [Models](https://github.com/OpenQML-org/OpenQML/blob/main/docs/models.md) for costs and input formats.
See [Stateful and generative models](https://github.com/OpenQML-org/OpenQML/blob/main/docs/frontier-models.md) for sequence training, density carries and method sources.

Framework layers are available under `oq.torch`, `oq.jax` and `oq.flax`.
JAX uses `layer.init(key)` and `layer(parameters, inputs)`; Flax uses Linen `init`/`apply`.
Dense import returns a module in PyTorch, `(layer, parameters)` in JAX, and `(module, variables)` in Flax.

## Execution and documentation

Automatic circuit execution compares configured time, price and error forecasts.
QPU profiles include queue, pricing and noise estimates; without profiles,
local execution uses GPU, then CPU. `PriorityBackend` retains explicit hardware priority.
Native tensor layers run on their selected framework device. There is no fixed
circuit-width ceiling; memory, contraction width and work budgets determine capacity.
`AdaptiveMPSBackend` offers explicit approximation with checked observable-error bounds.
`compile_program` retains SDK-supported dynamic loops; `convert` uses portable circuit IR.
The model planner schedules the whole network against shared time and cost limits.
Distributed contractions shard intermediate tensors; GPU scaling requires hardware measurements.

[Models](https://github.com/OpenQML-org/OpenQML/blob/main/docs/models.md) · [Architecture](https://github.com/OpenQML-org/OpenQML/blob/main/docs/deep-learning.md) · [Distributed execution](https://github.com/OpenQML-org/OpenQML/blob/main/docs/distributed-execution.md) · [Training and execution](https://github.com/OpenQML-org/OpenQML/blob/main/docs/training-and-execution.md) · [Tensor networks](https://github.com/OpenQML-org/OpenQML/blob/main/docs/tensor-networks.md) · [Hybrid training](https://github.com/OpenQML-org/OpenQML/blob/main/docs/hybrid-training.md) · [Quantum RL](https://github.com/OpenQML-org/OpenQML/blob/main/docs/quantum-rl.md) · [Circuit conversion](https://github.com/OpenQML-org/OpenQML/blob/main/docs/semantic-kernels.md) · [QPU configuration](https://github.com/OpenQML-org/OpenQML/blob/main/docs/runtime.md)

## License

[Apache-2.0](https://github.com/OpenQML-org/OpenQML/blob/main/LICENSE).
