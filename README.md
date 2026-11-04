# OpenQML

[PyPI](https://pypi.org/project/openqml/) · [CI](https://github.com/OpenQML-org/OpenQML/actions/workflows/ci.yml) · [Documentation](https://github.com/OpenQML-org/OpenQML/tree/main/docs)

Quantum–classical models, quantitative finance and circuit interoperability in one Python package.

## Install

Python 3.10 or later:

```bash
python -m pip install --upgrade openqml
```

The core depends on NumPy, packaging and psutil. Frameworks and quantum SDKs are
optional; importing OpenQML loads neither NumPy nor execution engines.

## Allocate a portfolio

```python
import openqml as oq

returns = [[0.01, -0.02], [-0.01, 0.03], [0.02, 0.01]]
portfolio = oq.finance.allocate(returns, caps=0.7, fees=0.001)
print(portfolio.weights)
```

The allocator includes factor risk, position limits, funding and transaction
costs. Supply your own return observations and previous holdings.

## Train a model

Install `openqml[torch]`, `openqml[jax]` or `openqml[flax]`, then use the same
`fit`/`predict` interface:

```python
import openqml as oq

x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 8
y = [0, 1, 1, 0] * 8
model = oq.Model(task="classification", hidden_features=8, depth=2)
model.fit(x, y)
print(model.predict(x[:2]))
```

Set `framework="jax"` or `framework="flax"` to change the training integration.
Tasks include `regression`, `classification` and `forecast`. The native network
and trainer remain available as `model.module` and `model.trainer`.

## Interfaces

| Task | Entry point |
| --- | --- |
| Supervised hybrid models | `oq.Model` |
| Custom convolution, attention, graph and generative networks | `oq.HybridArchitecture` |
| Risk, allocation, accounting and pricing | `oq.finance` |
| Native framework layers | `oq.torch`, `oq.jax`, `oq.flax` |
| Circuits and conversion | `oq.Circuit`, `oq.convert`, `oq.compile_program` |
| Tensor contraction and distribution | `oq.TensorNetworkPlan`, `oq.DistributedTensorPlan` |
| Quantum reinforcement learning | Framework `QuantumActorCritic` and `PPOTrainer` |
| Hardware and cost policies | `oq.configure_runtime`, `oq.CostAwareBackend` |

`framework` selects the training integration, `backend` selects circuit execution,
and `device` selects native tensor placement. Modules, integrations and backends
load when used. Existing low-level APIs remain available through `oq`.

Automatic pricing uses available exact calculations before amplified sampling.
Quantum estimation accepts an error target, calibrated contrast and query limits.
Quantum query advantages depend on coherent oracles, noise and total execution
cost. Circuit capacity follows memory, work and backend limits; there is no fixed
library-wide qubit ceiling.

[API guide](https://github.com/OpenQML-org/OpenQML/blob/main/docs/api.md) · [Finance](https://github.com/OpenQML-org/OpenQML/blob/main/docs/finance.md) · [Models](https://github.com/OpenQML-org/OpenQML/blob/main/docs/models.md) · [Stateful models](https://github.com/OpenQML-org/OpenQML/blob/main/docs/frontier-models.md) · [Training](https://github.com/OpenQML-org/OpenQML/blob/main/docs/hybrid-training.md) · [Tensor networks](https://github.com/OpenQML-org/OpenQML/blob/main/docs/tensor-networks.md) · [Runtime](https://github.com/OpenQML-org/OpenQML/blob/main/docs/runtime.md)

## License

[Apache-2.0](https://github.com/OpenQML-org/OpenQML/blob/main/LICENSE).
