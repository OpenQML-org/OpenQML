# OpenQML

[PyPI](https://pypi.org/project/openqml/) · [Documentation](https://github.com/OpenQML-org/OpenQML/blob/main/docs/api.md) · [CI](https://github.com/OpenQML-org/OpenQML/actions/workflows/ci.yml)

## Quantum–classical hybrid AI

Build and train hybrid neural networks with PyTorch, JAX or Flax. Use `Model.fit` and `predict` for supervised tasks, or compose convolution, attention, graph, recurrent and generative architectures with native layers. Validation loss, early stopping and portable checkpoints keep model selection and continued training in the same workflow. [Explore hybrid AI](https://github.com/OpenQML-org/OpenQML/blob/main/docs/models.md).

## Quantum–classical hybrid quantitative research

Model return distributions, estimate tail risk, allocate constrained portfolios and price bounded payoffs. Purged walk-forward refits produce aligned out-of-sample predictions; self-financing accounting applies execution delays, fees and market impact. Differentiable financial calculations share native tensors with hybrid networks. Coherent estimation exposes precision, confidence, noise and query budgets. [Explore quantitative research](https://github.com/OpenQML-org/OpenQML/blob/main/docs/finance.md).

## Cross-framework quantum programming

Connect Qiskit, Cirq, PennyLane and OpenQASM through a shared circuit representation. Prepare conversions, inspect target compatibility and stream parameter sweeps from one imported source. Conversion plans preserve parameter ownership, explicit channels, classical control and global phase where the target supports them. [Explore interoperability](https://github.com/OpenQML-org/OpenQML/blob/main/docs/interoperability.md).

## Install

Python 3.10 or later:

```bash
python -m pip install -U openqml
```

The core depends on NumPy, packaging and psutil. Training frameworks, quantum SDKs and execution engines load on demand. Install the integrations you use, such as `openqml[torch]`, `openqml[jax]`, `openqml[flax]` or `openqml[qiskit,cirq]`.

## Quick start

Allocate from observed returns:

```python
import openqml as oq

returns = [[0.01, -0.02], [-0.01, 0.03], [0.02, 0.01]]
portfolio = oq.finance.allocate(returns, caps=0.7, fees=0.001)
print(portfolio.weights)
```

Train a hybrid model after installing `openqml[torch]`:

```python
import openqml as oq

x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 8
y = [0, 1, 1, 0] * 8
model = oq.Model(task="classification", hidden_features=8, depth=2)
model.fit(x, y, epochs=10)
print(model.predict(x[:2]))
```

Select `framework="jax"` or `framework="flax"` to change training frameworks. `backend` chooses circuit execution; `device` chooses native tensor placement. Qubit counts follow the selected backend's available resources.

[API](https://github.com/OpenQML-org/OpenQML/blob/main/docs/api.md) · [Training and validation](https://github.com/OpenQML-org/OpenQML/blob/main/docs/hybrid-training.md) · [Quantitative research](https://github.com/OpenQML-org/OpenQML/blob/main/docs/finance.md) · [Interoperability](https://github.com/OpenQML-org/OpenQML/blob/main/docs/interoperability.md) · [Tensor networks](https://github.com/OpenQML-org/OpenQML/blob/main/docs/tensor-networks.md) · [Runtime](https://github.com/OpenQML-org/OpenQML/blob/main/docs/runtime.md)

## License

[Apache-2.0](LICENSE)
