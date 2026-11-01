# OpenQML

[PyPI](https://pypi.org/project/openqml/) · [CI](https://github.com/OpenQML-org/OpenQML/actions/workflows/ci.yml) · [Documentation](https://github.com/OpenQML-org/OpenQML/blob/main/docs/hybrid-training.md)

OpenQML is a Python library for hybrid quantum-classical learning and quantum
circuit interoperability.

## Features

- Trainable quantum layers and circuit hypernetwork adapters for PyTorch/JAX, with native autodiff.
- Quantum actor-critic policies, PPO/SAC objectives and explicit hardware-response surrogates.
- Circuit conversion across Qiskit, PennyLane, Cirq and OpenQASM, with explicit semantic verification.
- QPU→GPU→CPU dispatch, sparse readouts, quantum geometry and measurement estimation.

## Installation

Requires Python 3.10 or later.

```bash
python -m pip install openqml
```

```python
import openqml as oq
```

The core depends on NumPy, packaging and psutil. Optional framework integrations
are available through extras such as `openqml[torch]`, `openqml[jax]` and `openqml[all]`.

## Documentation

[Architecture](https://github.com/OpenQML-org/OpenQML/blob/main/docs/architecture.svg) · [Semantic verification](https://github.com/OpenQML-org/OpenQML/blob/main/docs/semantic-kernels.md) ·
[Hybrid training](https://github.com/OpenQML-org/OpenQML/blob/main/docs/hybrid-training.md) · [Quantum RL](https://github.com/OpenQML-org/OpenQML/blob/main/docs/quantum-rl.md) · [Runtime](https://github.com/OpenQML-org/OpenQML/blob/main/docs/runtime.md)

There is no fixed total circuit-width ceiling. Execution remains subject to
representation, memory and work budgets. Approximation is opt-in and reports its bounds.

## License

[Apache License 2.0](https://github.com/OpenQML-org/OpenQML/blob/main/LICENSE).
