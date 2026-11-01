# OpenQML

[PyPI](https://pypi.org/project/openqml/) · [CI](https://github.com/OpenQML-org/OpenQML/actions/workflows/ci.yml) · [Documentation](docs/research-methods.md)

OpenQML is a Python library for hybrid quantum-classical learning and quantum
circuit interoperability.

## Features

- Trainable quantum layers for PyTorch and JAX, with native automatic differentiation.
- Circuit conversion across Qiskit, PennyLane, Cirq and OpenQASM, with explicit semantic verification.
- Resource-aware execution, sparse Pauli propagation, quantum geometry and measurement estimation.

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

[Architecture](docs/architecture.svg) · [Semantic verification](docs/semantic-kernels.md) ·
[Scalable execution](docs/scalable-methods.md) · [Research methods](docs/research-methods.md)

There is no fixed total circuit-width ceiling. Execution remains subject to
representation, memory and work budgets. Approximation is opt-in and reports its bounds.

## License

[Apache License 2.0](LICENSE).
