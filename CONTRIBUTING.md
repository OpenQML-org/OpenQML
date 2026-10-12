# Contributing

Use Python 3.10+ and install the editable package and development tools:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

Keep the core NumPy-only. Import optional frameworks inside their adapters, and
never contact a service or change global random state during package import.
Use explicit seeds and document wire order, mathematical conventions, and
resource bounds. Reject unsupported conversions instead of dropping operations.

Numerical changes need independent references: analytic values, full small-system
matrices, finite differences for gradients, or exact eigenspectra. Learning
changes need training-only preprocessing and held-out metrics. Do not add fixed
success numbers, unstated parameter matching, or quantum advantage claims.

For new datasets/models/tasks, register metadata and include the source or
generator version, reproducible settings, and a classical comparator where
appropriate. Synthetic toy tasks must be labeled as such.

The strict QASM subset and simulation budgets are public contracts. Broadening
them requires correctness and allocation/work checks before documenting support.
Add runnable examples and keep the Chinese README and English quickstart aligned.
