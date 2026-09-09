# OpenQML

An open exchange for quantum machine learning experiments — modelled on
[OpenML](https://openml.org), specialised for quantum. Every result carries the
metadata that decides whether two quantum numbers are comparable — encoding,
qubits, backend, analytic or sampled — and every built-in model has a classical
control arm.

🌐 [OpenQML.org](https://openqml.org)

```bash
pip install openqml          # NumPy is the only hard dependency
```

```python
from openqml import QuantumKernelClassifier, get_task, leaderboard, run_model_on_task

run = run_model_on_task(QuantumKernelClassifier(feature_map="iqp"), get_task(1))
print(run.summary())
run.publish()
print(leaderboard(1))
```

No account, no network, no configuration: a local store and eight reproducible
datasets ship with the package, so that runs right after install. Everything a
session needs is on the top-level namespace — models, entity lookups, circuit
templates, backends.

## Why it looks like OpenML

Datasets, tasks, flows and runs, one-to-one with OpenML, because people already
know the model. The added columns are the quantum parts:

| entity | what OpenQML adds |
| --- | --- |
| `datasets` | `data_type` (`tabular`, `bitstring`, `quantum_state`, `hamiltonian`), qubit count, suggested encoding |
| `tasks` | `state_classification` and `ground_state_estimation` task types |
| `flows` | circuit-level description: feature map, ansatz, backend |
| `runs` | backend, shot count, analytic vs sampled, wall-clock cost |
| `evaluations` | leaderboards, and `compare()` for putting a classical arm in the same table |
| `study` | benchmark suites |

A task fixes the split rule and the measure, so two people who never spoke still
produce comparable numbers. Splits derive from a seed stored on the task, not
from an index file that has to be shipped around.

## The table this exists to produce

```python
print(get_suite("qml-cls-1").compare([
    QuantumKernelClassifier(feature_map="iqp"),
    QuantumKernelClassifier(feature_map="zz"),
    VariationalQuantumClassifier(),
    ClassicalKernelClassifier(kernel="rbf"),
    MajorityClassifier(),
]))
```

```
                                   model  moons-2d  parity-4bit   tfim-6q  circles-2d
QuantumKernelClassifier[feature_map=iqp]    0.9900       1.0000    1.0000      0.9950
 QuantumKernelClassifier[feature_map=zz]    0.9850       0.5625    1.0000      1.0000
            VariationalQuantumClassifier    0.8100       1.0000    0.9750      0.9150
               ClassicalKernelClassifier    0.9900       0.0000    0.9500      1.0000
                      MajorityClassifier    0.5000       0.5000    0.5000      0.5000
```

On **parity** the IQP feature map is exactly right and the RBF kernel scores
**0.00**, below the majority floor: held-out bit strings never appear in
training, so a smooth interpolator confidently predicts the opposite label —
anti-learning is a result, not a bug. On **moons** and **circles** the classical
kernel matches the best quantum entry in microseconds. The **variational** row is
behind the kernel row on three of four tasks while costing about two orders of
magnitude more compute. That is the honest state of these two approaches at this
scale, and `compare()` returns the whole grid, because a benchmark that reports
only its wins is not a benchmark.

## What ships

**Eight datasets** — tabular, bit-string, quantum-state and Hamiltonian — each
regenerated from a seed at first use, so nothing is downloaded and a fresh
install reproduces the same bytes (`python -m openqml datasets`).

**Ten models** (`openqml.list_models()`): quantum and classical kernel
estimators sharing one solver so that only the Gram matrix differs, a
variational classifier and regressor, VQE with exact diagonalisation as the
yardstick, and majority/ridge floors. Anything with `fit`/`predict` works, so a
scikit-learn estimator drops straight into `run_model_on_task`.

Circuits are data — an angle is a number, a trainable weight `w(i)` or an input
feature `x(j)` — so a model is portable rather than a closure someone has to
trust. Gradients come from one analytic adjoint sweep rather than the `2P+1`
forward runs of the parameter-shift rule; `gradient="auto"` picks the sweep only
where it is actually cheaper, and `model.gradient_method_` reports which ran.

## Honest limits

* Simulating a circuit on a classical computer demonstrates **no quantum
  advantage**. Every number here is a statement about a model class, not about
  hardware.
* The simulator is exact up to 24 qubits, with binomial shot noise as the only
  noise model. No decoherence, no error mitigation, no hardware backend.
* Under shots a VQE energy is an unbiased estimate, so it scatters either side
  of the exact ground state; the variational bound constrains the exact
  expectation, not a finite-sample estimate of it.
* `h2-surrogate-2q` is a synthetic operator family, not chemistry.
* Where a quantum model wins the reason is structural and stated plainly: a
  re-uploading circuit *is* a truncated Fourier series, an IQP kernel *does*
  represent parity. A win that cannot be explained should be treated as an
  artefact until it can.

## More

* [docs/guide.md](docs/guide.md) — the entities in detail, the catalog, the
  models, feature scaling, circuits, backends, the CLI.
* [docs/internals.md](docs/internals.md) — the simulator, the adjoint gradients,
  the backend protocol, server mode.
* [examples/](examples) — six runnable scripts, `01_quickstart.py` first.
* `python -m pytest tests -q` — 126 tests, ~2 s.

MIT licensed.
