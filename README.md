# OpenQML

An open exchange for quantum machine learning experiments — modelled on
[OpenML](https://openml.org), specialised for quantum.

🌐 [OpenQML.org](https://openqml.org)

```python
from openqml import QuantumKernelClassifier, get_task, leaderboard, run_model_on_task

task  = get_task(1)
model = QuantumKernelClassifier(feature_map="iqp", n_qubits=2)
run   = run_model_on_task(model, task)

print(run.summary())
run.publish()
print(leaderboard(1))
```

No account, no network, no configuration: the package ships with a local store
and eight reproducible datasets, so the snippet above works right after
`pip install`.

## Install

```bash
pip install openqml            # NumPy is the only hard dependency
pip install -e ".[dev]"        # from a clone, with the test suite
```

pandas is optional (tables degrade to lists of dicts); PennyLane is optional
(an extra backend).

## Importing

Everything a normal session needs is on the top-level namespace — models,
entity lookups, circuit templates, backends:

```python
from openqml import VQE, Circuit, QuantumKernelClassifier, compare, get_task, w, x
```

The submodules stay where they are (`openqml.models`, `openqml.circuits`,
`openqml.backends`, …) for the rest, and `openqml.list_models()`,
`list_backends()`, `list_measures()` say what is available.

## The four entities

The OpenML mapping is one-to-one; the added columns are the quantum parts,
which is what decides whether two results are comparable at all.

| entity | what OpenQML adds |
| --- | --- |
| `datasets` | `data_type` (`tabular`, `bitstring`, `quantum_state`, `hamiltonian`), qubit count, suggested encoding |
| `tasks` | `state_classification` and `ground_state_estimation` task types |
| `flows` | circuit-level description: feature map, ansatz, backend |
| `runs` | backend, shot count, analytic vs sampled, wall-clock cost |
| `evaluations` | leaderboards, and `compare()` for putting a classical arm in the same table |
| `study` | benchmark suites |

```python
dataset = get_dataset("tfim-1d-6q-ground-states")
X, y, names = dataset.get_data(target="phase")   # X is a stack of statevectors

task = get_task(3)                               # dataset + split rule + measure
train, test = task.get_train_test_split_indices(fold=0)

flow  = model_to_flow(model)                     # portable model description
model = flow_to_model(flow)                      # ...and back
```

A task fixes the split rule and the measure, so two people who never spoke
still produce comparable numbers. Splits derive from a seed stored on the task,
not from an index file that has to be shipped around.

## Bundled catalog

| id | dataset | type | shape | note |
| --- | --- | --- | --- | --- |
| 1 | `moons-2d` | tabular | 200 × 2 | non-linear but easy |
| 2 | `circles-2d` | tabular | 200 × 2 | no linear separator |
| 3 | `blobs-4d` | tabular | 150 × 4 | control: classical should win |
| 4 | `parity-4bit` | bitstring | 16 × 4 | global structure, no local shortcut |
| 5 | `tfim-1d-6q-ground-states` | quantum_state | 40 × 64 | exact TFIM ground states, phase-labelled |
| 6 | `h2-surrogate-2q` | hamiltonian | 12 operators | VQE target with exact reference energies |
| 7 | `periodic-sin3` | tabular | 120 × 1 | where re-uploading genuinely helps |
| 8 | `bars-and-stripes-3x3` | bitstring | 14 × 9 | small generative target |

Every one is regenerated from a seed at first use — nothing is downloaded, and
a fresh install reproduces the same bytes. Dataset 6 is a *synthetic
surrogate*: the coefficients are not an electronic-structure calculation.

## Models

| model | kind | notes |
| --- | --- | --- |
| `QuantumKernelClassifier` / `Regressor` | quantum | fidelity kernel + kernel ridge, no training loop |
| `ClassicalKernelClassifier` / `Regressor` | classical | *identical solver*, RBF/linear/poly kernel |
| `VariationalQuantumClassifier` | quantum | feature map + ansatz, analytic gradients |
| `VariationalQuantumRegressor` | quantum | data re-uploading with a linear read-out |
| `VQE` | quantum | ground-state energy, adjoint or parameter-shift + Adam |
| `ExactDiagonalisation` | reference | the yardstick, not a competitor |
| `MajorityClassifier` / `LinearRegressor` | classical | the floor |

The quantum and classical kernel estimators share one solver on purpose: the
only thing that differs is the Gram matrix, which is the thing under test.
Anything with `fit`/`predict` works, so a scikit-learn estimator drops straight
into `run_model_on_task`.

**Feature ranges.** An angle encoding maps a feature onto a rotation, so a
value outside `[-1, 1]` wraps past 2π and two distant points collapse onto the
same angle. Every model takes `rescale`, defaulting to `"auto"`: min-max scale
only when the training data actually leaves that range. It matters both ways —
on `blobs-4d` (features to ±2.5) rescaling takes the quantum kernel from 0.60
to 0.73; on `parity-4bit` always rescaling would map `{0, 1}` onto `{-1, 1}`
and drop the IQP kernel from 1.00 to 0.19.

## Comparing models

```python
suite = get_suite("qml-cls-1")
print(suite.compare([
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

* On **parity** the IQP feature map is exactly right and the RBF kernel scores
  **0.00**, below the majority floor: held-out bit strings never appear in
  training, so a smooth interpolator confidently predicts the opposite label.
  Anti-learning is a result, not a bug.
* On **moons** and **circles** the classical kernel matches the best quantum
  entry, in microseconds instead of seconds.
* The **variational** row is behind the kernel row on three of four tasks while
  costing about two orders of magnitude more compute (77× over this suite,
  measured). That is the honest state of these two approaches at this scale.

`compare()` returns the whole grid, because a benchmark that reports only its
wins is not a benchmark.

## Circuits are data

```python
from openqml import Circuit, w, x

circuit = Circuit(2).h(0).ry(x(0), 0).cnot(0, 1).rz(w(0), 1)
print(circuit.draw())
circuit.to_json()          # serialisable, publishable, re-runnable
```

An angle is a number, a trainable weight `w(i)`, or an input feature `x(j)` —
which is what makes a model portable rather than a closure someone has to
trust. Templates cover the usual feature maps (`angle`, `amplitude`, `zz`,
`iqp`, data re-uploading) and ansätze (`hardware_efficient`,
`strongly_entangling`, `real_amplitudes`).

## Speed and gradients

The simulator holds a `(batch, 2**n)` state and applies a gate in place, with
no permutation and no stacked 2×2 matmul; gradients are one analytic adjoint
sweep rather than the `2P+1` forward runs of the parameter-shift rule. Against
the previous implementation of the same API, on one CPU core: a variational
classifier's 5-fold CV on `moons-2d` went 4.6 s → 0.33 s, a 10-qubit VQE
1.02 s → 0.13 s, and every score in this README is unchanged to ten decimal
places.

`gradient="auto"` picks the sweep only where it is actually cheaper — the shift
rule batches all `2P+1` parameter sets into one call and wins at small sizes —
and `model.gradient_method_` reports which ran. Shot noise always falls back to
the shift rule.

Full numbers, the derivation, the backend protocol and server mode:
**[docs/internals.md](docs/internals.md)**.

## Backends

`default.statevector` is a pure-NumPy simulator (≤ 24 qubits, optional shot
noise). `pennylane.default.qubit` is used if PennyLane is installed. Adding
another takes one call:

```python
from openqml import register_backend

register_backend("my.simulator", lambda n_qubits, **kw: MySim(n_qubits, **kw))
```

A backend that implements only the minimum still works — the helpers fall back
to a loop, and one that cannot differentiate simply gets the parameter-shift
rule. The protocol is in [docs/internals.md](docs/internals.md#backend-protocol).

## Command line and examples

```bash
python -m openqml info                         # or: datasets, tasks, leaderboard 1
python -m openqml show dataset tfim-1d-6q-ground-states

python examples/01_quickstart.py               # run, publish, leaderboard
python examples/02_quantum_vs_classical.py     # the comparison table above
python examples/03_vqe_ground_states.py        # VQE vs exact diagonalisation
python examples/04_custom_dataset_and_task.py  # publish your own
python examples/05_periodic_advantage.py       # re-uploading vs ridge regression
python examples/06_variational_vs_kernel.py    # accuracy against cost (~3 s)
```

## What this package does not claim

* Simulating a circuit on a classical computer demonstrates **no quantum
  advantage**. Every number here is a statement about a model class, not about
  hardware.
* The simulator is exact up to 24 qubits, with binomial shot noise as the only
  noise model. No decoherence, no error mitigation, no hardware backend.
* `h2-surrogate-2q` is a synthetic operator family, not chemistry.
* Where a quantum model wins in the bundled suite the reason is structural and
  stated plainly: a re-uploading circuit *is* a truncated Fourier series, an
  IQP kernel *does* represent parity. A win that cannot be explained should be
  treated as an artefact until it can.

## Tests

```bash
python -m pytest tests -q     # 91 tests, ~2 s
```

They pin the claims above: the variational bound holds in VQE, the IQP kernel
solves parity while the RBF kernel anti-learns it, batched execution matches
the one-at-a-time path exactly, the adjoint sweep matches the parameter-shift
rule and both match finite differences, splits are deterministic and
stratified, flows refuse to import arbitrary modules, runs survive a filesystem
round trip, and every name this README tells you to import resolves.

MIT licensed. `openqml` is a plausible name for other projects too — check the
index before publishing under it; nothing in the code depends on the
distribution name.
