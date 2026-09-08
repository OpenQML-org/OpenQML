# OpenQML

An open exchange for quantum machine learning experiments — modelled on
[OpenML](https://openml.org), specialised for quantum.

```python
import openqml

task  = openqml.get_task(1)
model = openqml.models.QuantumKernelClassifier(feature_map="iqp", n_qubits=2)
run   = openqml.run_model_on_task(model, task)

print(run.summary())
run.publish()
print(openqml.leaderboard(1))
```

No account, no network, no configuration: the package ships with a local store
and eight reproducible datasets, so the snippet above works right after
`pip install`.

---

## Why it looks like OpenML

Because the four-entity model is the right one and people already know it. The
mapping is one-to-one; the differences are the quantum parts.

| OpenML | OpenQML | what is added |
| --- | --- | --- |
| `datasets` | `openqml.datasets` | `data_type` (`tabular`, `bitstring`, `quantum_state`, `hamiltonian`), qubit count, suggested encoding |
| `tasks` | `openqml.tasks` | `state_classification` and `ground_state_estimation` task types |
| `flows` | `openqml.flows` | circuit-level description: feature map, ansatz, backend |
| `runs` | `openqml.runs` | backend, shot count, analytic vs sampled, wall-clock cost |
| `evaluations` | `openqml.evaluations` | leaderboards, and `compare()` for putting a classical arm in the same table |
| `study` | `openqml.study` | benchmark suites |

## Install

```bash
pip install openqml                    # from a package index
pip install -e ".[dev]"                # from a clone, with the test suite
```

The only hard dependency is NumPy. pandas is optional (tables degrade to
dicts); PennyLane is optional (an extra backend).

## The four entities

```python
import openqml

openqml.list_datasets()                         # 8 bundled, plus anything you publish
dataset = openqml.get_dataset("tfim-1d-6q-ground-states")
X, y, names = dataset.get_data(target="phase")  # X here is a stack of statevectors

task = openqml.get_task(3)                      # dataset + split rule + measure
train, test = task.get_train_test_split_indices(fold=0)

flow  = openqml.model_to_flow(model)            # portable model description
model = openqml.flow_to_model(flow)             # ...and back

run = openqml.run_model_on_task(model, task)    # predictions + per-fold scores
run.publish()
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
a fresh install reproduces the same bytes. Dataset 6 is a *synthetic surrogate*,
labelled as such in its own description: the coefficients are not an
electronic-structure calculation, and the reference energies are exact
diagonalisations of the same operators the VQE optimises.

## Models

| model | kind | notes |
| --- | --- | --- |
| `QuantumKernelClassifier` / `Regressor` | quantum | fidelity kernel + kernel ridge, no training loop |
| `ClassicalKernelClassifier` / `Regressor` | classical | *identical solver*, RBF/linear/poly kernel |
| `VariationalQuantumClassifier` | quantum | feature map + ansatz, parameter-shift gradients |
| `VariationalQuantumRegressor` | quantum | data re-uploading with a linear read-out |
| `VQE` | quantum | ground-state energy, parameter-shift + Adam |
| `ExactDiagonalisation` | reference | the yardstick, not a competitor |
| `MajorityClassifier` / `LinearRegressor` | classical | the floor |

The quantum and classical kernel estimators share one solver on purpose: the
only thing that differs is the Gram matrix, which is the thing under test.
Anything with `fit`/`predict` works, so a scikit-learn estimator drops straight
into `run_model_on_task`.

### Feature ranges

An angle encoding maps a feature onto a rotation, so a value outside `[-1, 1]`
wraps past 2π and two distant points collapse onto the same angle. Every model
takes `rescale`, defaulting to `"auto"`: min-max scale to `[-1, 1]` only when
the training data actually leaves that range.

That default matters in both directions. On `blobs-4d`, whose features span
±2.5, rescaling takes the quantum kernel from 0.60 to 0.73 and the variational
classifier from 0.31 (below chance) to 0.71. On `parity-4bit`, whose bits are
already in range, `"auto"` leaves the data untouched — always rescaling would
map `{0, 1}` onto `{-1, 1}` and drop the IQP kernel from 1.00 to 0.19. Pass
`rescale=False` or `rescale=True` to force either behaviour.

## Comparing models

```python
suite = openqml.get_suite("qml-cls-1")
print(suite.compare([
    openqml.models.QuantumKernelClassifier(feature_map="iqp"),
    openqml.models.QuantumKernelClassifier(feature_map="zz"),
    openqml.models.VariationalQuantumClassifier(),
    openqml.models.ClassicalKernelClassifier(kernel="rbf"),
    openqml.models.MajorityClassifier(),
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

Four things in that table are worth stating out loud:

* On **parity**, the IQP feature map is exactly right and the RBF kernel scores
  **0.00** — below the majority floor. Held-out bit strings never appear in
  training, so a smooth interpolator confidently predicts the opposite label.
  Anti-learning is a result, not a bug.
* On **TFIM phases**, where the input already *is* a quantum state, the
  fidelity kernel needs no encoding step at all and comes out ahead.
* On **moons** and **circles**, the classical kernel matches the best quantum
  entry, in microseconds instead of seconds.
* The **variational** row is behind the kernel row on three of four tasks while
  costing four orders of magnitude more compute. That is the honest state of
  these two approaches at this scale.

`compare()` runs every model over every task and returns the whole grid,
because a benchmark that reports only its wins is not a benchmark.

## Circuits are data

```python
from openqml.circuits import Circuit, w, x

circuit = Circuit(2).h(0).ry(x(0), 0).cnot(0, 1).rz(w(0), 1)
print(circuit.draw())
circuit.to_json()          # serialisable, publishable, re-runnable
circuit.n_parameters, circuit.n_features, circuit.depth
```

Angles are either numbers, a reference to a trainable weight `w(i)`, or a
reference to an input feature `x(j)`. That is what makes a model portable
rather than a closure someone has to trust. Templates are provided for the
usual feature maps (`angle`, `amplitude`, `zz`, `iqp`, data re-uploading) and
ansätze (`hardware_efficient`, `strongly_entangling`, `real_amplitudes`).

## Performance

The simulator carries a batch dimension: the state is a `(batch, 2, ..., 2)`
tensor, a gate is one reshape and one matmul, and angles may differ per batch
element. Encoding a dataset is therefore a single pass, and a parameter-shift
gradient — all `2P+1` shifted parameter sets against all samples in the
mini-batch — is one array of circuits instead of thousands of calls.

Measured on one CPU core, against the same code path with batching switched off:

| workload | batched | one circuit at a time |
| --- | --- | --- |
| encode 400 rows (4 qubits, 34 gates) | 6 ms | 325 ms |
| one gradient step (1312 circuit evaluations) | 20 ms | 1199 ms |

End-to-end, on the bundled tasks:

| workload | time |
| --- | --- |
| encode 200 rows through the ZZ map | 4 ms |
| quantum kernel, 5-fold CV on `moons-2d` | 17 ms |
| VQE over all 12 Hamiltonians (80 steps each) | 0.43 s, mean error 2.3e-05 |
| variational classifier, 5-fold CV on `moons-2d` | 4.5 s |
| four kernel models across the whole suite | 0.19 s |
| the test suite (52 tests) | 1.6 s |

Two more things keep the common paths cheap: k-fold cross-validation encodes
its rows once rather than once per fold (a small cache, clearable with
`models.clear_state_cache()`), and `<Z>` read-outs come straight off the
probability vector instead of a separate matrix product per wire.

Batches larger than `backends.MAX_BATCH_ELEMENTS` are split automatically, so
callers never size a gradient against the qubit count by hand.

## Backends

`default.statevector` is a pure-NumPy simulator (≤ 24 qubits, optional shot
noise). `pennylane.default.qubit` is used if PennyLane is installed. Adding
another takes one call:

```python
openqml.backends.register_backend("my.simulator", lambda n_qubits, **kw: MySim(n_qubits, **kw))
```

A backend needs `run(circuit, weights, features)`, `state`, `expval(word)` and
`expval_hamiltonian(terms)`. Set `supports_batch = True` and implement
`run_batch`, `states`, `z_expvals(wires)` and `energies(terms)` to get the fast
path; without them the helpers fall back to a loop and everything still works.
See `openqml/backends/pennylane_backend.py` for a worked bridge.

## Local store and server mode

By default `openqml.config.server == "local"` and everything lives under
`~/.cache/openqml`. To talk to a deployment:

```python
openqml.config.server = "https://example.org/api/v1"
openqml.config.apikey = "..."
```

The client expects a small REST surface — `GET /{entity}`, `GET /{entity}/{id}`,
`POST /{entity}`, `DELETE /{entity}/{id}` for `datasets`, `tasks`, `flows`,
`runs` and `suites`, JSON in and out, bearer-token auth. The JSON is exactly
what `to_dict()` produces on each entity, so the local store doubles as the
schema reference for the server side.

## Command line

```bash
python -m openqml info
python -m openqml datasets
python -m openqml tasks
python -m openqml leaderboard 1
python -m openqml show dataset tfim-1d-6q-ground-states
```

## Examples

```bash
python examples/01_quickstart.py               # run, publish, leaderboard
python examples/02_quantum_vs_classical.py     # the comparison table above
python examples/03_vqe_ground_states.py        # VQE vs exact diagonalisation
python examples/04_custom_dataset_and_task.py  # publish your own
python examples/05_periodic_advantage.py       # re-uploading vs ridge regression
python examples/06_variational_vs_kernel.py    # accuracy against cost (~30 s)
```

## What this package does not claim

* Simulating a circuit on a classical computer demonstrates **no quantum
  advantage**. Every number here is a statement about a model class, not about
  hardware.
* The simulator is exact up to 24 qubits, with binomial shot noise as the only
  noise model. No decoherence model, no error mitigation, no hardware backend.
* `h2-surrogate-2q` is a synthetic operator family, not chemistry.
* Where a quantum model wins in the bundled suite, the reason is structural and
  stated plainly: a re-uploading circuit *is* a truncated Fourier series, an IQP
  kernel *does* represent parity. A win that cannot be explained should be
  treated as an artefact until it can.

## Tests

```bash
python -m pytest tests -q     # 52 tests, ~2 s
```

The suite pins the claims above: the variational bound holds in VQE, the IQP
kernel solves parity while the RBF kernel anti-learns it, batched execution
matches the one-at-a-time path exactly, the training gradient matches finite
differences, splits are deterministic and stratified, flows refuse to import
arbitrary modules, and runs survive a filesystem round trip.

## Layout

```
openqml/
  config.py         server selection, cache directory
  _store.py         local on-disk store + REST client
  entities.py       shared publish/serialise behaviour
  catalog.py        the bundled datasets, tasks and suites
  circuits/         circuit IR, feature maps, ansätze
  backends/         batched statevector simulator, optional PennyLane bridge
  datasets/         dataset entity, generators, create/list/get
  tasks/            task types, split machinery
  flows/            model description and reconstruction
  runs/             execution, scoring, run entity
  evaluations/      leaderboards and compare()
  study/            benchmark suites
  models/           quantum models and their classical control arms
  hamiltonians.py   Pauli sums, dense matrices, exact ground states
  metrics.py        evaluation measures
  cli.py            python -m openqml
```

`openqml` is a plausible name for other projects too — check the index before
publishing under it. Nothing in the code depends on the distribution name.

MIT licensed.
