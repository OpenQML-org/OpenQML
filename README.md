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
| `VariationalQuantumClassifier` | quantum | feature map + ansatz, [analytic gradients](#gradients) |
| `VariationalQuantumRegressor` | quantum | data re-uploading with a linear read-out |
| `VQE` | quantum | ground-state energy, adjoint or parameter-shift + Adam |
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
  costing about two orders of magnitude more compute — 77× over this suite,
  measured. (It was four orders before the simulator and the gradients were
  rewritten; the gap is smaller now, and still the wrong way round.) That is
  the honest state of these two approaches at this scale.

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

Three things carry it, in order of how much they matter.

**The state is `(batch, 2**n)` and a gate never moves it.** Qubit 0 is the most
significant bit, so the amplitudes any gate mixes are already a fixed stride
apart: reshaping to `(batch, left, 2, right)` is a view, and the gate is a few
elementwise operations on the two halves written straight back in place. No
permutation, no copy, and no stacked 2×2 matmul — which is what a per-batch
angle used to force, and it is the worst shape NumPy has. Angles may still
differ per batch element, so one call covers a whole dataset or every shifted
parameter set a gradient needs.

Gates also carry their structure. Diagonal ones (`rz`, `phase`, `rzz`, `cz`,
`crz`, `z`, `s`, `t`) scale the halves and build nothing; permutations (`x`,
`cnot`, `swap`) exchange two slices; controlled rotations touch only the
control-is-set half; the Hadamard is a sum and a difference. Between them that
is most of the gates in the bundled feature maps and ansätze.

**Gradients are one backward sweep, not `2P+1` forward runs.** The
parameter-shift rule costs two whole circuit evaluations per gate parameter —
O(P²) gate applications for one gradient. On an exact simulator the same
numbers come out of a single adjoint sweep in O(P) state passes. See
[Gradients](#gradients) below; it is the same number, not an approximation.

**Cross-validation encodes its rows once** rather than once per fold (a small
cache, clearable with `models.clear_state_cache()`), and `<Z>` read-outs come
straight off the probability vector — one matmul against a cached sign matrix,
whatever the number of wires. The all-Z terms of a Hamiltonian collapse into a
single cached diagonal, so a VQE step costs one pass over the state instead of
one per term.

Measured on one CPU core, against the previous implementation of the same API.
Every score in this README is unchanged to ten decimal places; only the clock
moved.

| workload | before | after | |
| --- | --- | --- | --- |
| encode 200 rows through the ZZ map | 1.1 ms | 0.3 ms | 4.5× |
| encode 400 rows (4 qubits, 34 gates) | 6.1 ms | 1.6 ms | 3.9× |
| encode 512 rows (8 qubits, 74 gates) | 75 ms | 38 ms | 2.0× |
| one gradient step (1312 circuit evaluations) | 19 ms | 7 ms | 2.7× |
| quantum kernel, 5-fold CV on `moons-2d` | 11 ms | 7 ms | 1.5× |
| four kernel models across the whole suite | 96 ms | 70 ms | 1.4× |
| VQE over all 12 Hamiltonians (80 steps each) | 0.40 s | 0.22 s | 1.8× |
| VQE, 10 qubits, 40 steps | 1.02 s | 0.13 s | 8.1× |
| variational classifier, 5-fold CV on `moons-2d` | 4.6 s | 0.33 s | 14× |
| variational classifier, 5-fold CV on `blobs-4d` | 20.0 s | 1.5 s | 13× |
| the test suite (69 tests) | | 1.2 s | |

Batches larger than `backends.MAX_BATCH_ELEMENTS` are split automatically, so
callers never size a gradient against the qubit count by hand. The chunk is
also capped at `MAX_BATCH_AMPLITUDES` total amplitudes: a gate is memory-bound,
so the fastest chunk is the one whose working set stays in L2, not the largest
one that fits in RAM.

## Gradients

A variational model reports which rule it used in `model.gradient_method_`, and
takes `gradient=` to force one:

| `gradient` | what happens |
| --- | --- |
| `"auto"` (default) | the adjoint sweep where it is both valid and cheaper, else the shift rule |
| `"adjoint"` | the sweep, or an error saying why it cannot be used |
| `"parameter_shift"` | the two-term rule, always |

The two agree to machine precision — the test suite pins that, and a model
trained either way reaches bitwise-identical weights. Writing
`f(θ) = <ψ|O|ψ>` with `|ψ> = U_P…U_1|0>` and walking backwards while carrying
both `|ψ_j>` and `|b_j> = (U_j+1…U_P)† O |ψ>`, the derivative for gate `j` is
`Im(<b_j| G_j |ψ_j>)` for that gate's generator. Every gate here is
`exp(-iθG/2)` for a Pauli-built `G` and has a closed-form inverse, so the walk
is cheap. Where a weight drives several gates the sweep is *more* correct than
the two-term rule, which does not apply there at all (the shift path falls back
to central differences).

`"auto"` does not simply always pick the sweep, because asymptotically cheaper
is not the same as cheaper here. The shift rule stacks all `2P+1` parameter
sets into one batched run: O(gates) NumPy calls on a large array, against O(P)
calls on a small one. Below a few thousand amplitude-parameters the per-call
overhead is the whole cost and the shift rule wins — a 6-qubit VQE is on that
side of the line and a 10-qubit one is 8× the other side.
`backends.prefers_adjoint()` states the crossover and is calibrated by
measuring both rules from 2 to 12 qubits.

Shot noise always falls back to the shift rule: the sweep is analytic and has
no meaning under sampling.

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
path; add `z_jacobian`, `energy_jacobian` and `is_differentiable` to get the
adjoint one. Without any of them the helpers fall back to a loop, and
everything still works — a backend that cannot differentiate simply gets the
parameter-shift rule. See `openqml/backends/pennylane_backend.py` for a worked
bridge.

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
python examples/06_variational_vs_kernel.py    # accuracy against cost (~3 s)
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
python -m pytest tests -q     # 69 tests, ~1 s
```

The suite pins the claims above: the variational bound holds in VQE, the IQP
kernel solves parity while the RBF kernel anti-learns it, batched execution
matches the one-at-a-time path exactly, the adjoint sweep matches the
parameter-shift rule and both match finite differences, the training gradient
matches finite differences, splits are deterministic and stratified, flows
refuse to import arbitrary modules, and runs survive a filesystem round trip.

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
