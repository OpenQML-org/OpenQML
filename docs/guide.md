# Guide

Detail moved out of the README. For the simulator, the gradients, the backend
protocol and server mode, see [internals.md](internals.md).

## The four entities

```python
from openqml import flow_to_model, get_dataset, get_task, model_to_flow

dataset = get_dataset("tfim-1d-6q-ground-states")
X, y, names = dataset.get_data(target="phase")   # X is a stack of statevectors

task = get_task(3)                               # dataset + split rule + measure
train, test = task.get_train_test_split_indices(fold=0)

flow  = model_to_flow(model)                     # portable model description
model = flow_to_model(flow)                      # ...and back
```

A task fixes the split rule and the measure, so two people who never spoke still
produce comparable numbers. Splits derive from a seed stored on the task, not
from an index file that has to be shipped around.

A flow names a model *class* and its parameters. Rebuilding one imports code, so
only modules under `openqml.` and `sklearn.` are imported, and the name has to
resolve to a class — a bare function would turn "rebuild this model" into "call
anything with attacker-chosen arguments". Pass `allowed_prefixes=None` to
`flow_to_model` if you trust the source.

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

Every one is regenerated from a seed at first use — nothing is downloaded, and a
fresh install reproduces the same bytes. Dataset 6 is a *synthetic surrogate*:
the coefficients are not an electronic-structure calculation, and the reference
energies are exact diagonalisations of the same operators the VQE optimises.

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

### Feature ranges

An angle encoding maps a feature onto a rotation, so a value outside `[-1, 1]`
wraps past 2π and two distant points collapse onto the same angle. Every model
takes `rescale`, defaulting to `"auto"`: min-max scale to `[-1, 1]` only when the
training data actually leaves that range.

That default matters in both directions. On `blobs-4d`, whose features span
±2.5, rescaling takes the quantum kernel from 0.60 to 0.73 and the variational
classifier from 0.31 (below chance) to 0.71. On `parity-4bit`, whose bits are
already in range, `"auto"` leaves the data untouched — always rescaling would map
`{0, 1}` onto `{-1, 1}` and drop the IQP kernel from 1.00 to 0.19. Pass
`rescale=False` or `rescale=True` to force either behaviour.

The scaler is fit on the training fold only. Note that *whether* a scaler is
fitted is decided per fold, so a dataset with one out-of-range outlier can be
preprocessed differently in the fold that holds it out.

### Under shot noise

`VQE(shots=...)` reports the energy of the state it returns. That is an unbiased
estimate, so it scatters either side of the exact ground-state energy — the
variational bound constrains the exact expectation, not a finite-sample estimate
of it. Average over seeds before reading anything into a single number.

## Circuits are data

```python
from openqml import Circuit, w, x

circuit = Circuit(2).h(0).ry(x(0), 0).cnot(0, 1).rz(w(0), 1)
print(circuit.draw())
circuit.to_json()          # serialisable, publishable, re-runnable
circuit.n_parameters, circuit.n_features, circuit.depth
```

An angle is a number, a trainable weight `w(i)`, or an input feature `x(j)` —
which is what makes a model portable rather than a closure someone has to trust.
Templates cover the usual feature maps (`angle`, `amplitude`, `zz`, `iqp`, data
re-uploading) and ansätze (`hardware_efficient`, `strongly_entangling`,
`real_amplitudes`).

A weight reference may carry a `scale` or `offset`. The adjoint sweep handles
those; the two-term shift rule cannot, so a circuit using them is routed to the
fallback automatically.

## Backends

`default.statevector` is a pure-NumPy simulator (≤ 24 qubits, optional shot
noise). `pennylane.default.qubit` is used if PennyLane is installed. Adding
another takes one call:

```python
from openqml import register_backend

register_backend("my.simulator", lambda n_qubits, **kw: MySim(n_qubits, **kw))
```

A backend implementing only the minimum still works — the helpers fall back to a
loop, and one that cannot differentiate gets the parameter-shift rule. The full
protocol is in [internals.md](internals.md#backend-protocol).

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
python examples/02_quantum_vs_classical.py     # the comparison table
python examples/03_vqe_ground_states.py        # VQE vs exact diagonalisation
python examples/04_custom_dataset_and_task.py  # publish your own
python examples/05_periodic_advantage.py       # re-uploading vs ridge regression
python examples/06_variational_vs_kernel.py    # accuracy against cost (~3 s)
```

## What the tests pin

`python -m pytest tests -q` — 126 tests, ~2 s.

The variational bound holds in VQE and the shot-noise path is not biased below
it; the IQP kernel solves parity while the RBF kernel anti-learns it; batched
execution matches the one-at-a-time path exactly, chunked or not; the adjoint
sweep matches the parameter-shift rule and both match finite differences; splits
are deterministic and stratified; a flow refuses to import arbitrary modules, to
call a bare function, or to widen its own allowlist; a dataset id cannot escape
the cache directory; run hashes are stable across processes; and every name the
README tells you to import resolves.
