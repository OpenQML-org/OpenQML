# Internals

Detail moved out of the README: what makes the simulator fast, how the
gradients are derived, what a backend has to implement, and the REST surface
the client expects in server mode.

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
numbers come out of a single adjoint sweep in O(P) state passes. It is the same
number, not an approximation.

**Cross-validation encodes its rows once** rather than once per fold (a small
cache, clearable with `openqml.clear_state_cache()`), and `<Z>` read-outs come
straight off the probability vector — one matmul against a cached sign matrix,
whatever the number of wires. The all-Z terms of a Hamiltonian collapse into a
single cached diagonal, so a VQE step costs one pass over the state instead of
one per term.

Measured on one CPU core, against the previous implementation of the same API.
Every score in the README is unchanged to ten decimal places; only the clock
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

## Backend protocol

Register a backend with one call:

```python
from openqml import register_backend

register_backend("my.simulator", lambda n_qubits, **kw: MySim(n_qubits, **kw))
```

| level | what to implement | what you get |
| --- | --- | --- |
| minimum | `run(circuit, weights, features)`, `state`, `expval(word)`, `expval_hamiltonian(terms)` | everything works, one circuit at a time |
| batched | `supports_batch = True`, `run_batch`, `states`, `z_expvals(wires)`, `energies(terms)` | the fast path |
| differentiable | `z_jacobian`, `energy_jacobian`, `is_differentiable` | the adjoint sweep |

Without the optional levels the helpers fall back to a loop and everything
still works — a backend that cannot differentiate simply gets the
parameter-shift rule. See `openqml/backends/pennylane_backend.py` for a worked
bridge.

## Local store and server mode

By default `openqml.config.server == "local"` and everything lives under
`~/.cache/openqml` (override with `OPENQML_CACHE_DIR`). To talk to a
deployment:

```python
openqml.config.server = "https://example.org/api/v1"
openqml.config.apikey = "..."
```

The client expects a small REST surface — `GET /{entity}`, `GET /{entity}/{id}`,
`POST /{entity}`, `DELETE /{entity}/{id}` for `datasets`, `tasks`, `flows`,
`runs` and `suites`, JSON in and out, bearer-token auth. The JSON is exactly
what `to_dict()` produces on each entity, so the local store doubles as the
schema reference for the server side.

`openqml.reset_local_store()` deletes everything published locally and re-seeds
the bundled entities.

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
