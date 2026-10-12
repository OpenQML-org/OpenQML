# OpenQML Python SDK

OpenQML is a NumPy-native Python library for quantum circuits, toy quantum learning
models, and reproducible local quantum/classical comparisons, inspired by
[openqml.org](https://openqml.org). It is an independent SDK, not the website's
JavaScript simulator or an implementation of every model in its catalog.

Requires Python 3.10+. The core dependency is NumPy; import does not contact any
service. The current alpha version has not been published to PyPI. Extract the
source bundle and run this in the directory containing pyproject.toml:

```bash
python -m pip install .
```

The delivery bundle also includes a wheel under `dist/`, which can be installed
directly without Git.

```python
import openqml as q

state = q.Circuit(2).h(0).cx(0, 1).run()
print(state.sample(1024, seed=7))
print(state.expectation("ZZ"))
print(state.reduced_density([0]))

data = q.load_dataset("moons", seed=7)
X_train, X_test, y_train, y_test = data.split(seed=7)
model = q.QuantumKernelClassifier().fit(X_train, y_train)
print(model.score(X_test, y_test))

comparison = q.compare_models("moons", seeds=(1, 2, 3), epochs=40)
print(comparison.summary())
comparison.save("benchmark-results.json")
```

The models expose `fit/predict/score/get_params`. The variational classifier and
regressor use exact gate-wise parameter-shift derivatives with the loss chain
rule. QuantumKernelClassifier is a multiclass kernel ridge classifier, not an
SVM; its softmax scores are not calibrated probabilities. Classical logistic,
linear ridge, and Fourier ridge comparators are included.

The synthetic datasets and suites run offline. Benchmarks preserve data hashes,
generation options, split seeds, configurations, versions, timings, and metrics.
The SDK protocol is `openqml-sdk-bench-v1`, distinct from the website's protocol.
Default comparisons are not parameter matched and imply no quantum advantage.

q0 is the least significant bit and is rightmost in strings. Matrix wire lists
put the least significant local wire first. All rotations use `(angle, wire)`.

Sparse states such as GHZ can run up to 40 qubits within explicit support/work
budgets. Arbitrary 40-qubit entangled states are not guaranteed to fit. Full
vectors/probabilities default to at most 18 requested wires; exact noise/reset
uses density matrices on at most 8 total qubits. Use selected-wire readouts for
large sparse states. Budgets are conservative estimates, not process-wide
operating-system memory limits or wall-clock timeouts.

OpenQASM exchange supports a strict QASM 2 subset and rejects unsupported
constructs. `circuit.to_qiskit()` is an optional adapter, with no hardware job
submission. Install Qiskit separately or install `.[qiskit]` from a source checkout.

```bash
python -m openqml demo --qubits 40 --seed 7
python -m openqml datasets
python -m openqml benchmark --task sine --seeds 1 2 3 --epochs 40
python -m pip install -e ".[dev]"
python -m pytest -q
```

See [API documentation](api.md), the [examples](../examples/), and the
[contribution guide](../CONTRIBUTING.md) for contracts and validation.
