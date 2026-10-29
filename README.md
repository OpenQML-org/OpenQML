# OpenQML

OpenQML adds trainable quantum layers to PyTorch and JAX models and compiles supported quantum circuits across Qiskit, PennyLane, Cirq and OpenQASM with explicit semantic checks.

## Install and use

```bash
python -m pip install openqml
```

All supported integrations are included and load on access through one namespace.

```python
import openqml as oq

circuit = oq.Circuit(2).ry(oq.Parameter("theta"), 0).cx(0, 1)
compiled = oq.compile_verified(circuit, target="openqasm3")
plan = oq.plan_execution(circuit, (oq.PauliString("ZI"),))
values = plan.evaluate({"theta": 0.4})
```

Native APIs are available through `oq.torch`, `oq.jax`, `oq.flax`, `oq.optax`, `oq.numpy` and the SDK namespaces, including `oq.qiskit`, `oq.pennylane`, `oq.cirq` and `oq.qiskit_aer`.

## Hybrid learning

```python
import openqml as oq

model = oq.torch.nn.Sequential(
    oq.torch.nn.Linear(4, 2),
    oq.torch.QuantumLayer.data_reuploading(2, depth=1),
    oq.torch.nn.Linear(2, 1),
)
optimizer = oq.torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss = model(oq.torch.randn(8, 4)).square().mean()
loss.backward()
optimizer.step()
```

One optimizer updates classical and quantum weights. JAX layers use `layer.init(key)` and `layer(weights, inputs)`; Flax and Optax are supported. Native tensor/Pauli execution preserves higher derivatives. Numerical SDK and MPS bridges provide first-order JVPs and VJPs.

## Verifiable circuit compilation

| API | Guarantee |
| --- | --- |
| `oq.prove_equivalence(a, b)` | Bounded algebraic equivalence for supported normal forms and all defined parameter bindings |
| `oq.compile_verified(c, target=...)` | Exports with an algebraic certificate; refuses unproved conversions |
| `oq.plan_conversion(...).verify(mode="native")` | Independent target-runtime comparison over a complete input basis at explicit bindings |
| `oq.semantic_coverage()` | Representative framework/IR feature matrix and diagnostics |

The proof kernel uses affine Boolean phase polynomials, exact word rewrites and optional projective Clifford conjugation. Strict mode compares global phase. Projective mode explicitly ignores it. Stored floating constants remain exact binary rationals; SDK rounding or unsupported identities can leave a conversion **unproved**, which does not establish inequivalence. Certificates cover the adapter's IR interpretation; native runtime checks remain separate.

Arbitrary Python programs are outside automatic translation. `oq.inspect_python(text)` diagnoses source without executing it. Dynamic control, custom matrices and noise require supported IR representations. External noise models, pulse calibrations and hardware context are diagnosed rather than silently preserved.

## Adaptive execution

`oq.plan_execution(...)` reports its method, rejected candidates and memory admission before execution. Existing numerical APIs also accept `backend="auto"`.

| Method | Admission |
| --- | --- |
| Stabilizer | Named Clifford gates and bounded tableau storage |
| Pauli | Bounded exact symbolic propagation of declared readouts |
| OpenQML MPS | Nearest-neighbor one/two-wire circuits; planned full-SVD dimensions, no singular-value truncation |
| Dense vector/density | Memory-derived capacity, including classical branches |
| CUDA dense | Explicit dense device preference and sufficient available GPU memory |

Direct vector execution and quantum layers have no fixed 20-qubit ceiling. The default vector budget remains 256 MiB; `max_qubits` is an optional additional cap. Four complex128 buffers per row require:

| Qubits | Working budget |
| ---: | ---: |
| 22 | 256 MiB |
| 24 | 1 GiB |
| 26 | 4 GiB |

Adaptive plans default to one quarter of observed host/container availability, capped at 4 GiB. Autodiff tapes and backend costs still matter; admission models are not process RSS limits. Difficult non-Clifford, noisy and highly entangled circuits can exhaust every method.

## Evidence and installation cost

The [compiler/execution audit](https://github.com/OpenQML-org/OpenQML/blob/main/benchmarks/results/compiler_execution_cpu.json) records randomized differential checks, symbolic certificates, a 1,024-qubit Clifford case, independently checked 128-qubit non-Clifford MPS readouts and measured dense execution. These are representation-specific results, not general large-qubit simulation guarantees.

[CI](https://github.com/OpenQML-org/OpenQML/actions/workflows/ci.yml) covers Linux, Windows and both macOS architectures. GPU execution requires suitable hardware and is skipped on CPU runners. The full SDK bundle is substantial: one [measured Linux installation](https://github.com/OpenQML-org/OpenQML/blob/main/benchmarks/results/installation_linux.json) occupies about 6.39 GiB. `oq.environment_report(check_imports=True, include_sizes=True)` reports actual dependencies, compatibility and footprint.

## Architecture

![OpenQML project architecture](https://raw.githubusercontent.com/OpenQML-org/OpenQML/main/docs/architecture.svg)

[Apache license](https://github.com/OpenQML-org/OpenQML/blob/main/LICENSE)
