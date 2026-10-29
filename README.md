# OpenQML

OpenQML adds trainable quantum layers to PyTorch and JAX models and converts supported circuits between Qiskit, PennyLane, Cirq and OpenQASM through a shared intermediate representation.

## Install

```bash
python -m pip install openqml
```

This installs PyTorch, JAX, Flax, Optax, Qiskit, PennyLane, Cirq, Aer and OpenQASM tooling. All supported integrations are included by default and load when accessed.

## One OpenQML import

```python
import openqml as oq

circuit = oq.Circuit(2).ry(oq.Parameter("theta"), 0).cx(0, 1)
value = oq.expectation(circuit, oq.PauliString("ZI"), {"theta": 0.4})
qasm = oq.convert(circuit, target="openqasm3").value
restored = oq.convert(qasm, source="openqasm3", target="ir").value
```

Core operations use the `oq` namespace. Framework integrations load only when accessed:

| Namespace | Use |
| --- | --- |
| `oq.torch` | Quantum layers and native PyTorch APIs |
| `oq.jax`, `oq.flax`, `oq.optax` | Quantum layers, JIT, gradients and optimizers |
| `oq.qiskit`, `oq.cirq`, `oq.pennylane`, `oq.qiskit_aer` | Native SDKs and simulation |
| `oq.numpy`, `oq.sympy`, `oq.openqasm3` | Arrays, symbolic expressions and OpenQASM tooling |

## Hybrid training

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

One optimizer updates the classical and quantum parameters. For JAX, construct `oq.jax.QuantumLayer`, initialize weights with `layer.init(key)`, then call `layer(weights, inputs)`. Native tensor backends support higher derivatives; numerical bridges provide first-order JVPs and VJPs.

Prepared plans share symbolic work and duplicate readouts. Native readouts preserve higher derivatives; `oq.ShiftPlan` streams numerical products and reuses points only for declared deterministic backends. Job batches and constant caches are bounded.

## Circuit interoperability

Convert supported SDK circuit objects, IR, JSON and OpenQASM with `oq.convert(program, target="ir")`. Inspect `oq.plan_conversion(...)`, its `semantic_scope` and diagnostics before export. `plan.verify(...)` performs bounded numerical checks at selected bindings.

Arbitrary Python programs and their third-party dependencies are outside automatic conversion. `oq.inspect_python(text)` scans source without executing it and always marks whole-program conversion unsupported. It reports control flow, custom-instruction clues, opaque imports, noise/hardware context and analysis-budget limits. A completed scan does not prove runtime validity. `QuantumProgram.from_callable` executes the supplied factory and captures its returned circuit only.

Dynamic control and custom instructions require a supported IR/adapter representation. Represent noise explicitly as supported IR channels. External noise models, backend configuration and pulse calibrations are not carried over; unsupported context produces diagnostics.

## Resources and scientific scope

Dense defaults remain **20 qubits and 256 MiB**; conversion/preparation defaults generally allow **100,000 operations**. Operation counts constrain program processing, while dense quantum storage grows exponentially:

| Qubits | One complex128 state vector |
| ---: | ---: |
| 20 | 16 MiB |
| 30 | 16 GiB |
| 40 | 16 TiB |

`oq.estimate_resources(circuit)` reports storage and policy eligibility without allocating states. The budget model reserves four state buffers per batch row; local matrices, autodiff tapes, backend workspace and density branches have separate costs. `oq.ExecutionOptions.for_statevector(...)` derives an opt-in limit from an explicit budget.

For suitable structured circuits, use `oq.PauliPlan` or `oq.ExpectationPlan` to evaluate declared readouts from the all-zero state. Layers support `execution_method="pauli"` or `"planned"` with an explicit source-width limit. Exact propagation can still exceed its term/work budgets and will fail explicitly.

| Validated structured case | Qubits | Readouts |
| --- | ---: | --- |
| GHZ chain with a variable phase | 1,024 | Global correlations |
| Crossed Bell pairs | 1,024 | Global X/Z correlations |
| Connected square cluster graph | 256 | Local and checkerboard stabilizers |

The [scale audit](https://github.com/OpenQML-org/OpenQML/blob/main/benchmarks/results/scientific_cpu.json) records analytical checks at four bindings, small dense references, source fingerprints and approximate RSS observations. Tests cover native first/second derivatives and structured hybrid training. These results concern specific families and readouts, rather than generic simulation at those widths. A separate 100,000-operation IR check performs no quantum simulation; a growth-limit case verifies explicit rejection.

Use `oq.CircuitBuilder(...).build()` for long construction loops; immutable `Circuit.append()` copies its tuple. Catch `oq.OpenQMLError` or a named subclass.

## Installation and platform evidence

The complete dependency bundle has a substantial installation cost. One [measured Linux environment](https://github.com/OpenQML-org/OpenQML/blob/main/benchmarks/results/installation_linux.json) contained **6.39 GiB** of logical runtime files across 97 distributions, excluding pip/build caches. Sizes and selected binaries vary by platform. Intel macOS automatically selects NumPy 1.x to match its available PyTorch wheel.

`oq.environment_report(check_imports=True, include_sizes=True)` reports dependency conflicts, isolated imports, the Torch/NumPy bridge and installed file sizes. It installs nothing and does not establish accelerator compatibility.

| Validated CPU CI target | Python | Checks |
| --- | --- | --- |
| Linux x86_64 | 3.10, 3.11, 3.12 | Full suite and package build |
| Windows x86_64 | 3.12 | Default installation, backends, conversions, gradients and diagnostics |
| macOS Apple Silicon | 3.12 | Same portability checks |
| macOS Intel | 3.11 | Same portability checks |

[CI](https://github.com/OpenQML-org/OpenQML/actions/workflows/ci.yml) archives resolved dependencies, environment reports and scientific checks. Other combinations and CUDA/Metal/ROCm execution remain unverified. Linux RSS measurements are approximate call windows; cold imports/compilation are reported separately from warm execution.

## Architecture

![OpenQML project architecture](https://raw.githubusercontent.com/OpenQML-org/OpenQML/main/docs/architecture.svg)

[Apache license](https://github.com/OpenQML-org/OpenQML/blob/main/LICENSE)
