# OpenQML

[PyPI](https://pypi.org/project/openqml/) · [Documentation](https://github.com/OpenQML-org/OpenQML/blob/main/docs/api.md) · [CI](https://github.com/OpenQML-org/OpenQML/actions/workflows/ci.yml)

[Contributing](CONTRIBUTING.md) · [Governance](GOVERNANCE.md) · [Security](SECURITY.md)

## Quantum–classical hybrid AI

Build and train hybrid neural networks with PyTorch, JAX or Flax. Use `Model.fit` and `predict` for supervised tasks, or compose convolution, attention, graph, recurrent and generative architectures with native layers. Validation loss, early stopping and portable checkpoints keep model selection and continued training in the same workflow. [Explore hybrid AI](https://github.com/OpenQML-org/OpenQML/blob/main/docs/models.md).

Train native quantum blocks with resumable, distance-aware Quantum Natural Muon
and classical blocks with AdamW through one `hybrid_qnm` optimizer. Microbatch
accumulation and DDP share an empirical, matrix-free quantum Fisher metric.
Sample chunking bounds the active state workspace during geometry products.
External Torch/JAX layers can pack forward and parameter-shift rows into one
provider submission and reuse their linearization during backward.

## Quantum–classical hybrid quantitative research

Train scenario distributions, allocate constrained portfolios and project decisions onto explicit CVaR budgets with a funded cash position. Low-rank factor solves and native PyTorch/JAX derivatives connect financial objectives to hybrid models. Purged walk-forward refits and self-financing accounting preserve chronological evaluation, execution delays, fees and market impact. [Explore quantitative research](https://github.com/OpenQML-org/OpenQML/blob/main/docs/finance.md).

Learn scenario probabilities, Wasserstein radii and anisotropic transport metrics.
`ScenarioPlan.execution_loss` connects proper-score calibration to after-cost
decisions. Exact piecewise quadratic projections fund proportional fees and
market impact while enforcing CVaR or expectile budgets.
`ScenarioPlan.tree` adds funded multi-stage policies and nested conditional risk.

## Cross-framework quantum programming

Connect Qiskit, Cirq, PennyLane and OpenQASM through a shared circuit representation. Compile grouped switch cases, finite loops and atomic classical guards; reuse shared definitions and symbolic expressions. Prepare conversions once, stream parameter sweeps and verify their semantics against round-trip or native execution. [Explore interoperability](https://github.com/OpenQML-org/OpenQML/blob/main/docs/interoperability.md).

Typed Qiskit logical, bitwise and unsigned comparison predicates lower to bounded
shared decision graphs. Their branch selection completes before either payload
executes, including payloads that overwrite a measured predicate input.
Fixed-width unsigned arithmetic includes division with symbolically proved
nonzero divisors. Correlated selector proofs admit bounded dynamic shifts and
bit indexing while preserving atomic branch selection.
Deferred Qiskit inverses, controls and numeric powers expand through the same
checked conversion routes.

Simulate physical noise with amplitude/phase damping, depolarization and custom Kraus channels. Correct independent readout errors or extrapolate measured noise scales with explicit conditioning and sampling-overhead limits. Caller-initialized Qiskit V2 primitives connect real hardware; see [execution limits and mitigation](https://github.com/OpenQML-org/OpenQML/blob/main/docs/runtime.md#physical-noise-and-error-mitigation).

## Install

Python 3.10 or later:

```bash
python -m pip install -U openqml
```

The core depends on NumPy, packaging and psutil. Training frameworks, quantum SDKs and execution engines load on demand. Install the integrations you use, such as `openqml[torch]`, `openqml[jax]`, `openqml[flax]` or `openqml[qiskit,cirq]`.

For NVIDIA GPU simulation, `openqml[aer-gpu]` on supported Linux systems connects the shared IR and scheduler to Aer's C++/CUDA engine, including optional cuStateVec and cuTensorNet paths. Native PyTorch/JAX training keeps tensors and gradients on the selected GPU. Caller-initialized Qiskit V2 primitives support asynchronous QPU submission with recoverable job receipts. See [GPU and QPU execution](docs/runtime.md).

OpenQML implements circuit lowering, fusion planning, readout arithmetic and resource
scheduling. Aer owns the integrated C++/CUDA simulator; PyTorch/JAX and their compilers
provide tensor kernels for OpenQML's own execution algorithms. These are separate
execution paths. `AerBackend.execution_info()` and `CUDABackend.execution_info()`
describe the configured engine and kernel provider without initializing hardware.

## Quick start

Allocate from observed returns:

```python
import openqml as oq

returns = [[0.01, -0.02], [-0.01, 0.03], [0.02, 0.01]]
portfolio = oq.finance.allocate(returns, caps=0.7, fees=0.001)
print(portfolio.weights)
```

Train a hybrid model after installing `openqml[torch]`:

```python
import openqml as oq

x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 8
y = [0, 1, 1, 0] * 8
model = oq.Model(task="classification", hidden_features=8, depth=2)
model.fit(x, y, epochs=10)
print(model.predict(x[:2]))
```

Select `framework="jax"` or `framework="flax"` to change training frameworks. `backend` chooses circuit execution; `device` chooses native tensor placement. Qubit counts follow the selected backend's available resources.

Convert a circuit with the core installation:

```python
import openqml as oq

circuit = oq.Circuit(2).h(0).cx(0, 1)
result = oq.convert(circuit, target="openqasm3")
print(result.value)
```

`convert` imports, checks and exports in one call. Native SDK targets require
their corresponding integration. Use `plan_conversion` when repeatedly binding
or exporting the same source.

Hybrid models, financial operators and circuit execution share a
native compute kernel. It centralizes array placement, matrix products,
contraction planning, numerical checks and memory admission. Configure a common
workspace ceiling for an isolated execution scope:

```python
import openqml as oq

with oq.compute_context(policy=oq.ComputePolicy(max_memory_mb=512)):
    result = oq.expectations(oq.Circuit(1).h(0), [oq.PauliString("Z")])
```

Use `compile_compute` for checked Torch/JAX compilation. Iterative solves certify
their true residual before returning a solution. Operation-specific budgets
remain available. Shared caches hold bounded,
immutable planning metadata; trainable tensors stay in each forward pass.
[Compute policies and numerical contracts](https://github.com/OpenQML-org/OpenQML/blob/main/docs/compute.md).

`Circuit.append` shares immutable prefix chunks. Local NumPy execution fuses
single/two-wire gate groups and applies them through tensor views. Use
`compile_circuit` to compile a complete native circuit and its declared readouts.
Implementations are organized in responsibility packages; the root API and
published module imports remain available.

[API](https://github.com/OpenQML-org/OpenQML/blob/main/docs/api.md) · [Training and validation](https://github.com/OpenQML-org/OpenQML/blob/main/docs/hybrid-training.md) · [Quantitative research](https://github.com/OpenQML-org/OpenQML/blob/main/docs/finance.md) · [Interoperability](https://github.com/OpenQML-org/OpenQML/blob/main/docs/interoperability.md) · [Runtime](https://github.com/OpenQML-org/OpenQML/blob/main/docs/runtime.md)

## License

[Apache-2.0](LICENSE)
