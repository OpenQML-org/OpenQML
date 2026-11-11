"""Independent dense/circuit references, bounded contractions and native training."""

import math

import numpy as np
import pytest

import openqml as q
from openqml._mps import MPSBackend
from openqml._tensor_contraction import TensorNetworkPlan
from openqml._tensor_train import TensorTrainPlan


def parity_weights(plan):
    w = np.zeros(plan.num_parameters)
    nodes = w[: 12 * plan.num_outputs * (plan.num_qubits - 1)].reshape(
        plan.num_outputs, plan.num_qubits - 1, 12
    )
    nodes[..., 0] = math.pi / 2
    nodes[..., 1] = math.pi / 2
    nodes[..., 2] = -math.pi / 2
    nodes[..., 4] = math.pi / 2
    nodes[..., 6] = -math.pi / 2
    return w


def dense_from_core_indices(plan, cores):
    matrix = np.empty((math.prod(plan.output_modes), math.prod(plan.input_modes)))
    for o in range(matrix.shape[0]):
        out_index = np.unravel_index(o, plan.output_modes)
        for i in range(matrix.shape[1]):
            in_index = np.unravel_index(i, plan.input_modes)
            value = np.ones((1, 1))
            for core, a, b in zip(cores, in_index, out_index, strict=True):
                value = value @ core[:, a, b, :]
            matrix[o, i] = value[0, 0]
    return matrix[: plan.out_features, : plan.in_features]


@pytest.mark.parametrize(
    "ins,outs,features",
    (
        ((2, 3, 2), (3, 2, 2), (11, 10)),
        ((2, 2, 2), (4, 1, 1), (8, 4)),
        ((1, 4, 2), (2, 2, 2), (7, 7)),
        ((3,), (2,), (3, 2)),
    ),
)
def test_mpo_contractions_match_independent_virtual_index_sums(ins, outs, features):
    plan = TensorTrainPlan(ins, outs, 2, *features)
    rng = np.random.default_rng(sum(ins) + sum(outs))
    cores = tuple(rng.normal(size=s) for s in plan.core_shapes)
    x = rng.normal(size=(2, 3, plan.in_features))
    dense = dense_from_core_indices(plan, cores)
    np.testing.assert_allclose(plan.apply(cores, x), x @ dense.T, atol=3e-14)
    if outs == (4, 1, 1):
        assert plan.direction == "right"


def random_circuit(n, depth=2):
    rng = np.random.default_rng(21 + n)
    builder = q.CircuitBuilder(n)
    theta = q.Parameter("theta")
    for _layer in range(depth):
        for i in range(n):
            builder.ry(theta * (i + 1) / n, i).rz(float(rng.normal()), i)
        order = rng.permutation(n)
        for a, b in zip(order[::2], order[1::2], strict=False):
            builder.rxx(float(rng.normal()), int(a), int(b)).rzz(theta.sin(), int(a), int(b))
    if n >= 3:
        builder.ccx(0, 1, 2)
    return builder.build().with_global_phase(q.Parameter("phase"))


@pytest.mark.parametrize("n", (1, 2, 3, 5))
def test_general_circuit_tensor_network_matches_independent_dense_states(n):
    circuit = random_circuit(n)
    obs = (
        q.PauliString("Y" + "I" * (n - 1)),
        q.PauliString("Z" * n),
        q.Hamiltonian((q.PauliString("X" * n, 0.3), q.PauliString("I" * n, -0.7))),
    )
    plan = TensorNetworkPlan(circuit, obs)
    values = {"theta": np.array([[-0.3], [0.7]]), "phase": np.array([[0.2, 0.4, 0.8]])}
    expected = q.expectations(circuit, obs, values, backend="numpy")
    np.testing.assert_allclose(plan.evaluate(values), expected, atol=5e-14)
    np.testing.assert_allclose(
        plan.evaluate({"theta": values["theta"]}), expected[:, :1], atol=5e-14
    )
    assert "phase" not in plan.active_parameters


def slicing_circuit():
    rng = np.random.default_rng(19)
    builder = q.CircuitBuilder(10)
    for _ in range(5):
        for i in range(10):
            builder.ry(float(rng.normal()), i).rz(float(rng.normal()), i)
        order = rng.permutation(10)
        for a, b in zip(order[::2], order[1::2], strict=True):
            builder.rxx(float(rng.normal()), int(a), int(b)).rzz(
                float(rng.normal()), int(a), int(b)
            )
    return builder.build()


@pytest.mark.parametrize("reuse", [False, True])
def test_exact_slicing_reduces_live_memory_and_never_prunes_amplitudes(reuse):
    circuit, obs = slicing_circuit(), (q.PauliString("Z" * 10),)
    whole = TensorNetworkPlan(circuit, obs, max_memory_mb=64, max_slices=1)
    budget = (whole._metadata_bytes + 0.6 * (whole.working_bytes - whole._metadata_bytes)) / 2**20
    sliced = TensorNetworkPlan(
        circuit, obs, max_memory_mb=budget, max_slices=64, reuse_intermediates=reuse
    )
    assert sliced.working_bytes <= budget * 2**20 < whole.working_bytes
    assert sliced.to_dict()["contractions"][0]["slices"] > 1
    np.testing.assert_allclose(
        sliced.evaluate(), q.expectations(circuit, obs, backend="numpy"), atol=1e-13
    )
    with pytest.raises(q.ResourceError):
        TensorNetworkPlan(circuit, obs, max_memory_mb=budget, max_slices=1)
    with pytest.raises(q.ResourceError, match="max_flops"):
        TensorNetworkPlan(circuit, obs, max_flops=1)


def test_optional_cotengra_search_and_unsupported_quantum_effects():
    pytest.importorskip("cotengra")
    circuit = random_circuit(4)
    obs = (q.PauliString("XYZZ"),)
    plan = TensorNetworkPlan(circuit, obs, optimizer="cotengra", search_trials=2)
    np.testing.assert_allclose(
        plan.evaluate({"theta": 0.4}),
        q.expectations(circuit, obs, {"theta": 0.4, "phase": 0.8}, backend="numpy"),
        atol=2e-14,
    )
    with pytest.raises(q.ValidationError, match="unitary"):
        TensorNetworkPlan(q.Circuit(1).amplitude_damping(0.2, 0), (q.PauliString("Z"),))


def test_sliced_execution_reuses_pure_gate_tensors_without_caching_readouts(monkeypatch):
    import openqml.simulator as simulator

    circuit, obs = slicing_circuit(), (q.PauliString("Z" * 10),)
    whole = TensorNetworkPlan(circuit, obs, max_memory_mb=64, max_slices=1)
    budget = (whole._metadata_bytes + 0.6 * (whole.working_bytes - whole._metadata_bytes)) / 2**20
    sliced = TensorNetworkPlan(circuit, obs, max_memory_mb=budget, max_slices=64)
    calls, original = [], simulator._gate_matrix

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(simulator, "_gate_matrix", counted)
    for _ in range(2):
        sliced.evaluate()
    assert len(calls) == 2 * len(circuit.operations)


def test_wide_entangled_circuit_contracts_and_differentiates_without_a_full_state(monkeypatch):
    torch = pytest.importorskip("torch")
    import openqml.simulator as simulator

    n = 48
    x, w = q.Parameter("x"), q.Parameter("w")
    builder = q.CircuitBuilder(n)
    for i in range(n):
        builder.ry(x, i).rz(0.2, i)
    for _ in range(2):
        for start in (0, 1):
            for i in range(start, n - 1, 2):
                builder.rxx(w, i, i + 1)
    circuit = builder.build()
    obs = (q.PauliString("X" + "Z" * (n - 1)),)

    def reference(a, b):
        return q.expectations(circuit, obs, {"x": a, "w": b}, backend=MPSBackend())[0]

    expected = reference(0.21, 0.37)
    epsilon = 1e-5
    derivatives = (
        (reference(0.21 + epsilon, 0.37) - reference(0.21 - epsilon, 0.37)) / (2 * epsilon),
        (reference(0.21, 0.37 + epsilon) - reference(0.21, 0.37 - epsilon)) / (2 * epsilon),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A global wavefunction was allocated.")

    monkeypatch.setattr(simulator, "_initial_state", forbidden)
    plan = TensorNetworkPlan(circuit, obs, max_memory_mb=16)
    a, b = (
        torch.tensor(0.21, dtype=torch.float64, requires_grad=True),
        torch.tensor(0.37, dtype=torch.float64, requires_grad=True),
    )
    result = plan.evaluate({"x": a, "w": b}, backend="torch", like=a)[0]
    grads = torch.autograd.grad(result, (a, b))
    assert float(result.detach()) == pytest.approx(expected, abs=1e-13)
    np.testing.assert_allclose([float(g) for g in grads], derivatives, atol=2e-7, rtol=2e-5)


def test_readme_quickstart_is_executable_and_completes_an_optimizer_step():
    pytest.importorskip("torch")
    import re
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", readme, re.S)
    assert blocks
    namespace = {}
    for block in blocks:
        exec(compile(block, "README.md", "exec"), namespace)
    model = namespace["model"]
    assert tuple(model.predict_proba(namespace["x"]).shape) == (32, 2)
    assert model.trainer.updates == sum(row["updates"] for row in model.history)
    assert model.trainer.updates >= 1
    assert np.isfinite(model.history[-1]["loss"])
    assert model.trainer.optimizer.state
    assert any(state["exp_avg"].count_nonzero() for state in model.trainer.optimizer.state.values())
