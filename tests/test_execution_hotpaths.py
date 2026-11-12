import pickle
from dataclasses import asdict, fields, replace

import numpy as np
import pytest

import openqml as oq


def compiled_derivative(xp, backend, circuit, observable, run, name, options):
    transform = xp.func.jacrev if backend == "torch" else xp.jacrev
    if backend == "torch" and tuple(int(v) for v in xp.__version__.split(".")[:2]) < (2, 4):
        with pytest.raises(oq.ValidationError, match="PyTorch >= 2.4"):
            oq.compile_circuit(circuit, observable, backend=backend, transform=transform, **options)
        value = xp.tensor(0.2, requires_grad=True)
        return xp.autograd.grad(run({name: value}).sum(), value)[0].reshape(1)
    return oq.compile_circuit(circuit, observable, backend=backend, transform=transform, **options)(
        {name: 0.2}
    )[name]


def test_persistent_append_validates_only_new_operations_and_preserves_forks(monkeypatch):
    import openqml.ir as ir

    validate, calls, count = ir._validate_operation, 0, 8000

    def bounded(operation, width, classical):
        nonlocal calls
        calls += 1
        assert calls <= count + 3, "Appending revalidated an immutable prefix"
        return validate(operation, width, classical)

    monkeypatch.setattr(ir, "_validate_operation", bounded)
    shared = oq.Operation("x", (0,))
    circuit = oq.Circuit(2)
    for _ in range(count):
        circuit = circuit.append(shared)
    fork = circuit.h(1)
    other = circuit.z(0)
    assert calls == count + 2
    assert isinstance(circuit.operations, tuple)
    assert len(circuit.operations) == count
    assert fork.operations[:-1] == other.operations[:-1] == circuit.operations
    with pytest.raises(oq.ValidationError, match="undeclared"):
        fork.x(2)
    assert len(fork.operations) == count + 1


def test_persistent_circuit_keeps_dataclass_and_deep_prefix_serialization_contracts():
    circuit = oq.Circuit(2).h(0).cx(0, 1)
    assert fields(oq.Circuit)[1].default == ()
    assert replace(circuit, num_clbits=3).operations == circuit.operations
    assert replace(circuit, operations=()).operations == ()
    assert asdict(circuit)["operations"][0]["name"] == "h"
    assert hash(circuit) == hash(oq.Circuit.from_json(circuit.to_json()))
    operation = oq.Operation("x", (0,))
    for _ in range(40_000):
        circuit = circuit.append(operation)
    restored = pickle.loads(pickle.dumps(circuit))
    assert restored == circuit
    assert restored.operations[-1] == operation


def test_pre_prefix_pickle_state_keeps_public_fields():
    circuit = oq.Circuit(2).h(0).cx(0, 1)
    restored = object.__new__(oq.Circuit)
    restored.__setstate__(
        {
            name: getattr(circuit, name)
            for name in ("num_qubits", "operations", "num_clbits", "global_phase")
        }
    )
    assert restored == circuit
    assert restored.x(1).operations[:-1] == circuit.operations


@pytest.mark.parametrize("wires", [(0,), (3,), (0, 3), (3, 0), (1, 2)])
def test_numpy_local_views_never_gather_input(wires, monkeypatch):
    import openqml.simulator as sim

    rng = np.random.default_rng(4)
    state = rng.normal(size=(2, 3, 16)) + 1j * rng.normal(size=(2, 3, 16))
    size = 1 << len(wires)
    matrix = rng.normal(size=(2, 1, size, size)) + 1j * rng.normal(size=(2, 1, size, size))
    order = tuple(wires) + tuple(q for q in range(4) if q not in wires)
    axes = (0, 1) + tuple(2 + q for q in order)
    inverse = (0, 1) + tuple(2 + order.index(q) for q in range(4))
    expected = matrix @ state.reshape((2, 3) + (2,) * 4).transpose(axes).reshape(2, 3, size, -1)
    expected = expected.reshape((2, 3) + (2,) * 4).transpose(inverse).reshape(state.shape)

    def reject(*args, **kwargs):
        pytest.fail("A one/two-wire gate gathered an indexed input buffer")

    monkeypatch.setattr(np, "take", reject)
    actual = sim._apply_numpy(state, matrix, wires, 4)
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    assert not np.shares_memory(actual, state)


def test_default_fusion_reduces_state_passes_and_keeps_parameter_batches(monkeypatch):
    import openqml.simulator as sim

    angle = oq.Parameter("theta")
    builder = oq.CircuitBuilder(4)
    for _ in range(20):
        builder.h(3).ry(angle, 0).cx(3, 0)
    circuit = builder.build().with_global_phase(angle / 7)
    bindings = {"theta": np.array([0.2, -0.1])}
    expected = oq.statevector(circuit, bindings, backend="numpy", fusion=False)
    original, calls = sim._apply_matrix, []

    def applied(state, matrix, wires, width, ops):
        calls.append(wires)
        return original(state, matrix, wires, width, ops)

    monkeypatch.setattr(sim, "_apply_matrix", applied)
    actual = oq.statevector(circuit, bindings, backend="numpy")
    assert len(calls) == 1
    np.testing.assert_allclose(actual, expected, atol=2e-14)


def test_fusion_small_budget_falls_back_without_rejecting_original_execution():
    circuit = oq.Circuit(1).h(0).x(0).h(0)
    budget = 128 / 2**20
    with oq.compute_context(policy=oq.ComputePolicy(max_memory_mb=budget)):
        actual = oq.statevector(circuit, backend="numpy", max_memory_mb=budget)
    np.testing.assert_allclose(actual, [1, 0], atol=1e-14)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_whole_graph_circuit_compilation_checks_results_and_derivatives(backend):
    xp = pytest.importorskip(backend)
    circuit = oq.Circuit(2).ry(oq.Parameter("theta"), 0).cx(0, 1)
    options = {"compiler": "eager"} if backend == "torch" else {}
    run = oq.compile_circuit(circuit, [oq.PauliString("ZI")], backend=backend, **options)
    value = run({"theta": 0.2})
    result = value.detach().numpy() if backend == "torch" else np.asarray(value)
    np.testing.assert_allclose(result, [np.cos(0.2)], atol=2e-7)
    gradient = compiled_derivative(
        xp, backend, circuit, [oq.PauliString("ZI")], run, "theta", options
    )
    gradient = gradient.detach().numpy() if backend == "torch" else np.asarray(gradient)
    np.testing.assert_allclose(gradient, [-np.sin(0.2)], atol=2e-7)
    with pytest.raises(oq.NumericalError):
        run({"theta": float("nan")})
    with pytest.raises(oq.ValidationError):
        run({"missing": 0.2})


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize(
    "channel,expected,derivative",
    [
        ("amplitude_damping", np.sqrt(0.8), -0.5 / np.sqrt(0.8)),
        ("phase_damping", np.sqrt(0.8), -0.5 / np.sqrt(0.8)),
        ("depolarizing", 1 - 0.8 / 3, -4 / 3),
        ("bit_flip", 1, 0),
        ("phase_flip", 0.6, -2),
    ],
)
def test_compiled_physical_channel_retains_domain_checks_and_native_gradients(
    backend, channel, expected, derivative
):
    xp = pytest.importorskip(backend)
    circuit = getattr(oq.Circuit(1).h(0), channel)(oq.Parameter("gamma"), 0)
    options = {"compiler": "eager"} if backend == "torch" else {}
    run = oq.compile_circuit(circuit, [oq.PauliString("X")], backend=backend, **options)
    value = run({"gamma": 0.2})
    value = value.detach().numpy() if backend == "torch" else np.asarray(value)
    np.testing.assert_allclose(value, [expected], atol=2e-7)
    gradient = compiled_derivative(
        xp, backend, circuit, [oq.PauliString("X")], run, "gamma", options
    )
    gradient = gradient.detach().numpy() if backend == "torch" else np.asarray(gradient)
    np.testing.assert_allclose(gradient, [derivative], atol=2e-7)
    with pytest.raises(oq.ValidationError):
        run({"gamma": 1.2})


def test_compiled_batch_admission_precedes_input_conversion():
    torch = pytest.importorskip("torch")
    circuit = oq.Circuit(2).ry(oq.Parameter("theta"), 0)
    run = oq.compile_circuit(
        circuit,
        [oq.PauliString("ZI")],
        backend="torch",
        compiler="eager",
        max_memory_mb=256 / 2**20,
    )
    with pytest.raises(oq.ResourceError):
        run({"theta": torch.ones(2)})
    noisy = oq.Circuit(1).amplitude_damping(oq.Parameter("gamma"), 0)
    run = oq.compile_circuit(
        noisy, [oq.PauliString("Z")], backend="torch", compiler="eager", max_memory_mb=256 / 2**20
    )
    with pytest.raises(oq.ResourceError):
        run({"gamma": torch.ones(2) * 0.2})


def test_independent_checked_programs_do_not_share_recompilation_budget():
    torch = pytest.importorskip("torch")
    programs = [
        oq.compile_compute(
            lambda value, offset=index: value.square() + offset,
            backend="torch",
            compiler="eager",
            fullgraph=True,
        )
        for index in range(12)
    ]
    for index, program in enumerate(programs):
        torch.testing.assert_close(program(torch.tensor(2.0)), torch.tensor(4.0 + index))


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_compiled_classical_feedback_and_noise_match_density_execution(backend):
    pytest.importorskip(backend)
    theta = oq.Parameter("theta")
    true = oq.Circuit(2).rx(theta / 2, 1).reset(0)
    false = oq.Circuit(2).ry(theta, 1)
    circuit = (
        oq.Circuit(2)
        .ry(theta, 0)
        .measure(0, 0)
        .if_else(oq.ClassicalCondition((0,)), true, false)
        .phase_flip(0.1, 1)
    )
    observable = [oq.PauliString("IX"), oq.PauliString("IZ")]
    options = {"compiler": "eager"} if backend == "torch" else {}
    run = oq.compile_circuit(circuit, observable, backend=backend, **options)
    values = {"theta": np.array([0.2, 0.7])}
    actual = run(values)
    actual = actual.detach().numpy() if backend == "torch" else np.asarray(actual)
    np.testing.assert_allclose(
        actual, oq.expectations(circuit, observable, values, backend="numpy"), atol=2e-7
    )
