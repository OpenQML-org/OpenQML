"""Independent analytic/reference checks for IR v4 quantum and classical effects."""

from dataclasses import replace

import numpy as np
import pytest

from openqml import (
    BackendCapabilities,
    BackendError,
    Circuit,
    ClassicalCondition,
    Parameter,
    PauliString,
    ResourceError,
    ValidationError,
    circuit_stats,
    decompose,
    density_matrix,
    execute,
    execute_batch,
    expectation,
    expectations_batch,
    parameter_shift,
    sample,
    simplify,
    statevector,
)
from openqml.ir import GATES, UNITARY_GATES


@pytest.mark.parametrize("gate", sorted(UNITARY_GATES - {"unitary"}))
def test_density_unitary_rules_match_pure_state_outer_product(gate):
    arity, nparams = GATES[gate]
    c = (
        Circuit(3)
        .h(0)
        .ry(0.71, 2)
        .cx(2, 1)
        .gate(
            gate,
            *(2, 0, 1)[:arity],
            params=(0.31, -0.52, 0.19)[:nparams],
        )
        .with_global_phase(0.29)
    )
    vector = statevector(c)
    np.testing.assert_allclose(density_matrix(c), np.outer(vector, vector.conj()), atol=2e-14)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("amplitude_damping", [[0.65, np.sqrt(0.7) / 2], [np.sqrt(0.7) / 2, 0.35]]),
        ("phase_damping", [[0.5, np.sqrt(0.7) / 2], [np.sqrt(0.7) / 2, 0.5]]),
        ("depolarizing", [[0.5, 0.3], [0.3, 0.5]]),
        ("bit_flip", [[0.5, 0.5], [0.5, 0.5]]),
        ("phase_flip", [[0.5, 0.2], [0.2, 0.5]]),
    ],
)
def test_named_channels_have_explicit_probability_conventions(name, expected):
    c = getattr(Circuit(1).h(0), name)(0.3, 0)
    np.testing.assert_allclose(density_matrix(c), expected, atol=2e-14)
    assert execute(c).statevector is None
    assert sum(sample(c, 53, seed=4).values()) == 53
    with pytest.raises(ValidationError):
        statevector(c)


def test_arbitrary_complex_kraus_on_reversed_nonadjacent_wires():
    rng = np.random.default_rng(51)
    isometry, _ = np.linalg.qr(rng.normal(size=(12, 4)) + 1j * rng.normal(size=(12, 4)))
    kraus = isometry.reshape(3, 4, 4)
    preparation = Circuit(3).h(0).ry(0.7, 1).cx(0, 2).p(0.4, 2)
    vector = statevector(preparation)
    # Expand local operators by direct basis-index arithmetic, independently of
    # the simulator's tensor contraction and row/column axis conventions.
    expected = np.zeros((8, 8), complex)
    for local in kraus:
        full = np.zeros((8, 8), complex)
        for source in range(8):
            bits = [int(b) for b in format(source, "03b")]
            local_source = 2 * bits[2] + bits[0]
            for target in range(4):
                output = [target % 2, bits[1], target // 2]
                full[int("".join(map(str, output)), 2), source] = local[target, local_source]
        state = full @ vector
        expected += np.outer(state, state.conj())
    c = preparation.channel(kraus, 2, 0)
    np.testing.assert_allclose(density_matrix(c), expected, atol=2e-14)
    assert Circuit.from_json(c.to_json()) == c
    assert np.linalg.eigvalsh(expected).min() >= -1e-14


def test_reset_discards_entanglement_and_keeps_other_wire_mixed():
    c = Circuit(2).h(0).cx(0, 1).reset(0)
    np.testing.assert_allclose(density_matrix(c), np.diag([0.5, 0.5, 0, 0]), atol=1e-14)
    assert expectation(c, PauliString("ZI")) == pytest.approx(1)
    assert expectation(c, PauliString("IZ")) == pytest.approx(0)


def feedback():
    # Selecting true once is essential: its measurement overwrites c0 to zero
    # before X1, which must still execute.
    return (
        Circuit(2)
        .h(0)
        .measure(0, 0)
        .if_else(
            ClassicalCondition((0,), 1),
            Circuit(2).x(0).measure(0, 0).x(1),
        )
        .measure(1, 1)
    )


def test_atomic_feedback_preserves_records_and_unconditional_ensemble():
    c = feedback()
    np.testing.assert_allclose(density_matrix(c), np.diag([0.5, 0.5, 0, 0]), atol=1e-14)
    result = execute(c, shots=5000, seed=6)
    assert set(result.counts) == {"00", "01"}
    assert result.counts["01"] / 5000 == pytest.approx(0.5, abs=0.025)
    np.testing.assert_allclose(result.density_matrix, density_matrix(c))
    assert Circuit.from_json(c.to_json()) == c


def test_nested_multibit_conditions_compose_and_preserve_symbolic_blocks():
    w = Parameter("w")
    inner = (
        Circuit(2)
        .if_else(
            ClassicalCondition((1,), 0),
            Circuit(2).ry(w, 0),
            Circuit(2).x(1),
        )
        .with_global_phase(w / 3)
    )
    c = Circuit(2).x(0).measure(0, 0).if_else(ClassicalCondition((1, 0), 2), inner)
    assert c.parameters == ("w",)
    remapped = Circuit(3).compose(c, qubits=(2, 0), clbits=(3, 1))
    assert remapped.operations[-1].condition == ClassicalCondition((1, 3), 2)
    assert remapped.parameters == ("w",)
    remapped = remapped.bind({"w": 0.6})
    expected = Circuit(3).x(2).ry(0.6, 2)
    np.testing.assert_allclose(density_matrix(remapped), density_matrix(expected), atol=1e-14)
    lowered = decompose(c, basis={"p", "ry", "cx"})
    np.testing.assert_allclose(density_matrix(lowered, {"w": 0.6}), density_matrix(c, {"w": 0.6}))
    np.testing.assert_allclose(
        density_matrix(simplify(c), {"w": 0.6}), density_matrix(c, {"w": 0.6})
    )


@pytest.mark.parametrize(
    "name",
    [
        "amplitude_damping",
        "phase_damping",
        "depolarizing",
        "bit_flip",
        "phase_flip",
    ],
)
def test_channel_derivatives_with_shared_nonlinear_parameters_and_feedback(name):
    x = Parameter("x")
    p = 0.1 + x.sin() ** 2 / 3
    branch = getattr(Circuit(2).ry(x * x, 1), name)(p, 1)
    c = getattr(Circuit(2).ry(x, 0).measure(0, 0), name)(p, 0).if_else(
        ClassicalCondition((0,), 1),
        branch,
        Circuit(2).rx(x / 2, 1),
    )
    for paulis in ("XI", "IY", "ZZ"):
        ob = PauliString(paulis)
        actual = parameter_shift(c, ob, {"x": 0.43})["x"]
        finite = (expectation(c, ob, {"x": 0.430001}) - expectation(c, ob, {"x": 0.429999})) / 2e-6
        assert actual == pytest.approx(finite, abs=3e-9)


def test_probability_endpoints_and_invalid_kraus_conditions():
    p = Parameter("p")
    for value in (-0.01, 1.01, np.nan, True):
        with pytest.raises(ValidationError):
            Circuit(1).amplitude_damping(value, 0)
    for kraus in ([], [np.eye(2) * 2], [np.ones((3, 3))], [np.eye(2) * np.nan]):
        with pytest.raises(ValidationError):
            Circuit(1).channel(kraus, 0)
    for bits, value in (((), 0), ((0, 0), 1), ((0,), 2), ((-1,), 0)):
        with pytest.raises(ValidationError):
            ClassicalCondition(bits, value)
    c = Circuit(1).h(0).amplitude_damping(p, 0)
    with pytest.raises(ValidationError, match="below 1"):
        parameter_shift(c, PauliString("X"), {"p": 1})
    # Linear Pauli mixtures have finite, exact probability derivatives at both ends.
    for value in (0.0, 1.0):
        derivative = parameter_shift(Circuit(1).bit_flip(p, 0), PauliString("Z"), {"p": value})
        assert derivative["p"] == pytest.approx(-2)


def test_density_and_classical_branch_budgets_precede_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("allocation happened before resource validation")

    monkeypatch.setattr(np, "zeros", forbidden)
    with pytest.raises(ResourceError, match="Density"):
        density_matrix(Circuit(10).reset(0), max_memory_mb=1)
    c = Circuit(3).h(0).h(1).measure(0, 0).measure(1, 1)
    c = c.if_else(ClassicalCondition((0, 1), 3), Circuit(3).x(2))
    with pytest.raises(ResourceError, match="branch"):
        density_matrix(c, max_branches=2)
    with pytest.raises(ResourceError):
        density_matrix(Circuit(100_000).reset(0))


def test_mixed_numeric_batch_vectorizes_and_chunks_without_psd_eigensolves(monkeypatch):
    import openqml.density as module

    original, calls = module._simulate, []

    def counted(*args, **kwargs):
        calls.append(kwargs.get("batch_shape"))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_simulate", counted)
    x = Parameter("x")
    c = (
        Circuit(3)
        .ry(x, 0)
        .amplitude_damping(0.2, 0)
        .measure(0, 0)
        .if_else(
            ClassicalCondition((0,), 1),
            Circuit(3).ry(x * x, 2),
        )
    )
    rows = [{"x": value} for value in np.linspace(-0.7, 0.8, 25)]
    obs = [PauliString("IIZ"), PauliString("ZIZ")]
    expected = np.array([[expectation(c, ob, row) for ob in obs] for row in rows])
    calls.clear()
    actual = expectations_batch(c, obs, rows, max_memory_mb=0.05)
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    assert len(calls) < len(rows)
    assert sum(shape[0] for shape in calls) == len(rows)
    assert max(shape[0] for shape in calls) < len(rows)


def test_density_only_backend_and_invalid_provider_states():
    class Device:
        name = "mixed-test"
        capabilities = BackendCapabilities(density_matrix=True, channels=True, reset=True)

        def __init__(self):
            self.calls = 0

        def run_density(self, circuit, *, options):
            self.calls += 1
            return density_matrix(circuit)

    device = Device()
    c = Circuit(1).h(0)
    obs = [PauliString("X"), PauliString("Y"), PauliString("Z")]
    np.testing.assert_allclose(expectations_batch([c, c], obs, backend=device), [[1, 0, 0]] * 2)
    assert device.calls == 2
    assert execute(c, backend=device).density_matrix is not None
    assert len(execute_batch([c, c], backend=device)) == 2
    assert sum(sample(c, 31, backend=device).values()) == 31
    assert sample(Circuit(1).x(0).bit_flip(0.0, 0).measure(0, 2), 31, backend=device) == {"001": 31}
    with pytest.raises(ValidationError, match="mixed states"):
        statevector(c, backend=device)
    for rho in (np.diag([1.2, -0.2]), [[1, 1], [0, 0]], np.eye(2), [[np.nan, 0], [0, 1]]):
        device.run_density = lambda *args, rho=rho, **kwargs: rho
        with pytest.raises(BackendError):
            density_matrix(c, backend=device)


def test_ir_version3_read_and_effect_payload_guards():
    c = Circuit(1).ry(Parameter("x").sin(), 0).unitary(np.eye(2), 0)
    payload = c.to_dict()
    payload["version"] = 3
    for op in payload["operations"]:
        for field in ("kraus", "condition", "blocks"):
            op.pop(field)
    assert Circuit.from_dict(payload) == c
    payload = Circuit(1).reset(0).to_dict()
    payload["version"] = 3
    for field in ("kraus", "condition", "blocks"):
        payload["operations"][0].pop(field)
    with pytest.raises(ValidationError, match="version 4"):
        Circuit.from_dict(payload)
    payload = feedback().to_dict()
    payload["operations"][2]["condition"]["bits"] = {}
    with pytest.raises(ValidationError):
        Circuit.from_dict(payload)
    with pytest.raises(ValidationError):
        replace(feedback(), num_clbits=0)


def test_statistics_include_branch_bodies_and_sparse_classical_dependencies():
    stats = circuit_stats(feedback().amplitude_damping(0.2, 1).reset(0))
    assert (stats.channels, stats.resets, stats.conditionals, stats.measurements) == (1, 1, 1, 3)
    assert stats.depth >= 6
    c = Circuit(1_000_000_000).if_else(
        ClassicalCondition((999_999_999,), 0),
        Circuit(1_000_000_000).x(999_999_999),
    )
    assert circuit_stats(c).depth == 2


def test_density_provider_options_are_forwarded_and_tensor_context_is_not_ignored():
    class Device:
        name = "density-options"
        capabilities = BackendCapabilities(density_matrix=True)

        def run_density(self, circuit, *, options):
            assert options.max_branches == 7
            return density_matrix(circuit)

    device = Device()
    density_matrix(Circuit(1), backend=device, max_branches=7)
    for options in ({"like": np.ones(1)}, {"batch_shape": (3,)}, {"max_branches": 0}):
        with pytest.raises(ValidationError):
            density_matrix(Circuit(1), backend=device, **options)


def test_builtin_numpy_instance_retains_native_density_batch_options():
    from openqml import get_backend

    rho = density_matrix(Circuit(1), backend=get_backend("numpy"), batch_shape=(3,))
    assert rho.shape == (3, 2, 2)
