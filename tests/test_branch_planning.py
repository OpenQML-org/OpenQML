"""Liveness must retain quantum correlations, full final records and derivatives."""

from dataclasses import asdict

import numpy as np
import pytest

from openqml import (
    Circuit,
    ClassicalCondition,
    Parameter,
    PauliString,
    ResourceError,
    ValidationError,
    density_matrix,
    expectations_batch,
    parameter_shift,
    plan_density,
    sample,
    statevector,
)
from openqml.density import _simulate, channel_kraus
from openqml.ir import CHANNEL_GATES, UNITARY_GATES


def history_circuit(length, *, reuse=False):
    c = Circuit(2)
    for i in range(length):
        bit = 0 if reuse else i
        c = (
            c.h(0)
            .measure(0, bit)
            .if_else(
                ClassicalCondition((bit,), 1),
                Circuit(2).x(1),
            )
            .reset(0)
        )
    return c


def _full_matrix(local, qubits, n):
    matrix = np.zeros((2**n, 2**n), complex)
    for source in range(2**n):
        bits = [int(b) for b in format(source, f"0{n}b")]
        index = sum(bits[q] << (len(qubits) - i - 1) for i, q in enumerate(qubits))
        for target in range(2 ** len(qubits)):
            output = bits.copy()
            for i, q in enumerate(qubits):
                output[q] = (target >> (len(qubits) - i - 1)) & 1
            destination = sum(b << (n - i - 1) for i, b in enumerate(output))
            matrix[destination, source] = local[target, index]
    return matrix


def reference_instruments(circuit):
    """Independent full-operator reference retaining every final classical bit."""
    n = circuit.num_qubits

    def evolve(body, states):
        for op in body.operations:
            output = {}
            if op.name in UNITARY_GATES:
                columns = []
                for source in range(2**n):
                    prep = Circuit(n)
                    for q, bit in enumerate(format(source, f"0{n}b")):
                        if bit == "1":
                            prep = prep.x(q)
                    columns.append(statevector(prep.append(op)))
                operators = [np.stack(columns, axis=1)]
            elif op.name in CHANNEL_GATES | {"reset"}:
                operators = [_full_matrix(k, op.qubits, n) for k in channel_kraus(op)]
            for key, rho in states.items():
                if op.name == "if_else":
                    index = sum((bit in key) << i for i, bit in enumerate(op.condition.bits))
                    branch = evolve(op.blocks[0 if index == op.condition.value else 1], {key: rho})
                    for k, value in branch.items():
                        output[k] = output.get(k, 0) + value
                elif op.name == "measure":
                    for value in (0, 1):
                        projection = _full_matrix(np.diag([1 - value, value]), op.qubits, n)
                        target = key | {op.clbits[0]} if value else key - {op.clbits[0]}
                        output[target] = output.get(target, 0) + projection @ rho @ projection
                elif op.name == "barrier":
                    output[key] = rho
                else:
                    output[key] = sum(k @ rho @ k.conj().T for k in operators)
            states = output
        return states

    initial = np.zeros((2**n, 2**n), complex)
    initial[0, 0] = 1
    return evolve(circuit, {frozenset(): initial})


def test_hundreds_of_feedback_records_keep_two_live_branches():
    c = history_circuit(128)
    plan = plan_density(c, max_branches=2)
    assert (plan.peak_branches, plan.output_branches, plan.max_live_bits) == (2, 1, 1)
    assert plan.recorded_measurements == 128 and plan.unrecorded_measurements == 0
    assert plan.estimated_memory_mb() == pytest.approx(2**-9)
    np.testing.assert_allclose(
        density_matrix(c, max_branches=2, max_memory_mb=0.01), np.diag([0.5, 0.5, 0, 0]), atol=1e-13
    )
    assert asdict(plan)["retains_classical"] is False
    with pytest.raises(ResourceError, match="branch"):
        plan_density(c, retain_classical=True)


def test_unread_measurements_become_dephasing_without_any_record_branches():
    c = Circuit(1)
    for i in range(96):
        c = c.h(0).measure(0, i)
    plan = plan_density(c, max_branches=1)
    assert plan.peak_branches == plan.output_branches == 1
    assert plan.unrecorded_measurements == 96 and plan.recorded_measurements == 0
    np.testing.assert_allclose(density_matrix(c, max_branches=1), np.eye(2) / 2, atol=1e-13)


def test_full_counts_preserve_last_write_while_dropping_overwritten_history():
    c = history_circuit(80, reuse=True).measure(1, 1)
    plan = plan_density(c, retain_classical=True, max_branches=4)
    assert plan.peak_branches <= 4 and plan.output_branches == 4
    counts = sample(c, 10_000, seed=9)
    assert set(counts) == {"00", "01", "10", "11"}
    for count in counts.values():
        assert count / 10_000 == pytest.approx(0.25, abs=0.025)


@pytest.mark.parametrize("seed", range(8))
def test_random_nested_feedback_matches_full_instrument_reference(seed):
    rng = np.random.default_rng(seed)
    c = Circuit(3).h(0).h(1).cx(0, 2)
    for _ in range(4):
        q, bit = int(rng.integers(3)), int(rng.integers(3))
        other = (q + 1) % 3
        c = c.ry(float(rng.normal()), q).amplitude_damping(0.2, other).measure(q, bit)
        true = (
            Circuit(3)
            .rx(float(rng.normal()), other)
            .measure(q, bit)
            .if_else(
                ClassicalCondition((bit,), 1),
                Circuit(3).x(other),
                Circuit(3).h(q),
            )
        )
        false = Circuit(3).reset(other).phase_flip(0.13, q)
        c = c.if_else(ClassicalCondition((bit,), int(rng.integers(2))), true, false)
    expected = reference_instruments(c)
    actual, _, _ = _simulate(c, retain_classical=True)
    # Zero-probability branches may be structurally present in either reference.
    zero = np.zeros((8, 8), complex)
    for key in set(expected) | set(actual):
        np.testing.assert_allclose(
            actual.get(key, zero).reshape(8, 8), expected.get(key, zero), atol=2e-13
        )
    np.testing.assert_allclose(density_matrix(c), sum(expected.values()), atol=2e-13)


def test_condition_bits_are_captured_once_even_when_child_discards_them():
    c = (
        Circuit(2)
        .h(0)
        .measure(0, 0)
        .if_else(
            ClassicalCondition((0,), 1),
            Circuit(2)
            .reset(0)
            .measure(0, 0)
            .x(1)
            .if_else(
                ClassicalCondition((0,), 0),
                Circuit(2).h(1),
            ),
            Circuit(2).x(0),
        )
    )
    np.testing.assert_allclose(
        density_matrix(c), sum(reference_instruments(c).values()), atol=1e-14
    )
    plan = plan_density(c)
    assert plan.output_branches == 1


def test_batch_planning_does_not_reintroduce_historical_branch_cap():
    x = Parameter("x")
    c = history_circuit(16).ry(x, 1)
    rows = [{"x": 0.1 * i} for i in range(8)]
    output = expectations_batch(c, [PauliString("ZI"), PauliString("IZ")], rows, max_memory_mb=0.02)
    np.testing.assert_allclose(output, [[1, 0]] * 8, atol=1e-13)


def test_sparse_register_plans_and_huge_quantum_estimates_do_not_allocate():
    c = Circuit(1).h(0).measure(0, 999_999_999)
    plan = plan_density(c)
    assert plan.peak_branches == 1 and plan.max_live_bits == 0
    plan = plan_density(c, retain_classical=True)
    assert plan.peak_branches == 2 and plan.max_live_bits == 1
    assert np.isinf(plan_density(Circuit(1_000_000_000)).estimated_memory_mb())
    assert plan_density(Circuit(1)).estimated_memory_mb(batch_size=0) > 0
    for kwargs in ({"retain_classical": 1}, {"max_branches": 0}):
        with pytest.raises(ValidationError):
            plan_density(Circuit(1), **kwargs)
    with pytest.raises(ValidationError):
        plan.estimated_memory_mb(batch_size=-1)


def test_count_branch_limits_remain_checked_before_state_allocation(monkeypatch):
    c = history_circuit(12)

    def forbidden(*args, **kwargs):
        raise AssertionError("state allocated before planning")

    monkeypatch.setattr(np, "zeros", forbidden)
    with pytest.raises(ResourceError, match="branch"):
        sample(c, 10)


def test_zero_weight_branches_keep_noise_probability_derivatives_torch():
    torch = pytest.importorskip("torch")
    from openqml import expectation

    c = (
        Circuit(2)
        .bit_flip(Parameter("p"), 0)
        .measure(0, 0)
        .if_else(
            ClassicalCondition((0,), 1),
            Circuit(2).x(1),
        )
    )
    p = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    value = expectation(c, PauliString("IZ"), {"p": p}, backend="torch")
    (derivative,) = torch.autograd.grad(value, (p,))
    assert derivative.item() == pytest.approx(-2.0)
    assert parameter_shift(c, PauliString("IZ"), {"p": 0})["p"] == pytest.approx(-2.0)


def test_zero_weight_measurement_branches_keep_second_derivative_jax():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from openqml import expectation

    c = (
        Circuit(2)
        .ry(Parameter("x"), 0)
        .measure(0, 0)
        .if_else(
            ClassicalCondition((0,), 1),
            Circuit(2).x(1),
        )
    )

    def function(x):
        return expectation(c, PauliString("IZ"), {"x": x}, backend="jax")

    assert jax.jit(jax.grad(jax.grad(function)))(jnp.array(0.0)).item() == pytest.approx(-1.0)


def deferred_condition():
    c = Circuit(2)
    for bit in range(5):
        c = c.h(0).measure(0, bit).reset(0)
    return (
        c.rxx(Parameter("x"), 0, 1)
        .amplitude_damping(Parameter("p"), 1)
        .if_else(
            ClassicalCondition(tuple(range(5)), 31),
            Circuit(2).x(1),
        )
    )


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_branch_axis_vectorization_broadcasts_independently_of_input_axes(backend, monkeypatch):
    import openqml.simulator as simulator
    from openqml import expectations

    original, calls = simulator._gate_matrix, []

    def counted(name, *args, **kwargs):
        calls.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(simulator, "_gate_matrix", counted)
    c = deferred_condition()
    x, p = np.array([[0.3], [0.7]]), np.array([[0.1, 0.3, 0.5]])
    if backend == "torch":
        torch = pytest.importorskip("torch")
        x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        p = torch.tensor(p, dtype=torch.float64, requires_grad=True)
    elif backend == "jax":
        jnp = pytest.importorskip("jax.numpy")
        x, p = jnp.asarray(x), jnp.asarray(p)
    actual = expectations(
        c, [PauliString("ZI"), PauliString("IZ")], {"x": x, "p": p}, backend=backend
    )
    assert calls.count("rxx") == 1 and calls.count("h") == 5
    xnp = np.array([[0.3], [0.7]])
    pnp = np.array([[0.1, 0.3, 0.5]])
    expected = np.stack(
        [np.broadcast_to(np.cos(xnp), (2, 3)), 30 / 32 * ((1 - pnp) * np.cos(xnp) + pnp)], axis=-1
    )
    if backend == "torch":
        gx, gp = torch.autograd.grad(actual.sum(), (x, p))
        np.testing.assert_allclose(
            gx, (-np.sin(xnp) * (1 + 30 / 32 * (1 - pnp))).sum(axis=1, keepdims=True), atol=1e-13
        )
        np.testing.assert_allclose(
            gp,
            np.broadcast_to((30 / 32 * (1 - np.cos(xnp))).sum(axis=0, keepdims=True), (1, 3)),
            atol=1e-13,
        )
        actual = actual.detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=3e-6 if backend == "jax" else 1e-13)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_public_readout_branch_limits_are_honored(backend):
    from openqml import expectation

    if backend != "numpy":
        pytest.importorskip(backend)
    c = deferred_condition()
    with pytest.raises(ResourceError, match="branch"):
        expectation(c, PauliString("ZI"), {"x": 0.3, "p": 0.2}, backend=backend, max_branches=16)
    assert np.isfinite(
        float(
            expectation(
                c, PauliString("ZI"), {"x": 0.3, "p": 0.2}, backend=backend, max_branches=32
            )
        )
    )
    with pytest.raises(ResourceError):
        expectations_batch(
            c, [PauliString("ZI")], [{"x": 0.3, "p": 0.2}], backend=backend, max_branches=16
        )


@pytest.mark.parametrize("framework", ["torch", "jax", "flax"])
def test_layer_branch_budget_rejects_before_forward(framework):
    pytest.importorskip(framework)
    from importlib import import_module

    layer = import_module(f"openqml.interfaces.{framework}").QuantumLayer
    if framework == "flax":
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        module = layer(deferred_condition(), ("x",), ("p",), max_branches=16)
        with pytest.raises(ResourceError):
            module.init(jax.random.key(1), jnp.ones((1, 1)))
    else:
        with pytest.raises(ResourceError):
            layer(deferred_condition(), ("x",), ("p",), max_branches=16)


def test_full_history_limit_precedes_liveness_snapshot_construction(monkeypatch):
    import openqml.planning as planning

    c = history_circuit(512)

    def forbidden(*args, **kwargs):
        raise AssertionError("liveness snapshots allocated before full-record budget guard")

    monkeypatch.setattr(planning, "_build_records", forbidden)
    with pytest.raises(ResourceError, match="record"):
        plan_density(c, retain_classical=True)


def test_memory_budget_caps_branch_metadata_when_user_sets_a_huge_branch_limit(monkeypatch):
    import openqml.density as density

    c = history_circuit(24)
    original = density._record_plan
    limits = []

    def counted(circuit, retain, bits, max_branches):
        limits.append(max_branches)
        return original(circuit, retain, bits, max_branches)

    monkeypatch.setattr(density, "_record_plan", counted)
    with pytest.raises(ResourceError, match="working-memory"):
        sample(c, 10, max_branches=10**9, max_memory_mb=0.01)
    assert limits == [10]


def test_never_written_condition_bits_require_no_classical_storage_tracking():
    c = Circuit(1)
    for bit in range(96):
        c = c.if_else(ClassicalCondition((bit,), 0), Circuit(1).x(0))
    plan = plan_density(c)
    assert plan.max_live_bits == 0 and plan.peak_branches == 1
    np.testing.assert_allclose(density_matrix(c), [[1, 0], [0, 0]], atol=1e-14)


def test_failed_plans_are_not_retained_in_liveness_cache():
    import openqml.planning as planning

    c = deferred_condition().barrier(0, 1)
    before = planning._cached_records.cache_info()
    for _ in range(2):
        with pytest.raises(ResourceError):
            plan_density(c, classical_bits=(0, 1, 2, 3), max_branches=8)
    after = planning._cached_records.cache_info()
    assert after.hits == before.hits and after.misses == before.misses + 2


def test_deep_conditionals_do_not_pin_one_old_density_per_parent_frame():
    import tracemalloc

    c = Circuit(8).ry(0.01, 0)
    for _ in range(40):
        c = Circuit(8).ry(0.01, 0).if_else(ClassicalCondition((0,), 0), c)
    tracemalloc.start()
    try:
        rho = density_matrix(c, max_memory_mb=4)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    vector = statevector(Circuit(8).ry(0.41, 0))
    np.testing.assert_allclose(rho, np.outer(vector, vector.conj()), atol=1e-13)
    # The old retained-parent behavior exceeded 40 rho buffers. Allow room for
    # contraction workspaces and Python metadata; this is not a hard RSS limit.
    assert peak < 12 * rho.nbytes
