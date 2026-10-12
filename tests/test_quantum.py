import itertools

import numpy as np
import pytest

import openqml as q


def reference_operator(matrix, targets, n_qubits, controls=()):
    """Independent bit-index definition, used only for small reference matrices."""
    result = np.zeros((1 << n_qubits, 1 << n_qubits), complex)
    for column in range(1 << n_qubits):
        if any(not (column >> control) & 1 for control in controls):
            result[column, column] = 1
            continue
        local = sum(((column >> wire) & 1) << i for i, wire in enumerate(targets))
        base = column
        for wire in targets:
            base &= ~(1 << wire)
        for row in range(1 << len(targets)):
            global_row = base | sum(((row >> i) & 1) << wire for i, wire in enumerate(targets))
            result[global_row, column] = matrix[row, local]
    return result


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_random_unitaries_and_wire_order(backend):
    rng = np.random.default_rng(13)
    vector = rng.normal(size=16) + 1j * rng.normal(size=16)
    vector /= np.linalg.norm(vector)
    initial = q.State.from_vector(vector)
    if backend == "sparse":
        initial = q.State(4, backend="sparse")
        initial._sparse = {i: a for i, a in enumerate(vector)}
    reference = vector.copy()
    circuit = q.Circuit(4)
    for targets, controls in [([2, 0], []), ([3], [0, 2]), ([1, 3], [2]), ([0, 1, 2, 3], [])]:
        size = 1 << len(targets)
        unitary, _ = np.linalg.qr(
            rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        )
        circuit.unitary(unitary, targets, controls=controls)
        reference = reference_operator(unitary, targets, 4, controls) @ reference
    state = circuit.run(initial_state=initial)
    np.testing.assert_allclose(state.statevector(), reference, atol=1e-12)
    assert np.isclose(np.sum(state.probabilities()), 1)
    np.testing.assert_allclose(
        circuit.inverse().run(initial_state=state).statevector(), vector, atol=1e-12
    )


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_all_pauli_expectations(backend):
    circuit = q.Circuit(3).h(0).ry(0.71, 1).rx(-0.13, 2).cx(0, 2).cz(1, 2)
    state = circuit.run(backend=backend)
    vector = state.statevector()
    for letters in itertools.product("IXYZ", repeat=3):
        label = "".join(letters)
        operator = q.Hamiltonian({label: 1}).matrix()
        np.testing.assert_allclose(
            state.expectation(label), np.vdot(vector, operator @ vector).real, atol=1e-12
        )


@pytest.mark.parametrize("label", ["III", "XIZ", "YXY", "ZZZ", "IXY"])
def test_pauli_rotations(label):
    theta = 0.67
    initial = q.Circuit(3).h(0).ry(0.7, 1).rx(1.1, 2).run()
    operator = q.Hamiltonian({label: 1}).matrix()
    expected = (
        np.cos(theta / 2) * np.eye(8) - 1j * np.sin(theta / 2) * operator
    ) @ initial.statevector()
    actual = q.Circuit(3).pauli_rotation(theta, label).run(initial_state=initial)
    np.testing.assert_allclose(actual.statevector(), expected, atol=1e-12)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
def test_qft_matches_positive_dft_and_inverse(n_qubits):
    size = 1 << n_qubits
    indices = np.arange(size)
    dft = np.exp(2j * np.pi * indices[:, None] * indices[None, :] / size) / np.sqrt(size)
    for index in range(size):
        initial = q.State.from_vector(np.eye(size, dtype=complex)[:, index])
        output = q.qft(n_qubits).run(initial_state=initial)
        np.testing.assert_allclose(output.statevector(), dft[:, index], atol=1e-12)
        np.testing.assert_allclose(
            q.qft(n_qubits, inverse=True).run(initial_state=output).statevector(),
            initial.statevector(),
            atol=1e-12,
        )


def test_qft_on_reordered_subset():
    vector = q.Circuit(3).h(0).ry(0.6, 1).rx(0.3, 2).run().statevector()
    local = np.exp(2j * np.pi * np.arange(4)[:, None] * np.arange(4)[None, :] / 4) / 2
    actual = q.Circuit(3).qft([2, 0]).run(initial_state=q.State.from_vector(vector))
    np.testing.assert_allclose(
        actual.statevector(), reference_operator(local, [2, 0], 3) @ vector, atol=1e-12
    )


@pytest.mark.parametrize("backend", ["dense", "sparse"])
@pytest.mark.parametrize("selected", [[0], [2], [0, 2], [2, 0], [1, 0, 2]])
def test_partial_trace_matches_definition(backend, selected):
    state = q.Circuit(3).h(0).ry(0.7, 1).rx(0.3, 2).cx(0, 2).run(backend=backend)
    rho = np.outer(state.statevector(), state.statevector().conj())
    reference = np.zeros((1 << len(selected), 1 << len(selected)), complex)
    rest = [wire for wire in range(3) if wire not in selected]
    for i, j in itertools.product(range(8), repeat=2):
        if all((i >> wire) & 1 == (j >> wire) & 1 for wire in rest):
            a = sum(((i >> wire) & 1) << k for k, wire in enumerate(selected))
            b = sum(((j >> wire) & 1) << k for k, wire in enumerate(selected))
            reference[a, b] += rho[i, j]
    np.testing.assert_allclose(state.reduced_density(selected), reference, atol=1e-12)
    np.testing.assert_allclose(state.probabilities(selected), reference.diagonal().real)


@pytest.mark.parametrize("p", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("channel", ["amplitude_damping", "dephasing", "depolarizing"])
def test_noise_is_physical_and_matches_kraus(p, channel):
    circuit = q.Circuit(2).h(0).ry(0.3, 1).cx(0, 1)
    pure = circuit.run().statevector()
    rho = np.outer(pure, pure.conj())
    getattr(circuit, channel)(p, 0)
    operation = circuit.operations[-1]
    reference = sum(
        reference_operator(k, [0], 2) @ rho @ reference_operator(k, [0], 2).conj().T
        for k in operation.kraus
    )
    state = circuit.run()
    np.testing.assert_allclose(state.reduced_density(), reference, atol=1e-12)
    assert np.isclose(np.trace(reference), 1)
    assert np.linalg.eigvalsh(reference).min() >= -1e-12
    for label in ("IX", "IY", "IZ", "XX", "YZ"):
        expected = np.trace(reference @ q.Hamiltonian({label: 1}).matrix()).real
        np.testing.assert_allclose(state.expectation(label), expected, atol=1e-12)
    rotated = q.Circuit(2).rx(0.73, 0).cx(1, 0).run(initial_state=state)
    unitary = reference_operator(q.Circuit(1).rx(0.73, 0).operations[0].matrix, [0], 2)
    unitary = reference_operator(np.array([[0, 1], [1, 0]]), [0], 2, [1]) @ unitary
    np.testing.assert_allclose(
        rotated.reduced_density(), unitary @ reference @ unitary.conj().T, atol=1e-12
    )


def test_reset_entangled_qubit_is_an_exact_channel():
    state = q.bell().reset(0).run()
    np.testing.assert_allclose(state.reduced_density(), np.diag([0.5, 0, 0.5, 0]))
    assert np.isclose(state.entropy([1]), 1)
    np.testing.assert_allclose(state.bloch_vector(0), [0, 0, 1])
    with pytest.raises(ValueError, match="mixed"):
        state.statevector()


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_projective_measurement_keeps_measured_eigenstate(basis):
    state = q.Circuit(1).ry(0.6, 0).rz(0.9, 0).run(seed=2)
    bit = state.measure(0, basis)
    np.testing.assert_allclose(state.expectation(basis), 1 - 2 * bit, atol=1e-12)
    assert state.measure(0, basis) == bit


@pytest.mark.parametrize("seed", range(8))
def test_teleportation_with_feedback(seed):
    circuit = q.Circuit(3).ry(0.713, 0).rz(0.441, 0).h(1).cx(1, 2)
    circuit.cx(0, 1).h(0).measure(0, "phase").measure(1, "flip")
    circuit.x(2, condition=("flip", 1)).z(2, condition=("phase", 1))
    state = circuit.run(seed=seed)
    expected = q.Circuit(1).ry(0.713, 0).rz(0.441, 0).run().reduced_density()
    np.testing.assert_allclose(state.reduced_density([2]), expected, atol=1e-12)


def test_sampling_does_not_collapse_and_seed_is_local():
    state = q.bell().run()
    before = state.statevector()
    assert state.sample(100, seed=13) == state.sample(100, seed=13)
    assert set(state.sample(100, seed=5)) == {"00", "11"}
    np.testing.assert_array_equal(before, state.statevector())


def test_sparse_40_qubit_readouts_and_guards():
    state = q.ghz(40).run()
    assert state.backend == "sparse"
    assert len(state.sparse_amplitudes()) == 2
    assert sum(state.sample(257, seed=1).values()) == 257
    np.testing.assert_allclose(state.probabilities([0, 39]), [0.5, 0, 0, 0.5])
    np.testing.assert_allclose(state.reduced_density([0, 39]), np.diag([0.5, 0, 0, 0.5]))
    assert np.isclose(state.expectation("X" * 40), 1)
    with pytest.raises(q.ResourceLimitError):
        state.probabilities()
    with pytest.raises(q.ResourceLimitError):
        state.statevector()
    with pytest.raises(q.ResourceLimitError):
        state.depolarizing(0.1, 0)
    budget = q.SimulationBudget(max_sparse_states=2)
    with pytest.raises(q.ResourceLimitError, match="Sparse"):
        q.Circuit(3).h(0).h(1).run(backend="sparse", budget=budget)
    with pytest.raises(q.ResourceLimitError):
        q.Circuit(18).run(budget=q.SimulationBudget(max_memory_mb=1))
    with pytest.raises(q.ResourceLimitError):
        q.Circuit(1).h(0).repeat(3).run(budget=q.SimulationBudget(max_operations=2))
    with pytest.raises(q.ResourceLimitError, match="width"):
        q.Circuit(64).run()


def test_resource_depth_respects_measurement_dependencies():
    circuit = q.Circuit(3).h(0).cx(0, 1).measure(0, "m").x(2, condition=("m", 1))
    assert circuit.resources()["depth"] == 4


def test_composition_clone_and_operation_readout_are_independent():
    block = q.bell()
    circuit = q.Circuit(3).compose(block, [2, 0])
    block.x(1)
    assert len(circuit) == 2
    np.testing.assert_allclose(circuit.run().probabilities(), [0.5, 0, 0, 0, 0, 0.5, 0, 0])
    op = circuit.operations[0]
    op.matrix[:] = 0
    assert np.isclose(circuit.run().probabilities().sum(), 1)
    state = circuit.run()
    copy = state.clone().x(1)
    assert not np.allclose(state.statevector(), copy.statevector())
    assert len(q.Circuit(1).h(0).compose(q.Circuit(1).x(0))) == 2
    block = q.Circuit(1).h(0)
    block.compose(block)
    assert len(block) == 2


def test_invalid_inputs_and_nonunitary_inverse():
    for count in (0, -1, 1.5, True, 65):
        with pytest.raises(ValueError):
            q.Circuit(count)
    for build in (
        lambda: q.Circuit(2).cx(0, 0),
        lambda: q.Circuit(1).rx(np.nan, 0),
        lambda: q.Circuit(1).x(1),
        lambda: q.Circuit(1).unitary([[1, 1], [0, 1]], [0]),
        lambda: q.Circuit(1).depolarizing(1.1, 0),
        lambda: q.Circuit(1).channel([np.zeros((2, 2))], [0]),
        lambda: q.Circuit(1).x(0, condition=("missing", 1)),
    ):
        with pytest.raises(ValueError):
            build()
    with pytest.raises(q.UnsupportedOperationError):
        q.Circuit(1).measure(0).inverse()
    with pytest.raises(ValueError):
        q.State.from_vector([1, 1])
    with pytest.raises(ValueError):
        q.Circuit(1).run(budget="bad")


def test_grover_and_qaoa_against_reference():
    assert q.grover(3, [5]).run().probabilities()[5] == pytest.approx(0.9453125)
    gamma, beta = 0.61, -0.3
    circuit = q.qaoa_maxcut(2, [(0, 1)], [gamma], [beta])
    cost = (np.eye(4) - q.Hamiltonian({"ZZ": 1}).matrix()) / 2
    cost_unitary = np.diag(np.exp(-1j * gamma * np.diag(cost)))
    rx = np.cos(beta) * np.eye(2) - 1j * np.sin(beta) * np.array([[0, 1], [1, 0]])
    expected = np.kron(rx, rx) @ cost_unitary @ np.ones(4) / 2
    np.testing.assert_allclose(circuit.run().statevector(), expected, atol=1e-12)


def test_vqe_reaches_certified_ground_energy():
    hamiltonian = q.Hamiltonian({"ZZ": -1, "XI": -0.5, "IX": -0.5})

    def ansatz(p):
        return q.Circuit(2).ry(p[0], 0).ry(p[1], 1).cx(0, 1)

    fit = q.vqe(hamiltonian, ansatz, [0.4, 0.3], steps=100, learning_rate=0.08)
    assert hamiltonian.ground_energy() == pytest.approx(-np.sqrt(2))
    assert fit.value - hamiltonian.ground_energy() < 1e-4
    assert fit.value == pytest.approx(ansatz(fit.parameters).run().expectation(hamiltonian))
