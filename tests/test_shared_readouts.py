"""Independent matrix references and native transform checks for shared readouts."""

from itertools import product

import numpy as np
import pytest

import openqml as oq
from openqml.density import _expectations_from_density
from openqml.simulator import _expectations_from_state, _TensorOps


def dense(word):
    matrices = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    result = np.ones((1, 1))
    for character in word:
        result = np.kron(result, matrices[character])
    return result


def readouts(width):
    words = [oq.PauliString("".join(p)) for p in product("IZ", repeat=width)]
    words += [oq.PauliString("X" * width, -0.4), oq.PauliString("Y" * width, 0.7)]
    return (
        *reversed(words),
        oq.Hamiltonian((words[1], words[1], oq.PauliString(words[-1].paulis, -0.7))),
        oq.PauliString("Z" * width, 0),
        words[1],
    )


def reference(state, observables, *, density=False):
    values = []
    for observable in observables:
        matrix = sum(
            (term.coefficient * dense(term.paulis))
            for term in (
                observable.terms if isinstance(observable, oq.Hamiltonian) else (observable,)
            )
        )
        values.append(
            np.trace(state @ matrix, axis1=-2, axis2=-1).real
            if density
            else np.einsum("...i,ij,...j->...", state.conj(), matrix, state).real
        )
    return np.stack(values, axis=-1)


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("shape", [(), (2, 3), (0,)])
@pytest.mark.parametrize("density", [False, True])
def test_mixed_readouts_match_independent_full_matrices(width, shape, density):
    rng = np.random.default_rng(27)
    state = rng.normal(size=shape + (2**width,)) + 1j * rng.normal(size=shape + (2**width,))
    # Unnormalized inputs exercise identity normalization as well as wire order.
    if density:
        state = state[..., :, None] * state.conj()[..., None, :]
    before, obs = state.copy(), readouts(width)
    evaluate = _expectations_from_density if density else _expectations_from_state
    np.testing.assert_allclose(
        evaluate(state, obs, width, _TensorOps()),
        reference(state, obs, density=density),
        atol=1e-12,
    )
    np.testing.assert_array_equal(state, before)


@pytest.mark.parametrize("density", [False, True])
def test_duplicate_terms_execute_once_and_zero_terms_are_still_validated(monkeypatch, density):
    import openqml.density as dm
    import openqml.simulator as sim

    module, name = (dm, "_density_pauli_expectation") if density else (sim, "_pauli_expectation")
    original, calls = getattr(module, name), []

    def counted(state, word, *args):
        calls.append(word)
        return original(state, word, *args)

    monkeypatch.setattr(module, name, counted)
    state = np.array([1, 1j, 0.3, -0.4j])
    if density:
        state = np.outer(state, state.conj())
    obs = (
        oq.PauliString("XY", -2),
        oq.Hamiltonian((oq.PauliString("XY", 3), oq.PauliString("ZZ"))),
        oq.PauliString("XY"),
    )
    evaluate = _expectations_from_density if density else _expectations_from_state
    np.testing.assert_allclose(
        evaluate(state, obs, 2, _TensorOps()), reference(state, obs, density=density)
    )
    assert calls == ["XY"]
    with pytest.raises(oq.ValidationError):
        evaluate(state, (oq.PauliString("X", 0),), 2, _TensorOps())


@pytest.mark.parametrize("strided", [False, True])
def test_wide_numpy_paulis_use_tiles_without_retaining_full_state_plans(monkeypatch, strided):
    import openqml.simulator as sim

    width, rng = 13, np.random.default_rng(41)
    size = (2 if strided else 1) * 2**width
    state = rng.normal(size=(2, 3, size)) + 1j * rng.normal(size=(2, 3, size))
    if strided:
        state = state[..., ::2]
    word = "YZXIYZXIYZXIY"
    # Independent tensor action avoids a prohibitively large dense matrix.
    transformed = state.reshape((2, 3) + (2,) * width)
    for q, character in enumerate(word):
        local = dense(character)
        transformed = np.moveaxis(transformed, 2 + q, -1) @ local.T
        transformed = np.moveaxis(transformed, -1, 2 + q)
    expected = np.sum(state.conj() * transformed.reshape(state.shape), axis=-1).real
    conjugate, sizes = np.conj, []

    def checked(block):
        sizes.append(block.shape[-1])
        return conjugate(block)

    sim._cached_pauli_plan.cache_clear()
    monkeypatch.setattr(sim.np, "conj", checked)
    np.testing.assert_allclose(sim._numpy_pauli_expectation(state, word), expected, atol=2e-12)
    assert len(sizes) > 1 and max(sizes) * (64 * 6 + 48) <= 1 << 20
    sim._pauli_plan(word)
    assert sim._cached_pauli_plan.cache_info().currsize == 0
    sim._pauli_plan("XY")
    assert sim._cached_pauli_plan.cache_info().currsize == 1


@pytest.mark.parametrize("many", [False, True])
def test_torch_full_graph_readout_capture(many):
    torch = pytest.importorskip("torch")
    obs = readouts(4) if many else (oq.PauliString("IZIZ"), oq.PauliString("XIYI"))
    state = torch.randn(2, 16, dtype=torch.complex128, requires_grad=True)

    def evaluate(value):
        return _expectations_from_state(value, obs, 4, _TensorOps("torch", value.real))

    captured = torch.compile(evaluate, backend="eager", fullgraph=True)
    torch.testing.assert_close(captured(state), evaluate(state))
    expected = torch.autograd.grad(evaluate(state).square().sum(), state)[0]
    actual = torch.autograd.grad(captured(state).square().sum(), state)[0]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("framework", ["torch", "jax"])
@pytest.mark.parametrize("precision", [32, 64])
def test_wide_native_readouts_do_not_fill_global_constant_caches(framework, precision):
    lib = pytest.importorskip(framework)
    import openqml.simulator as sim

    rng = np.random.default_rng(9)
    state = rng.normal(size=(2, 2**13)) + 1j * rng.normal(size=(2, 2**13))
    word = "X" * 13
    expected = (state.conj() * state[..., ::-1]).sum(axis=-1).real
    sim._cached_pauli_plan.cache_clear()
    sim._torch_cached_pauli_plan.cache_clear()
    if framework == "torch":
        value = lib.tensor(state, dtype=lib.complex128 if precision == 64 else lib.complex64)
        actual = sim._expectations_from_state(
            value, (oq.PauliString(word),), 13, sim._TensorOps("torch", value.real)
        ).numpy()[..., 0]
    else:
        import jax.numpy as jnp

        with oq.jax.enable_x64(precision == 64):
            value = jnp.array(state, dtype=jnp.complex128 if precision == 64 else jnp.complex64)
            actual = np.asarray(
                lib.jit(
                    lambda value: sim._expectations_from_state(
                        value, (oq.PauliString(word),), 13, sim._TensorOps("jax", value.real)
                    )
                )(value)
            )[..., 0]
    np.testing.assert_allclose(actual, expected, atol=2e-4 if precision == 32 else 2e-12)
    assert sim._cached_pauli_plan.cache_info().currsize == 0
    assert sim._torch_cached_pauli_plan.cache_info().currsize == 0


@pytest.mark.parametrize(
    "backend,dependency", [("cirq-density", "cirq"), ("aer-density", "qiskit_aer")]
)
@pytest.mark.parametrize("effects", [False, True])
def test_sdk_density_readouts_share_exact_ordered_results(backend, dependency, effects):
    pytest.importorskip(dependency)
    circuit = oq.Circuit(4).h(0).ry(0.4, 1).cx(0, 3)
    if effects:
        circuit = circuit.bit_flip(0.13, 2).measure(0, 0)
    obs = readouts(4)
    expected = reference(oq.density_matrix(circuit), obs, density=True)
    np.testing.assert_allclose(oq.expectations(circuit, obs, backend=backend), expected, atol=1e-12)
    np.testing.assert_allclose(
        oq.expectations_batch(circuit, obs, [{}, {}], backend=backend),
        np.stack([expected, expected]),
        atol=1e-12,
    )


@pytest.mark.parametrize("framework", ["torch", "jax"])
@pytest.mark.parametrize("density", [False, True])
def test_walsh_readouts_keep_native_first_and_second_derivatives(framework, density):
    lib = pytest.importorskip(framework)
    width = 4
    masks = tuple(range(2**width))
    obs = tuple(
        oq.PauliString("".join("Z" if mask & (1 << (width - q - 1)) else "I" for q in range(width)))
        for mask in masks
    )
    x = oq.Parameter("x")
    builder = oq.CircuitBuilder(width)
    for q in range(width):
        builder.ry((q + 1) * x, q)
    if density:
        builder.bit_flip(0.1, 0)
    circuit = builder.build()
    weights = np.linspace(-0.3, 0.5, len(obs))
    angle, step = 0.27, 1e-4

    def independent(value):
        return sum(
            weight
            * np.prod(
                [
                    np.cos((q + 1) * value) * (0.8 if density and q == 0 else 1)
                    for q in range(width)
                    if mask & (1 << (width - q - 1))
                ]
            )
            for mask, weight in zip(masks, weights, strict=True)
        )

    expected_first = (independent(angle + step) - independent(angle - step)) / (2 * step)
    expected_second = (
        independent(angle + step) - 2 * independent(angle) + independent(angle - step)
    ) / step**2
    if framework == "torch":
        value = lib.tensor(angle, dtype=lib.float64, requires_grad=True)

        def function(value):
            return (
                oq.expectations(circuit, obs, {"x": value}, backend="torch", like=value)
                * lib.tensor(weights)
            ).sum()

        first = lib.autograd.grad(function(value), value, create_graph=True)[0]
        second = lib.autograd.grad(first, value)[0]
        np.testing.assert_allclose(first.detach(), expected_first, atol=2e-6)
        np.testing.assert_allclose(second.detach(), expected_second, atol=2e-6)
        lib.testing.assert_close(lib.func.hessian(function)(value), second)
        lib.testing.assert_close(
            lib.func.vmap(function)(lib.tensor([angle, angle + 0.1], dtype=lib.float64)),
            lib.tensor([independent(angle), independent(angle + 0.1)], dtype=lib.float64),
        )
    else:
        import jax.numpy as jnp

        with oq.jax.enable_x64():

            def function(value):
                return (
                    oq.expectations(circuit, obs, {"x": value}, backend="jax", like=value)
                    * jnp.asarray(weights)
                ).sum()

            np.testing.assert_allclose(
                lib.jit(lib.grad(function))(jnp.array(angle)), expected_first, atol=2e-6
            )
            np.testing.assert_allclose(
                lib.jit(lib.hessian(function))(jnp.array(angle)), expected_second, atol=2e-6
            )
            np.testing.assert_allclose(
                lib.jit(lib.vmap(function))(jnp.array([angle, angle + 0.1])),
                [independent(angle), independent(angle + 0.1)],
                atol=1e-12,
            )


@pytest.mark.parametrize("target", ["qiskit", "cirq", "pennylane", "openqasm2", "openqasm3"])
def test_imported_circuits_share_native_readout_semantics(target):
    if target in {"qiskit", "cirq", "pennylane"}:
        pytest.importorskip(target)
    circuit = oq.Circuit(4).h(0).ry(0.43, 2).cx(0, 3).rzz(0.71, 1, 2)
    program = oq.convert(circuit, target=target).value
    restored = oq.convert(program, source=target, target="ir").value
    obs = readouts(4)
    expected = reference(oq.statevector(circuit), obs)
    np.testing.assert_allclose(oq.expectations(restored, obs), expected, atol=1e-12)
    np.testing.assert_allclose(
        oq.expectations_batch([restored, restored], obs), np.stack([expected, expected]), atol=1e-12
    )
