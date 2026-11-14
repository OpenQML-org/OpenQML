"""Continuous dynamics against analytic channels and independent generator exponentials."""

import numpy as np
import pytest

import openqml as oq


def _matrix(word):
    paulis = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1]),
    }
    result = np.ones((1, 1))
    for symbol in word:
        result = np.kron(result, paulis[symbol])
    return result


def _exact(plan, h, rates, duration, state):
    expm = pytest.importorskip("scipy.linalg").expm
    d = 1 << plan.num_qubits
    identity = np.eye(d)
    ham = sum(
        (
            value * term.coefficient * _matrix(term.paulis)
            for value, term in zip(h, plan.hamiltonians, strict=True)
        ),
        np.zeros((d, d)),
    )
    generator = -1j * (np.kron(ham, identity) - np.kron(identity, ham.T))
    jumps = [term.coefficient * _matrix(term.paulis) for term in plan.pauli_jumps]
    for site in plan.damping_qubits:
        local = np.ones((1, 1))
        for q in range(plan.num_qubits):
            local = np.kron(local, [[0, 1], [0, 0]] if q == site else np.eye(2))
        jumps.append(local)
    for rate, operator in zip(rates, jumps, strict=True):
        product = operator.conj().T @ operator
        generator += rate * (
            np.kron(operator, operator.conj())
            - 0.5 * np.kron(product, identity)
            - 0.5 * np.kron(identity, product.T)
        )
    return (expm(duration * generator) @ state.ravel()).reshape(d, d)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_generator_uncertainty_envelopes_bound_final_observable_error(backend):
    if backend == "torch":
        torch = pytest.importorskip("torch")

        def array(x):
            return torch.tensor(x, dtype=torch.float64)
    elif backend == "jax":
        pytest.importorskip("jax")
        import jax.numpy as jnp

        array = jnp.asarray
    else:
        array = np.asarray
    plan = oq.LindbladPlan(1, ("X", "Z"), ("Z",), (0,), observables=("X", "Y", "Z"), substeps=24)
    h = np.array([[0.4, 0.2], [-0.1, 0.3]])
    r, dt = np.array([[0.2, 0.1], [0.1, 0.4]]), [0.3, 0.5]
    dh, dr = np.full((2, 2), 0.002), np.full((2, 2), 0.001)
    nominal = plan.evolve(array(h), array(r), array(dt), backend=backend)
    changed = plan.evolve(array(h + dh), array(r + dr), array(dt), backend=backend)
    certificate = plan.uncertainty_bound(
        array(dh),
        array(dr),
        array(dt),
        numerical_error=nominal.error_bound,
        backend=backend,
    )
    actual = np.abs(np.asarray(nominal.outputs[-1]) - np.asarray(changed.outputs[-1]))
    assert np.all(
        actual <= np.asarray(certificate.observable_error_bounds) + np.asarray(changed.error_bound)
    )
    np.testing.assert_allclose(
        certificate.state_error_bound, np.asarray(nominal.error_bound) + 0.0096, atol=2e-7
    )
    masked = plan.uncertainty_bound(
        array(dh), array(dr), array(dt), backend=backend, mask=[1, 0], initial_error=0.01
    )
    assert float(masked.state_error_bound) == pytest.approx(0.0136, abs=2e-7)
    capped = plan.uncertainty_bound(array(dh * 1000), array(dr * 1000), backend=backend)
    assert float(capped.state_error_bound) == 2
    with pytest.raises(oq.ValidationError):
        plan.uncertainty_bound(array(-dh), array(dr), backend=backend)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_energy_decay_is_exact_and_finite_at_saturation(backend):
    if backend == "numpy":
        xp, dtype = np, np.float64
    else:
        lib = pytest.importorskip(backend)
        xp = lib if backend == "torch" else lib.numpy
        dtype = xp.float64 if backend == "torch" else xp.float32
    plan = oq.LindbladPlan(1, damping_qubits=(0,))
    h, rates = (
        xp.asarray(np.empty((3, 0)), dtype=dtype),
        xp.asarray([[0.4], [2], [1000]], dtype=dtype),
    )
    result = plan.evolve(h, rates, [0.2, 0.3, 2], state=np.diag([0, 1]), backend=backend)
    actual = result.outputs.detach().numpy() if backend == "torch" else np.asarray(result.outputs)
    expected = 1 - 2 * np.exp(-np.cumsum([0.08, 0.6, 2000]))
    np.testing.assert_allclose(actual[:, 0], expected, atol=4e-6)
    assert float(result.error_bound) == 0
    state = result.state.detach().numpy() if backend == "torch" else np.asarray(result.state)
    np.testing.assert_allclose(state, np.diag([1, 0]), atol=3e-6)


def test_symmetric_cptp_evolution_matches_generator_and_its_trace_norm_bound():
    rho = np.outer([1, 0, 0, 1], [1, 0, 0, 1]) / 2
    h, rates = np.array([0.7, -0.3, 0.2]), np.array([0.4, 0.2])
    differences = []
    for steps in (2, 4, 8):
        plan = oq.LindbladPlan(2, ("XI", "ZZ", "IY"), ("ZX",), (1,), substeps=steps)
        result = plan.evolve(h[None], rates[None], 0.4, state=rho)
        expected = _exact(plan, h, rates, 0.4, rho)
        error = np.abs(np.linalg.eigvalsh(result.state - expected)).sum()
        assert error <= float(result.error_bound) + 1e-12
        assert np.linalg.eigvalsh(result.state).min() >= -1e-12
        np.testing.assert_allclose(np.trace(result.state), 1, atol=2e-13)
        differences.append(error)
    assert differences[1] < 0.3 * differences[0]
    assert differences[2] < 0.3 * differences[1]


def test_commuting_pauli_hamiltonian_and_jumps_have_zero_splitting_error():
    plan = oq.LindbladPlan(2, ("ZI", "IZ"), ("ZZ",), substeps=1)
    psi = np.array([1, 1j, -1, 1]) / 2
    rho = np.outer(psi, psi.conj())
    result = plan.evolve(np.array([[0.3, -0.7]]), np.array([[0.2]]), 0.8, state=rho)
    np.testing.assert_allclose(result.state, _exact(plan, [0.3, -0.7], [0.2], 0.8, rho), atol=2e-13)
    assert result.error_bound == 0


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_open_system_controls_keep_first_and_second_derivatives(backend):
    lib = pytest.importorskip(backend)
    xp = lib if backend == "torch" else lib.numpy
    dtype = xp.float64 if backend == "torch" else xp.float32
    plan = oq.LindbladPlan(1, ("X", "Z"), ("Y",), (0,), substeps=2)
    value = xp.asarray([0.3, -0.2, 0.4, 0.1, 0.2], dtype=dtype)

    def function(v):
        h = xp.stack((v[:2], v[:2] * 0.7))
        rates = xp.stack((v[2:4], v[2:4] * 1.2))
        return plan.evolve(h, rates, v[4], backend=backend, checkpoint=True).outputs.sum()

    if backend == "torch":
        value.requires_grad_()
        assert lib.autograd.gradcheck(function, (value,))
        assert lib.autograd.gradgradcheck(function, (value,))
    else:
        first = lib.jit(lib.grad(function))(value)
        second = lib.jit(lib.jacfwd(lib.grad(function)))(value)
        assert np.isfinite(first).all() and np.isfinite(second).all()
        np.testing.assert_allclose(lib.jit(function)(value), function(value), atol=2e-6)
        expected = []

        def numeric(v):
            return plan.evolve(
                np.stack((v[:2], v[:2] * 0.7)), np.stack((v[2:4], v[2:4] * 1.2)), v[4]
            ).outputs.sum()

        point = np.asarray(value).astype(float)
        for i in range(5):
            direction = np.eye(5)[i] * 1e-5
            expected.append((numeric(point + direction) - numeric(point - direction)) / 2e-5)
        np.testing.assert_allclose(first, expected, rtol=0.003, atol=3e-5)


@pytest.mark.parametrize("backend", ["numpy", "torch", "jax"])
def test_masked_batched_paths_match_streamed_density_carry(backend):
    if backend == "numpy":
        xp, dtype = np, np.float64
    else:
        lib = pytest.importorskip(backend)
        xp = lib if backend == "torch" else lib.numpy
        dtype = xp.float64 if backend == "torch" else xp.float32
    plan = oq.LindbladPlan(1, ("X",), (), (0,), substeps=2)
    h = xp.asarray([[[0.3], [0.2], [0.1]], [[0.4], [0.1], [0.5]]], dtype=dtype)
    rates = xp.ones_like(h) * 0.2
    mask = xp.asarray([[1, 1, 0], [1, 0, 0]], dtype=dtype)
    full = plan.evolve(h, rates, 0.2, backend=backend, mask=mask)
    first = plan.evolve(h[:, :1], rates[:, :1], 0.2, backend=backend)
    second = plan.evolve(
        h[:, 1:], rates[:, 1:], 0.2, backend=backend, state=first.state, mask=mask[:, 1:]
    )

    def array(value):
        return value.detach().numpy() if backend == "torch" else np.asarray(value)

    np.testing.assert_allclose(array(second.state), array(full.state), atol=3e-6)
    np.testing.assert_allclose(
        array(full.outputs)[1, 1:], array(full.outputs)[1, :1].repeat(2, axis=0), atol=3e-6
    )


def test_torch_layer_trains_classical_controls_and_validates_checkpoint_identity():
    torch = pytest.importorskip("torch")
    plan = oq.LindbladPlan(1, ("X", "Z"), (), (0,), substeps=2)
    model = plan.torch_layer(3, dtype=torch.float64, device="cpu")
    values = torch.randn((2, 4, 3), dtype=torch.float64)
    old = model.encoder.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    loss = model(values, 0.1).outputs.square().mean()
    loss.backward()
    optimizer.step()
    assert not torch.equal(old, model.encoder.weight)
    restored = plan.torch_layer(3, dtype=torch.float64, device="cpu")
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(values, 0.1).outputs, model(values, 0.1).outputs)
    different = oq.LindbladPlan(1, ("Y", "Z"), (), (0,), substeps=2).torch_layer(
        3, dtype=torch.float64, device="cpu"
    )
    with pytest.raises(oq.ValidationError):
        different.load_state_dict(model.state_dict())


def test_flax_lindblad_layer_composes_with_native_parameter_trees():
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    plan = oq.LindbladPlan(1, ("X", "Z"), (), (0,), substeps=2)
    model = plan.flax_layer()
    inputs = jax.numpy.ones((2, 3, 4), dtype=jax.numpy.float32)
    state = model.init(jax.random.PRNGKey(4), inputs, 0.1)
    value, grad = jax.jit(
        jax.value_and_grad(lambda p: model.apply({"params": p}, inputs, 0.1).outputs.sum())
    )(state["params"])
    assert np.isfinite(value) and all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(grad))


def test_native_torch_lindblad_fullgraph_backward_without_cpp_toolchain():
    torch = pytest.importorskip("torch")
    plan = oq.LindbladPlan(1, ("X",), (), (0,), substeps=1)

    def function(value):
        return plan.evolve(
            value, torch.ones_like(value) * 0.2, 0.1, backend="torch", checkpoint=False
        ).outputs

    value = torch.tensor([[0.3], [0.4]], dtype=torch.float64, requires_grad=True)
    expected = function(value)
    actual = torch.compile(function, backend="aot_eager", fullgraph=True)(value)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), value)[0], torch.autograd.grad(expected.sum(), value)[0]
    )


def test_lindblad_admission_error_bounds_and_invalid_physical_inputs():
    with pytest.raises(oq.ResourceError):
        oq.LindbladPlan(32, ("X" * 32,))
    plan = oq.LindbladPlan(1, ("X",), (), (0,))
    with pytest.raises(oq.ValidationError):
        plan.evolve([[0.3]], [[-0.2]])
    with pytest.raises(oq.ValidationError):
        plan.evolve([[0.3]], [[0.2]], state=np.diag([1.1, -0.1]))
    with pytest.raises(oq.ResourceError, match="substeps"):
        plan.evolve([[0.3]], [[0.2]], tolerance=1e-12)
