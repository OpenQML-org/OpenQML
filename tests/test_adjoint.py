"""Adjoint contractions agree with independent derivatives and retain bounded state storage."""

import weakref

import numpy as np
import pytest

import openqml as oq
from openqml import (
    Circuit,
    ClassicalCondition,
    Hamiltonian,
    LayerCheckpoint,
    LayerSpec,
    Parameter,
    PauliString,
    ResourceError,
    ValidationError,
    adjoint_jvp,
    adjoint_vjp,
    convert,
    expectations,
    parameter_shift_jacobian,
    parameter_shift_vjp,
)

PARAMETRIC = ("rx", "ry", "rz", "p", "crx", "cry", "crz", "cp", "rxx", "ryy", "rzz", "u")
OBSERVABLES = (
    PauliString("XYZ", 0.7),
    Hamiltonian((PauliString("YXX", -0.3), PauliString("IIY", 0.2))),
    PauliString("III", -0.4),
)


def mixed_circuit():
    x, w, phase = (Parameter(n) for n in ("feature", "weight", "phase"))
    return (
        Circuit(3)
        .h(0)
        .sx(1)
        .ry(0.4, 2)
        .u(x * w, w.sin(), x + w, 1)
        .crx(x.sin() + w**2, 2, 0)
        .cry(w * 0.3, 0, 2)
        .crz(x, 1, 0)
        .cp(w, 2, 1)
        .rxx(x * w, 0, 2)
        .ryy(w, 2, 1)
        .rzz(x, 1, 0)
        .iswap(1, 2)
        .cswap(1, 0, 2)
        .ccx(0, 1, 2)
        .sxdg(0)
        .unitary(np.array([[0, 1j], [1j, 0]]), 2)
        .barrier()
        .with_global_phase(phase.sin() + x * w)
    )


def adapted_checkpoint():
    x, w, unused = (Parameter(n) for n in ("feature", "weight", "unused"))
    spec = LayerSpec(
        Circuit(1).rx(unused, 0).rx(-unused, 0).ry(x.sin() + x * w + w**2, 0),
        ("feature",),
        ("unused", "weight"),
    )
    return LayerCheckpoint(spec, [0.5, 0.3], "float64")


@pytest.mark.parametrize("gate", PARAMETRIC)
def test_all_parametric_gates_against_five_point_derivatives(gate):
    names = ("theta", "phi", "lambda") if gate == "u" else ("theta",)
    symbols = tuple(Parameter(name) for name in names)
    qubits = (2, 0) if gate in {"crx", "cry", "crz", "cp", "rxx", "ryy", "rzz"} else (1,)
    circuit = Circuit(3).h(0).ry(0.6, 1).rx(-0.4, 2).gate(gate, *qubits, params=symbols).cx(1, 2)
    row = dict(zip(names, (0.37, -0.2, 0.61), strict=False))
    cotangents = np.array([[0.6, -0.3, 0.9]])
    result = adjoint_vjp(circuit, OBSERVABLES, [row], cotangents, parameters=names)[0]
    h = 1e-4
    expected = []
    for name in names:
        values = []
        for shift in (-2, -1, 1, 2):
            values.append(
                cotangents[0]
                @ expectations(circuit, OBSERVABLES, row | {name: row[name] + shift * h})
            )
        expected.append((values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * h))
    np.testing.assert_allclose(result, expected, atol=3e-11, rtol=1e-10)


def test_shared_nonlinear_symbols_multi_parameter_gates_and_selected_order():
    circuit = mixed_circuit()
    snapshot = circuit.to_json()
    rows = [
        {"feature": 0.3, "weight": -0.4, "phase": 0.7},
        {"feature": -0.6, "weight": 0.2, "phase": -0.8},
    ]
    names = ("weight", "phase", "feature")
    cotangents = np.array([[0.5, -0.3, 0.4], [-0.2, 0.6, 0.3]])
    tangents = np.array([[0.2, 0.7, -0.3], [0.4, -0.3, 0.2]])
    actual = adjoint_vjp(circuit, OBSERVABLES, rows, cotangents, parameters=names)
    jacobian = parameter_shift_jacobian(circuit, OBSERVABLES, rows, parameters=names)
    np.testing.assert_allclose(actual, np.einsum("bmp,bm->bp", jacobian, cotangents), atol=2e-12)
    np.testing.assert_array_equal(actual[:, 1], 0)
    direction = adjoint_jvp(circuit, OBSERVABLES, rows, tangents, parameters=names)
    np.testing.assert_allclose(direction, np.einsum("bmp,bp->bm", jacobian, tangents), atol=2e-12)
    np.testing.assert_allclose(
        np.sum(direction * cotangents, axis=1), np.sum(actual * tangents, axis=1), atol=2e-12
    )
    assert circuit.to_json() == snapshot


def test_constant_phase_empty_rows_and_zero_products_do_not_evolve_states(monkeypatch):
    import openqml.adjoint as engine

    def fail(*args, **kwargs):
        raise AssertionError("A zero gradient evolved a state.")

    monkeypatch.setattr(engine, "_forward", fail)
    phase = Circuit(1).h(0).with_global_phase(Parameter("phase"))
    np.testing.assert_array_equal(
        adjoint_vjp(phase, [PauliString("Y")], [{"phase": 0.4}], [[1]]), [[0]]
    )
    circuit = Circuit(1).ry(Parameter("weight"), 0)
    obs, rows = [PauliString("Z")], [{"weight": 0.3}]
    assert adjoint_vjp(circuit, obs, [], np.empty((0, 1))).shape == (0, 1)
    assert adjoint_jvp(circuit, obs, [], np.empty((0, 1))).shape == (0, 1)
    np.testing.assert_array_equal(adjoint_vjp(circuit, obs, rows, [[0]]), [[0]])
    np.testing.assert_array_equal(adjoint_jvp(circuit, obs, rows, [[0]]), [[0]])
    assert adjoint_vjp(circuit, obs, rows, [[1]], parameters=[]).shape == (1, 0)


def test_state_work_is_linear_and_live_vectors_do_not_grow_with_depth(monkeypatch):
    import openqml.adjoint as engine

    apply = engine._apply_matrix
    references, calls, live = [], 0, 0

    def tracked(*args, **kwargs):
        nonlocal calls, live
        value = apply(*args, **kwargs)
        calls += 1
        references.append(weakref.ref(value))
        live = max(live, sum(ref() is not None for ref in references))
        return value

    monkeypatch.setattr(engine, "_apply_matrix", tracked)
    peaks = []
    for depth in (12, 120):
        calls, live = 0, 0
        references.clear()
        circuit = Circuit(2)
        for i in range(depth):
            circuit = circuit.ry(Parameter("weight") / depth, i % 2)
        result = adjoint_vjp(circuit, [PauliString("ZZ")], [{"weight": 0.4}], [[1]])
        np.testing.assert_allclose(result, [[-np.cos(0.2) * np.sin(0.2)]], atol=2e-12)
        assert calls == 4 * depth
        peaks.append(live)
    assert max(peaks) <= 7 and peaks[1] <= peaks[0] + 1


@pytest.mark.parametrize("method", [adjoint_vjp, adjoint_jvp])
@pytest.mark.parametrize("value", [[[True]], [[1j]], [[np.inf]], [[np.nan]], [1.0], "one"])
def test_products_require_finite_real_arrays_with_exact_shape(method, value):
    with pytest.raises(ValidationError):
        method(Circuit(1).ry(Parameter("w"), 0), [PauliString("Z")], [{"w": 0.3}], value)


@pytest.mark.parametrize("names", ["w", {"w"}, {"w": 1}, ["missing"], ["w", "w"], [True], 3])
def test_parameter_selections_are_ordered_known_unique_names(names):
    with pytest.raises(ValidationError):
        adjoint_vjp(
            Circuit(1).ry(Parameter("w"), 0),
            [PauliString("Z")],
            [{"w": 0.3}],
            [[1]],
            parameters=names,
        )


@pytest.mark.parametrize(
    "rows",
    [
        [{}],
        [{"w": 0.3, "extra": 1}],
        [{"w": True}],
        [{"w": np.inf}],
        [{"w": 1j}],
        [{"w": [0.3]}],
        [0.3],
        None,
    ],
)
def test_binding_validation_precedes_simulation(monkeypatch, rows):
    import openqml.adjoint as engine

    monkeypatch.setattr(engine, "_forward", lambda *args: pytest.fail("Simulated invalid bindings"))
    with pytest.raises(ValidationError):
        adjoint_vjp(Circuit(1).ry(Parameter("w"), 0), [PauliString("Z")], rows, [[1]])


def test_late_invalid_expression_and_observable_are_preflighted(monkeypatch):
    import openqml.adjoint as engine

    monkeypatch.setattr(
        engine, "_forward", lambda *args: pytest.fail("Simulation preceded preflight")
    )
    circuit = Circuit(1).ry(Parameter("w").sqrt(), 0)
    with np.errstate(invalid="ignore"), pytest.raises(ValidationError):
        adjoint_vjp(circuit, [PauliString("Z")], [{"w": 0.4}, {"w": -0.1}], [[1], [1]])
    for observables in (
        [],
        [PauliString("ZZ")],
        ["Z"],
        None,
        {PauliString("Z")},
        {PauliString("Z"): 1},
    ):
        with pytest.raises(ValidationError):
            adjoint_vjp(circuit, observables, [{"w": 0.4}], [[1]])


@pytest.mark.parametrize(
    "circuit",
    [
        Circuit(1).measure_all(),
        Circuit(1).reset(0),
        Circuit(1).bit_flip(0.2, 0),
        Circuit(1).channel([np.eye(2)], 0),
        Circuit(1).if_else(ClassicalCondition((0,), 0), Circuit(1).x(0)),
    ],
)
def test_effects_are_rejected_instead_of_assuming_unitarity(circuit):
    with pytest.raises(ValidationError, match="unitary"):
        adjoint_vjp(circuit, [PauliString("Z")], [{}], [[1]])


def test_state_and_local_matrix_budgets_precede_allocation(monkeypatch):
    import openqml.adjoint as engine

    monkeypatch.setattr(
        engine, "_forward", lambda *args: pytest.fail("Allocated an over-budget state")
    )
    for circuit, options in (
        (Circuit(1_000_000_000), {}),
        (Circuit(3).ry(Parameter("w"), 0), {"max_qubits": 2}),
        (Circuit(4).unitary(np.eye(16), 0, 1, 2, 3), {"max_memory_mb": 0.01}),
        (Circuit(2).ry(Parameter("w"), 0), {"max_memory_mb": 0.0001}),
    ):
        with pytest.raises(ResourceError, match="Adjoint"):
            adjoint_vjp(circuit, [PauliString("Z")], [{}], [[1]], **options)
    for option, value in (("max_qubits", True), ("max_memory_mb", 0), ("max_memory_mb", np.inf)):
        with pytest.raises(ValidationError):
            adjoint_vjp(Circuit(1), [PauliString("Z")], [{}], [[1]], **{option: value})


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "openqasm3", "pennylane"])
def test_imported_programs_share_the_adjoint_ir_path(framework):
    if framework != "openqasm3":
        pytest.importorskip(framework)
    source = Circuit(2).ry(Parameter("feature"), 0).cx(0, 1).rx(Parameter("weight"), 1)
    if framework == "pennylane":
        source = source.bind({"feature": 0.4, "weight": 0.3})
    native = convert(source, target=framework).value
    restored = convert(
        native,
        source=framework,
        target="ir",
        source_options={"parser": "builtin"} if framework == "openqasm3" else {},
    ).value
    rows = [dict.fromkeys(restored.parameters, 0.3)]
    observables = [PauliString("ZZ"), PauliString("YX")]
    cotangents = [[0.2, 0.8]]
    np.testing.assert_allclose(
        adjoint_vjp(restored, observables, rows, cotangents),
        parameter_shift_vjp(source, observables, rows, cotangents),
        atol=2e-12,
    )


def test_torch_adjoint_broadcasts_joint_gradients_and_forward_ad_without_shifts(monkeypatch):
    torch = pytest.importorskip("torch")
    import openqml.gradients as shifts
    from openqml.interfaces.torch import QuantumLayer

    monkeypatch.setattr(
        shifts, "parameter_shift_vjp", lambda *args, **kwargs: pytest.fail("Used shifted circuits")
    )
    monkeypatch.setattr(
        shifts, "parameter_shift_jvp", lambda *args, **kwargs: pytest.fail("Used shifted circuits")
    )
    layer = QuantumLayer.from_checkpoint(
        adapted_checkpoint(),
        optimize=True,
        backend="numpy",
        gradient_method="adjoint",
        batch_chunk_size=2,
    )
    inputs = torch.tensor([[[0.2], [-0.3], [0.4]]], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[[0.7, 0.3]], [[0.1, -0.4]]], dtype=torch.float64, requires_grad=True)
    actual = layer.functional(weights, inputs)
    w = weights[..., 1:]
    reference = torch.cos(torch.sin(inputs) + inputs * w + w**2)
    got = torch.autograd.grad(actual.sum(), (weights, inputs))
    expected = torch.autograd.grad(reference.sum(), (weights, inputs))
    torch.testing.assert_close(actual, reference, atol=2e-12, rtol=2e-12)
    for a, b in zip(got, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
    with torch.autograd.forward_ad.dual_level():
        dual_x = torch.autograd.forward_ad.make_dual(inputs, torch.full_like(inputs, 0.2))
        dual_w = torch.autograd.forward_ad.make_dual(weights, torch.full_like(weights, 0.1))
        primal, tangent = torch.autograd.forward_ad.unpack_dual(layer.functional(dual_w, dual_x))
        _, expected_tangent = torch.autograd.forward_ad.unpack_dual(
            torch.cos(torch.sin(dual_x) + dual_x * dual_w[..., 1:] + dual_w[..., 1:] ** 2)
        )
    torch.testing.assert_close(primal, reference, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(tangent, expected_tangent, atol=2e-12, rtol=2e-12)
    assert layer.checkpoint().spec == adapted_checkpoint().spec


def test_torch_adjoint_checkpoint_freezing_and_one_optimizer():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    checkpoint = adapted_checkpoint().freeze(["unused"])
    quantum = QuantumLayer.from_checkpoint(checkpoint, backend="numpy", gradient_method="adjoint")
    encoder = torch.nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        encoder.weight.fill_(0.2)
        encoder.bias.fill_(0.1)
    model = torch.nn.Sequential(encoder, quantum)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = [p.detach().clone() for p in model.parameters()]
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.ones(3, 2, dtype=torch.float64)).square().mean().backward()
        optimizer.step()
    assert all(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    assert quantum.to_spec() == checkpoint.spec


def test_jax_adjoint_jit_broadcasts_directional_products_and_vmap():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    with oq.jax.enable_x64():
        layer, _ = QuantumLayer.from_checkpoint(
            adapted_checkpoint(),
            optimize=True,
            backend="numpy",
            gradient_method="adjoint",
            batch_chunk_size=2,
        )
        inputs = jnp.array([[[0.2], [-0.3], [0.4]]], dtype=jnp.float64)
        weights = jnp.array([[[0.7, 0.3]], [[0.1, -0.4]]], dtype=jnp.float64)

        def reference(w, x):
            return jnp.cos(jnp.sin(x) + x * w[..., 1:] + w[..., 1:] ** 2)

        np.testing.assert_allclose(
            jax.jit(layer)(weights, inputs), reference(weights, inputs), atol=2e-12
        )
        actual = jax.jit(jax.grad(lambda w, x: layer(w, x).sum(), argnums=(0, 1)))(weights, inputs)
        expected = jax.grad(lambda w, x: reference(w, x).sum(), argnums=(0, 1))(weights, inputs)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, atol=2e-12)
        tangents = jnp.full_like(weights, 0.1), jnp.full_like(inputs, 0.2)
        primal, direction = jax.jit(layer.jvp)(weights, inputs, *tangents)
        expected_primal, expected_direction = jax.jvp(reference, (weights, inputs), tangents)
        np.testing.assert_allclose(primal, expected_primal, atol=2e-12)
        np.testing.assert_allclose(direction, expected_direction, atol=2e-12)
        mapped = jax.jit(jax.vmap(jax.grad(lambda w: layer(w, inputs[0]).sum())))(weights[:, 0])
        np.testing.assert_allclose(mapped, expected[0][:, 0], atol=2e-12)


def test_flax_adjoint_classical_weight_generator_uses_one_optax_tree():
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    optax = pytest.importorskip("optax")
    import jax.numpy as jnp

    from openqml.interfaces.flax import QuantumLayer

    with oq.jax.enable_x64():
        checkpoint = adapted_checkpoint().freeze(["unused"])
        module, variables = QuantumLayer.from_checkpoint(
            checkpoint, backend="numpy", gradient_method="adjoint"
        )
        inputs = jnp.array([[0.2], [0.4]], dtype=jnp.float64)
        params = {
            "encoder": jnp.array([[0.4]], dtype=jnp.float64),
            "generator": jnp.array([[0.3]], dtype=jnp.float64),
        }

        def loss(p):
            return jnp.square(
                module.apply(variables, inputs @ p["encoder"], weights=inputs @ p["generator"])
            ).mean()

        gradients = jax.jit(jax.grad(loss))(params)
        assert all(np.any(np.asarray(g) != 0) for g in gradients.values())
        optimizer = optax.adam(0.01)
        updates, _ = optimizer.update(gradients, optimizer.init(params), params)
        updated = optax.apply_updates(params, updates)
        assert float(loss(updated)) < float(loss(params))


@pytest.mark.parametrize("interface", ["torch", "jax"])
def test_explicit_gradient_methods_reject_incompatible_backends_and_effects(interface):
    pytest.importorskip(interface)
    if interface == "torch":
        from openqml.interfaces.torch import QuantumLayer
    else:
        from openqml.interfaces.jax import QuantumLayer
    spec = LayerSpec(Circuit(1).ry(Parameter("w"), 0), (), ("w",))
    for method in (None, {}, True, "finite-difference"):
        with pytest.raises(ValidationError, match="gradient_method"):
            QuantumLayer.from_spec(spec, gradient_method=method)
    with pytest.raises(ValidationError, match="external backend"):
        QuantumLayer.from_spec(spec, gradient_method="adjoint")
    with pytest.raises(ValidationError, match="numpy"):
        QuantumLayer.from_spec(spec, backend="aer", gradient_method="adjoint")
    effect = LayerSpec(Circuit(1).ry(Parameter("w"), 0).bit_flip(0.2, 0), (), ("w",))
    with pytest.raises(ValidationError, match="unitary"):
        QuantumLayer.from_spec(effect, backend="numpy", gradient_method="adjoint")
