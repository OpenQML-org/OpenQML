"""Portable symbol adaptation and frozen checkpoints preserve training semantics."""

import numpy as np
import pytest

import openqml as oq
from openqml import (
    Circuit,
    ClassicalCondition,
    ConversionError,
    Hamiltonian,
    LayerCheckpoint,
    LayerSpec,
    Parameter,
    PauliString,
    ValidationError,
    convert,
    density_matrix,
    expectations,
    parameter_shift_jacobian,
    plan_conversion,
    statevector,
)


def source_spec():
    feature, a, b = (Parameter(n) for n in ("feature", "alpha", "beta"))
    return LayerSpec(
        Circuit(1).ry(feature, 0).ry(a, 0).ry(b, 0).with_global_phase(b / 7),
        ("feature",),
        ("beta", "alpha"),
        (Hamiltonian((PauliString("Z", 0.7), PauliString("I", -0.2))),),
    )


def adapted_spec(source=None):
    feature, gain = Parameter("feature"), Parameter("gain")
    return (source_spec() if source is None else source).reparameterize(
        {"feature": feature.sin(), "alpha": feature * gain, "beta": gain**2},
        input_parameters=("feature",),
        weight_parameters=("gain",),
    )


def test_substitution_is_simultaneous_for_swaps_chains_and_self_references():
    a, b, c = (Parameter(n) for n in ("a", "b", "c"))
    source = Circuit(1).ry(a + 2 * b, 0).with_global_phase(a.sin() - b)
    snapshot = source.to_json()
    for mapping, values, angle, phase in (
        ({"a": b, "b": a}, {"a": 0.2, "b": 0.4}, 0.8, np.sin(0.4) - 0.2),
        ({"a": b, "b": c}, {"b": 0.2, "c": 0.4}, 1.0, np.sin(0.2) - 0.4),
        ({"a": a + 1}, {"a": 0.2, "b": 0.4}, 2.0, np.sin(1.2) - 0.4),
    ):
        result = source.substitute(mapping)
        expected = np.exp(1j * phase) * np.array([np.cos(angle / 2), np.sin(angle / 2)])
        np.testing.assert_allclose(statevector(result, values), expected, atol=1e-12)
        assert Circuit.from_json(result.to_json()) == result
    assert source.to_json() == snapshot
    assert source.substitute({}) is source


def test_nonlinear_tying_preserves_readouts_phase_and_symbolic_chain_rule():
    source = source_spec()
    snapshot = source.to_json()
    result = adapted_spec(source)
    assert result.input_parameters == ("feature",)
    assert result.weight_parameters == ("gain",)
    assert result.observables == source.observables
    x, g = 0.3, -0.4
    angle = np.sin(x) + x * g + g**2
    np.testing.assert_allclose(
        statevector(result.circuit, {"feature": x, "gain": g}),
        np.exp(1j * g**2 / 7) * np.array([np.cos(angle / 2), np.sin(angle / 2)]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        parameter_shift_jacobian(
            result.circuit,
            result.observables,
            [{"feature": x, "gain": g}],
            parameters=("feature", "gain"),
        )[0],
        [[-0.7 * np.sin(angle) * (np.cos(x) + g), -0.7 * np.sin(angle) * (x + 2 * g)]],
        atol=1e-12,
    )
    assert LayerSpec.from_json(result.to_json()) == result
    assert source.to_json() == snapshot


def test_substitution_preserves_nested_feedback_channels_and_all_branch_phases():
    data, a, b, noise = (Parameter(n) for n in ("data", "alpha", "beta", "noise"))
    source = (
        Circuit(2)
        .ry(data, 0)
        .measure(0, 3)
        .if_else(
            ClassicalCondition((3,), 1),
            Circuit(2).ry(a, 1).amplitude_damping(noise, 1).with_global_phase(a.sin()),
            Circuit(2).rx(b, 1).with_global_phase(b / 9),
        )
        .with_global_phase(a / 3)
    )
    x, g = Parameter("feature"), Parameter("gain")
    result = source.substitute({"data": x.sin(), "alpha": x * g, "beta": g**2, "noise": 0.25})
    x, g = 0.7, 0.4
    expected = (
        Circuit(2)
        .ry(np.sin(x), 0)
        .measure(0, 3)
        .if_else(
            ClassicalCondition((3,), 1),
            Circuit(2).ry(x * g, 1).amplitude_damping(0.25, 1).with_global_phase(np.sin(x * g)),
            Circuit(2).rx(g**2, 1).with_global_phase(g**2 / 9),
        )
        .with_global_phase(x * g / 3)
    )
    bound = result.bind({"feature": x, "gain": g})
    assert bound.num_clbits == 4 and bound.operations[2].condition == source.operations[2].condition
    np.testing.assert_allclose(bound.global_phase.constant, x * g / 3)
    for actual, wanted in zip(
        bound.operations[2].blocks, expected.operations[2].blocks, strict=True
    ):
        np.testing.assert_allclose(actual.global_phase.constant, wanted.global_phase.constant)
    np.testing.assert_allclose(density_matrix(bound), density_matrix(expected), atol=1e-12)
    plan = plan_conversion(result, target="openqasm3")
    diagnostic = next(d for d in plan.diagnostics if d.code == "CHANNEL_UNSUPPORTED")
    assert diagnostic.location == "operations[2].blocks[0].operations[1]"
    with pytest.raises(ConversionError, match="CHANNEL_UNSUPPORTED"):
        plan.export()


def test_symbol_rewrites_reuse_unmodified_matrix_and_kraus_operations():
    source = (
        Circuit(1)
        .unitary(np.array([[0, 1j], [1j, 0]]), 0)
        .channel([np.eye(2)], 0)
        .ry(Parameter("angle"), 0)
    )
    result = source.substitute({"angle": Parameter("weight").sin()})
    assert all(result.operations[i] is source.operations[i] for i in (0, 1))
    np.testing.assert_allclose(
        density_matrix(result, {"weight": 0.3}),
        density_matrix(source, {"angle": np.sin(0.3)}),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "mapping",
    [
        None,
        [],
        [("alpha", 0.2)],
        "alpha",
        {1: 0.2},
        {Parameter("alpha"): 0.2},
        {"missing": 0.2},
        {"alpha": True},
        {"alpha": "weight"},
        {"alpha": np.inf},
        {"alpha": complex(1, 1)},
        {"alpha": [0.2]},
    ],
)
def test_invalid_substitution_contracts_fail_without_changing_source(mapping):
    source = source_spec().circuit
    snapshot = source.to_json()
    with pytest.raises(ValidationError):
        source.substitute(mapping)
    assert source.to_json() == snapshot


@pytest.mark.parametrize(
    "inputs,weights",
    [
        (("feature",), ()),
        (("feature", "gain"), ("gain",)),
        (("feature",), ("gain", "gain")),
        (("feature",), ("gain", "alpha")),
        ("feature", ("gain",)),
        (("feature",), {"gain"}),
    ],
)
def test_reparameterization_requires_complete_unambiguous_ordered_ownership(inputs, weights):
    with pytest.raises(ValidationError):
        source_spec().reparameterize(
            {"alpha": Parameter("gain"), "beta": 0.2},
            input_parameters=inputs,
            weight_parameters=weights,
        )


def test_partial_binding_prunes_folded_symbols_and_preserves_remaining_order():
    x, y, a, b = (Parameter(n) for n in ("x", "y", "a", "b"))
    spec = LayerSpec(
        Circuit(1).ry(x * a + y + b, 0).with_global_phase(a * x),
        ("y", "x"),
        ("b", "a"),
    )
    result = spec.bind({"x": np.float64(0)})
    assert result.input_parameters == ("y",) and result.weight_parameters == ("b",)
    np.testing.assert_allclose(
        expectations(result.circuit, result.observables, {"y": 0.2, "b": 0.3}),
        [np.cos(0.5)],
        atol=1e-12,
    )
    assert result.bind({"y": 0.2, "b": 0.3}).circuit.parameters == ()
    assert LayerSpec.from_json(result.to_json()) == result
    assert spec.bind({}) == spec
    assert spec.input_parameters == ("y", "x") and spec.weight_parameters == ("b", "a")


@pytest.mark.parametrize(
    "bindings",
    [
        None,
        [],
        {1: 0.1, "missing": 0.2},
        {"missing": 0.1},
        {"alpha": Parameter("gain")},
        {"alpha": True},
        {"alpha": np.nan},
        {"alpha": 1j},
    ],
)
def test_partial_binding_accepts_only_known_finite_real_bindings(bindings):
    with pytest.raises(ValidationError):
        source_spec().bind(bindings)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_freezing_uses_saved_precision_and_retains_named_weights_and_source(dtype):
    source = LayerCheckpoint(source_spec(), {"alpha": 0.123456789, "beta": -0.35}, dtype)
    snapshot = source.to_json()
    result = source.freeze(name for name in ("alpha",))
    assert result.dtype == dtype and result.spec.weight_parameters == ("beta",)
    assert result.bindings == {"beta": source.bindings["beta"]}
    original = source.bindings | {"feature": 0.4}
    frozen = result.bindings | {"feature": 0.4}
    np.testing.assert_allclose(
        statevector(result.spec.circuit, frozen),
        statevector(source.spec.circuit, original),
        atol=1e-12,
    )
    assert LayerCheckpoint.from_json(result.to_json()) == result
    assert source.freeze([]) == source and source.freeze(["beta", "alpha"]) == source.freeze()
    assert source.freeze().weights == source.freeze().spec.weight_parameters == ()
    assert source.to_json() == snapshot


def test_freezing_can_eliminate_inputs_and_canceled_weight_symbols():
    x, a, b = (Parameter(n) for n in ("feature", "alpha", "beta"))
    source = LayerCheckpoint(
        LayerSpec(Circuit(1).ry(a * (x + b), 0), ("feature",), ("beta", "alpha")), [0.3, 0.0]
    )
    result = source.freeze(["alpha"])
    assert result.spec.input_parameters == result.spec.weight_parameters == result.weights == ()
    np.testing.assert_allclose(expectations(result.spec.circuit, result.spec.observables), [1.0])


@pytest.mark.parametrize(
    "names",
    ["alpha", {"alpha"}, {"alpha": 1}, ["feature"], ["missing"], ["alpha", "alpha"], [False], 3],
)
def test_freezing_rejects_invalid_or_ambiguous_weight_selections(names):
    source = LayerCheckpoint(source_spec(), [0.2, 0.3])
    with pytest.raises(ValidationError):
        source.freeze(names)


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "openqasm3"])
def test_imported_symbols_can_be_retied_and_exported_without_phase_loss(framework):
    if framework != "openqasm3":
        pytest.importorskip(framework)
    source = source_spec()
    native = convert(source.circuit, target=framework).value
    imported = LayerSpec.from_program(
        native,
        ("feature",),
        framework=framework,
        **({"parser": "builtin"} if framework == "openqasm3" else {}),
    )
    result = adapted_spec(imported)
    exported = convert(result.circuit, target=framework)
    restored = convert(
        exported.value,
        source=framework,
        target="ir",
        source_options={"parser": "builtin"} if framework == "openqasm3" else {},
    ).value
    bindings = {"feature": 0.3, "gain": -0.4}
    np.testing.assert_allclose(
        statevector(restored, bindings), statevector(adapted_spec().circuit, bindings), atol=1e-12
    )


def test_fully_bound_adapted_circuit_roundtrips_pennylane_and_unbound_is_diagnosed():
    pytest.importorskip("pennylane")
    result = adapted_spec()
    plan = plan_conversion(result.circuit, target="pennylane")
    assert any(d.code == "UNBOUND_PARAMETER" for d in plan.diagnostics)
    with pytest.raises(ConversionError, match="UNBOUND_PARAMETER"):
        plan.export()
    bound = result.bind({"feature": 0.3, "gain": -0.4})
    native = convert(bound.circuit, target="pennylane").value
    restored = convert(native, target="ir").value
    np.testing.assert_allclose(statevector(restored), statevector(bound.circuit), atol=1e-12)


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_torch_reparameterized_broadcasts_accumulate_native_and_bridge_gradients(backend):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    layer = QuantumLayer.from_spec(
        adapted_spec(),
        backend=backend,
        dtype=torch.float64,
        batch_chunk_size=2,
        **({"checkpoint_interval": 2} if backend == "torch" else {}),
    )
    weights = torch.tensor([[[-0.4]], [[0.2]]], dtype=torch.float64, requires_grad=True)
    inputs = torch.tensor([[[0.3], [-0.2], [0.6]]], dtype=torch.float64, requires_grad=True)
    actual = layer.functional(weights, inputs)
    angle = torch.sin(inputs) + inputs * weights + weights**2
    expected = 0.7 * torch.cos(angle) - 0.2
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    got = torch.autograd.grad(actual.sum(), (weights, inputs), create_graph=backend == "torch")
    want = torch.autograd.grad(expected.sum(), (weights, inputs), create_graph=backend == "torch")
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    if backend == "torch":
        torch.testing.assert_close(
            torch.autograd.grad(got[0].sum(), weights)[0],
            torch.autograd.grad(want[0].sum(), weights)[0],
            atol=1e-12,
            rtol=1e-12,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_jax_reparameterized_broadcasts_jit_grad_and_native_curvature(backend):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    with oq.jax.enable_x64():
        layer = QuantumLayer.from_spec(
            adapted_spec(),
            backend=backend,
            batch_chunk_size=2,
            **({"checkpoint_interval": 2} if backend == "jax" else {}),
        )
        weights = jnp.array([[[-0.4]], [[0.2]]], dtype=jnp.float64)
        inputs = jnp.array([[[0.3], [-0.2], [0.6]]], dtype=jnp.float64)

        def reference(w, x):
            return 0.7 * jnp.cos(jnp.sin(x) + x * w + w**2) - 0.2

        np.testing.assert_allclose(
            jax.jit(layer)(weights, inputs), reference(weights, inputs), atol=1e-12
        )
        actual = jax.jit(jax.grad(lambda w, x: layer(w, x).sum(), argnums=(0, 1)))(weights, inputs)
        expected = jax.grad(lambda w, x: reference(w, x).sum(), argnums=(0, 1))(weights, inputs)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-12)
        if backend == "jax":
            np.testing.assert_allclose(
                jax.jit(jax.hessian(lambda w: layer(w, inputs).sum()))(weights),
                jax.hessian(lambda w: reference(w, inputs).sum())(weights),
                atol=1e-12,
            )


def test_frozen_checkpoint_resumes_joint_torch_training_and_exports_constants():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    source = LayerCheckpoint(source_spec(), [0.2, -0.3], "float64")
    checkpoint = source.freeze(["alpha"])
    quantum = QuantumLayer.from_checkpoint(checkpoint, batch_chunk_size=2)
    encoder = torch.nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        encoder.weight.fill_(0.1)
        encoder.bias.fill_(0.2)
    model = torch.nn.Sequential(encoder, quantum)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial = [p.detach().clone() for p in model.parameters()]
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.ones(3, 2, dtype=torch.float64)).square().mean().backward()
        optimizer.step()
    assert all(not torch.equal(a, b) for a, b in zip(initial, model.parameters(), strict=True))
    trained = quantum.checkpoint()
    assert trained.spec == checkpoint.spec and set(trained.bindings) == {"beta"}
    exported = convert(trained.spec.circuit, target="openqasm3")
    assert "alpha" not in exported.value
    restored = convert(
        exported.value, source="openqasm3", target="ir", source_options={"parser": "builtin"}
    ).value
    values = trained.bindings | {"feature": 0.4}
    np.testing.assert_allclose(
        statevector(restored, values),
        statevector(source.spec.circuit, values | {"alpha": source.bindings["alpha"]}),
        atol=1e-12,
    )


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_fully_frozen_torch_layers_keep_input_gradients_and_empty_weight_gradients(backend):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    source = LayerCheckpoint(source_spec(), [0.2, -0.3], "float64")
    layer = QuantumLayer.from_checkpoint(source.freeze(), backend=backend, batch_chunk_size=2)
    inputs = torch.tensor([[0.2], [0.4], [-0.3]], dtype=torch.float64, requires_grad=True)
    output = layer(inputs)
    reference = 0.7 * torch.cos(inputs + sum(source.weights)) - 0.2
    torch.testing.assert_close(output, reference, atol=1e-12, rtol=1e-12)
    actual = torch.autograd.grad(output.sum(), (inputs, layer.weights))
    expected = torch.autograd.grad(reference.sum(), inputs)[0]
    torch.testing.assert_close(actual[0], expected, atol=1e-12, rtol=1e-12)
    assert actual[1].shape == (0,)


def test_partially_and_fully_frozen_checkpoints_transfer_into_jax_and_flax():
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    optax = pytest.importorskip("optax")
    import jax.numpy as jnp

    from openqml.interfaces.flax import QuantumLayer as FlaxLayer
    from openqml.interfaces.jax import QuantumLayer

    with oq.jax.enable_x64():
        source = LayerCheckpoint(source_spec(), [0.2, -0.3], "float64")
        for checkpoint in (source.freeze(["alpha"]), source.freeze()):
            checkpoint = LayerCheckpoint.from_json(checkpoint.to_json())
            layer, weights = QuantumLayer.from_checkpoint(checkpoint)
            module, variables = FlaxLayer.from_checkpoint(checkpoint)
            inputs = jnp.array([[0.2], [0.4]], dtype=jnp.float64)
            expected = np.asarray(
                [
                    expectations(
                        source.spec.circuit,
                        source.spec.observables,
                        source.bindings | {"feature": float(x)},
                    )
                    for x in (0.2, 0.4)
                ]
            )
            np.testing.assert_allclose(jax.jit(layer)(weights, inputs), expected, atol=1e-12)
            np.testing.assert_allclose(
                jax.jit(module.apply)(variables, inputs), expected, atol=1e-12
            )
            params = {
                "quantum": variables["params"],
                "encoder": jnp.array([[0.5]], dtype=jnp.float64),
            }
            optimizer = optax.adam(0.02)

            def loss(p, module=module, inputs=inputs):
                return jnp.square(
                    module.apply({"params": p["quantum"]}, inputs @ p["encoder"])
                ).mean()

            gradients = jax.jit(jax.grad(loss))(params)
            updates, _ = optimizer.update(gradients, optimizer.init(params), params)
            updated = optax.apply_updates(params, updates)
            assert float(loss(updated)) < float(loss(params))
            assert gradients["quantum"]["weights"].shape == weights.shape
            assert module.to_spec() == checkpoint.spec
