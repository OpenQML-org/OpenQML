"""Reusable blocks preserve parameter ownership, trained values and circuit semantics."""

import numpy as np
import pytest

from openqml import (
    Circuit,
    ClassicalCondition,
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
    statevector,
)


def blocks():
    x, w, v = Parameter("x"), Parameter("w"), Parameter("v")
    left = LayerSpec(Circuit(1).ry(x + w, 0), ("x",), ("w",))
    right = LayerSpec(Circuit(1).ry(x + v, 0), ("x",), ("v",))
    return left, right


def test_independent_namespaces_preserve_order_phase_readouts_and_sources():
    left, right = blocks()
    left = LayerSpec(
        left.circuit.with_global_phase(Parameter("x") / 3),
        ("x",),
        ("w",),
        (Hamiltonian((PauliString("Z", 0.7), PauliString("I", -0.2))),),
    )
    right = LayerSpec(right.circuit.with_global_phase(-Parameter("v") / 9), ("x",), ("v",))
    original = left.to_json(), right.to_json()
    result = left.compose(right, prefix="tail")
    assert result.input_parameters == ("x", "tail_x")
    assert result.weight_parameters == ("w", "tail_v")
    assert result.observables == left.observables
    values = {"x": 0.3, "tail_x": 0.4, "w": 0.2, "tail_v": -0.1}
    angle = sum(values.values())
    phase = values["x"] / 3 - values["tail_v"] / 9
    np.testing.assert_allclose(
        statevector(result.circuit, values),
        np.exp(1j * phase) * np.array([np.cos(angle / 2), np.sin(angle / 2)]),
    )
    np.testing.assert_allclose(
        expectations(result.circuit, result.observables, values), [0.7 * np.cos(angle) - 0.2]
    )
    assert (left.to_json(), right.to_json()) == original
    assert LayerSpec.from_json(result.to_json()) == result
    assert left.compose(right, prefix="tail", observables=(PauliString("X"),)).observables == (
        PauliString("X"),
    )


def test_explicit_shared_parameters_accumulate_each_occurrence():
    left, right = blocks()
    result = left.compose(right, shared_inputs={"x": "x"}, shared_weights={"v": "w"})
    assert result.input_parameters == ("x",) and result.weight_parameters == ("w",)
    rows = [{"x": 0.3, "w": 0.2}, {"x": -0.4, "w": 0.6}]
    jacobian = parameter_shift_jacobian(
        result.circuit, result.observables, rows, parameters=("x", "w")
    )
    for row, derivatives in zip(rows, jacobian, strict=True):
        expected = -2 * np.sin(2 * (row["x"] + row["w"]))
        np.testing.assert_allclose(derivatives, [[expected, expected]], atol=1e-12)


def test_reusing_a_namespace_requires_explicit_sharing():
    left, right = blocks()
    first = left.compose(right)
    with pytest.raises(ValidationError, match="namespace collision"):
        first.compose(right)
    reused = first.compose(right, shared_inputs={"x": "block_x"}, shared_weights={"v": "block_v"})
    values = {"x": 0.2, "w": 0.3, "block_x": 0.4, "block_v": -0.1}
    np.testing.assert_allclose(
        expectations(reused.circuit, reused.observables, values),
        [np.cos(values["x"] + values["w"] + 2 * (values["block_x"] + values["block_v"]))],
    )
    assert reused.input_parameters == first.input_parameters
    assert reused.weight_parameters == first.weight_parameters


def test_nested_feedback_wire_mapping_matches_physical_density_evolution():
    left = LayerSpec(
        Circuit(3).ry(Parameter("x"), 0).measure(0, 2), ("x",), (), (PauliString("ZII"),)
    )
    right = LayerSpec(
        Circuit(2).if_else(
            ClassicalCondition((0,), 1), Circuit(2).rx(Parameter("w"), 1).with_global_phase(0.2)
        ),
        (),
        ("w",),
    )
    result = left.compose(right, prefix="feedback", qubits=(2, 0), clbits=(2,))
    expected = (
        Circuit(3)
        .ry(0.7, 0)
        .measure(0, 2)
        .if_else(ClassicalCondition((2,), 1), Circuit(3).rx(0.4, 0).with_global_phase(0.2))
    )
    assert result.circuit.num_clbits == 3 and result.observables == left.observables
    np.testing.assert_allclose(
        density_matrix(result.circuit, {"x": 0.7, "feedback_w": 0.4}),
        density_matrix(expected),
        atol=1e-12,
    )
    exported = convert(result.circuit, target="openqasm3")
    restored = convert(
        exported.value,
        source="openqasm3",
        target="ir",
        source_options={"parser": "builtin"},
    ).value
    np.testing.assert_allclose(
        density_matrix(restored, {exported.parameter_map.get("x", "x"): 0.7, "feedback_w": 0.4}),
        density_matrix(expected),
        atol=1e-12,
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_checkpoint_composition_preserves_trained_values_by_source_order(dtype):
    a, z, b, v = (Parameter(n) for n in ("alpha", "zeta", "b", "a"))
    left = LayerCheckpoint(
        LayerSpec(Circuit(1).ry(z, 0).rx(a, 0), (), ("zeta", "alpha")), [0.2, -0.4], dtype
    )
    right = LayerCheckpoint(
        LayerSpec(Circuit(1).rz(b, 0).ry(v, 0), (), ("b", "a")), [0.3, 0.7], dtype
    )
    source = left.to_json(), right.to_json()
    result = left.compose(right, prefix="tail")
    assert result.spec.weight_parameters == ("zeta", "alpha", "tail_b", "tail_a")
    assert result.weights == left.weights + right.weights
    z, a = left.weights
    b, v = right.weights

    def ry(angle):
        return np.array(
            [[np.cos(angle / 2), -np.sin(angle / 2)], [np.sin(angle / 2), np.cos(angle / 2)]]
        )

    rx = np.array([[np.cos(a / 2), -1j * np.sin(a / 2)], [-1j * np.sin(a / 2), np.cos(a / 2)]])
    rz = np.diag([np.exp(-0.5j * b), np.exp(0.5j * b)])
    expected = ry(v) @ rz @ rx @ ry(z) @ np.array([1, 0])
    np.testing.assert_allclose(
        statevector(result.spec.circuit, result.bindings), expected, atol=1e-12
    )
    assert LayerCheckpoint.from_json(result.to_json()) == result
    assert (left.to_json(), right.to_json()) == source


def test_checkpoint_sharing_requires_equal_values_and_precision():
    left, right = blocks()
    first, second = LayerCheckpoint(left, [0.2]), LayerCheckpoint(right, [0.2])
    sharing = {"shared_inputs": {"x": "x"}, "shared_weights": {"v": "w"}}
    assert first.compose(second, **sharing).bindings == {"w": 0.2}
    with pytest.raises(ValidationError, match="conflicting values"):
        first.compose(LayerCheckpoint(right, [0.3]), **sharing)
    with pytest.raises(ValidationError, match="precision must match"):
        first.compose(LayerCheckpoint(right, [0.2], "float32"))
    with pytest.raises(ValidationError, match="LayerCheckpoint"):
        first.compose(right)


def test_phase_cancellation_drops_inactive_symbols_from_portable_interface():
    left = LayerSpec(Circuit(1).with_global_phase(Parameter("p")), (), ("p",))
    right = LayerSpec(Circuit(1).with_global_phase(-Parameter("q")), (), ("q",))
    result = LayerCheckpoint(left, [0.3]).compose(
        LayerCheckpoint(right, [0.3]), shared_weights={"q": "p"}
    )
    assert result.spec.weight_parameters == result.weights == ()
    assert result.spec.circuit.parameters == ()
    np.testing.assert_allclose(statevector(result.spec.circuit), [1, 0])


@pytest.mark.parametrize(
    "options",
    [
        {"shared_inputs": {"missing": "x"}},
        {"shared_inputs": {"x": "w"}},
        {"shared_weights": {"v": "x"}},
        {"shared_inputs": []},
        {"shared_weights": "v"},
        {"shared_weights": {"v": []}},
        {"prefix": ""},
        {"prefix": None},
        {"qubits": [0, 1]},
        {"qubits": [True]},
        {"qubits": 0},
        {"qubits": {0}},
        {"qubits": "0"},
        {"clbits": {}},
        {"observables": [PauliString("ZZ")]},
    ],
)
def test_invalid_composition_contracts_are_rejected(options):
    left, right = blocks()
    with pytest.raises(ValidationError):
        left.compose(right, **options)


def test_composition_requires_portable_specifications():
    with pytest.raises(ValidationError, match="LayerSpec"):
        blocks()[0].compose(Circuit(1))


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "pennylane", "openqasm3"])
def test_native_blocks_compose_through_the_shared_ir(framework):
    if framework != "openqasm3":
        pytest.importorskip(framework)
    left = LayerSpec.from_program(
        "OPENQASM 3.0; input float feature; input float weight; qubit q; ry(feature+weight) q;",
        ["feature"],
        framework="openqasm3",
        parser="builtin",
    )
    source = (
        Circuit(1).rx(0.4 if framework == "pennylane" else Parameter("v"), 0).with_global_phase(0.2)
    )
    native = convert(source, target=framework).value
    right = LayerSpec.from_program(native, (), framework=framework)
    result = left.compose(right, prefix="tail")
    bindings = {"feature": 0.3, "weight": 0.2} | (
        {} if framework == "pennylane" else {"tail_v": 0.4}
    )
    np.testing.assert_allclose(
        expectations(result.circuit, result.observables, bindings),
        [np.cos(0.4) * np.cos(0.5)],
        atol=1e-12,
    )
    c, s, a, b = np.cos(0.5 / 2), np.sin(0.5 / 2), np.cos(0.4 / 2), np.sin(0.4 / 2)
    np.testing.assert_allclose(
        statevector(result.circuit, bindings),
        np.exp(0.2j) * np.array([a * c - 1j * b * s, -1j * b * c + a * s]),
        atol=1e-12,
    )


def test_qiskit_and_cirq_blocks_share_inputs_without_aliasing_weights():
    pytest.importorskip("qiskit")
    pytest.importorskip("cirq")
    left, right = blocks()
    left = LayerSpec.from_program(convert(left.circuit, target="qiskit").value, ["x"])
    right = LayerSpec.from_program(convert(right.circuit, target="cirq").value, ["x"])
    result = left.compose(right, prefix="ansatz", shared_inputs={"x": "x"})
    assert result.weight_parameters == ("w", "ansatz_v")
    values = {"x": 0.3, "w": 0.2, "ansatz_v": -0.4}
    np.testing.assert_allclose(
        expectations(result.circuit, result.observables, values), [np.cos(0.4)]
    )


@pytest.mark.parametrize("interface", ["torch", "jax"])
@pytest.mark.parametrize("backend", ["native", "numpy"])
def test_composed_checkpoint_joint_gradients_hessians_and_chunking(interface, backend):
    pytest.importorskip(interface)
    left, right = blocks()
    checkpoint = LayerCheckpoint(left, [0.2], "float32").compose(
        LayerCheckpoint(right, [0.2], "float32"),
        shared_inputs={"x": "x"},
        shared_weights={"v": "w"},
    )
    options = {"backend": interface if backend == "native" else backend, "batch_chunk_size": 2}
    if backend == "native":
        options["checkpoint_interval"] = 2
    if interface == "torch":
        import torch

        from openqml.interfaces.torch import QuantumLayer

        layer = QuantumLayer.from_checkpoint(checkpoint, **options)
        x = torch.linspace(0.1, 0.7, 6, dtype=layer.weights.dtype).reshape(2, 3, 1).requires_grad_()
        actual, expected = layer(x), torch.cos(2 * (x + layer.weights))
        torch.testing.assert_close(actual, expected)
        for a, b in zip(
            torch.autograd.grad(actual.square().sum(), (x, layer.weights)),
            torch.autograd.grad(expected.square().sum(), (x, layer.weights)),
            strict=True,
        ):
            torch.testing.assert_close(a, b)
        if backend == "native":
            first = torch.autograd.grad(layer(x).sum(), layer.weights, create_graph=True)[0]
            second = torch.autograd.grad(first.sum(), layer.weights)[0]
            torch.testing.assert_close(
                second, -4 * torch.cos(2 * (x + layer.weights)).sum(0).sum(0)
            )
    else:
        import jax
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        layer, weights = QuantumLayer.from_checkpoint(checkpoint, **options)
        x = jnp.linspace(0.1, 0.7, 6, dtype=weights.dtype).reshape(2, 3, 1)
        expected = jnp.cos(2 * (x + weights))
        np.testing.assert_allclose(jax.jit(layer)(weights, x), expected, atol=3e-7)
        actual_grad = jax.jit(jax.grad(lambda w, a: jnp.square(layer(w, a)).sum(), argnums=(0, 1)))(
            weights, x
        )
        expected_grad = jax.grad(
            lambda w, a: jnp.square(jnp.cos(2 * (a + w))).sum(), argnums=(0, 1)
        )(weights, x)
        for a, b in zip(actual_grad, expected_grad, strict=True):
            np.testing.assert_allclose(a, b, atol=4e-6)
        _, tangent = jax.jit(layer.jvp)(
            weights, x, jnp.full_like(weights, 0.3), jnp.full_like(x, -0.1)
        )
        np.testing.assert_allclose(tangent, -0.4 * jnp.sin(2 * (x + weights)), atol=3e-7)
        if backend == "native":
            hessian = jax.jit(jax.hessian(lambda w: layer(w, x).sum()))(weights)
            np.testing.assert_allclose(
                hessian, [[-4 * jnp.cos(2 * (x + weights)).sum()]], atol=4e-6
            )


def test_composed_torch_layer_shares_a_classical_optimizer():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    left, right = blocks()
    spec = left.compose(right, shared_inputs={"x": "x"}, shared_weights={"v": "w"})
    torch.manual_seed(3)
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), QuantumLayer.from_spec(spec))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data = torch.ones(5, 2)
    before = [p.detach().clone() for p in model.parameters()]
    loss = model(data).square().mean()
    loss.backward()
    optimizer.step()
    assert float(model(data).square().mean().detach()) < float(loss.detach())
    assert all(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


def test_composed_flax_checkpoint_and_classical_parameters_share_one_tree():
    jax = pytest.importorskip("jax")
    flax = pytest.importorskip("flax")
    import flax.linen as nn
    import jax.numpy as jnp
    import optax

    from openqml.interfaces.flax import QuantumLayer

    left, right = blocks()
    checkpoint = LayerCheckpoint(left, [0.2], "float32").compose(
        LayerCheckpoint(right, [0.2], "float32"),
        shared_inputs={"x": "x"},
        shared_weights={"v": "w"},
    )
    module, restored = QuantumLayer.from_checkpoint(checkpoint, checkpoint_interval=2)
    x = jnp.array([[0.3], [0.7]], dtype=module.dtype)
    np.testing.assert_allclose(
        module.apply(restored, x), jnp.cos(2 * (x + restored["params"]["weights"])), atol=3e-7
    )

    class Model(nn.Module):
        @nn.compact
        def __call__(self, data):
            return QuantumLayer.from_spec(checkpoint.spec, name="quantum", checkpoint_interval=2)(
                nn.Dense(1, name="encoder")(data)
            )

    model, data = Model(), jnp.ones((5, 2))
    params = flax.core.unfreeze(model.init(jax.random.key(3), data)["params"])
    params["quantum"] = restored["params"]

    def loss(p):
        return jnp.square(model.apply({"params": p}, data)).mean()

    gradients = jax.jit(jax.grad(loss))(params)
    optimizer = optax.sgd(0.01)
    updates, _ = optimizer.update(gradients, optimizer.init(params), params)
    updated = optax.apply_updates(params, updates)
    assert float(loss(updated)) < float(loss(params))
    assert all(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(updated), strict=True
        )
    )
