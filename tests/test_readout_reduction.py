"""Readout cones preserve declared values, gradients and source provenance."""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openqml import (
    Circuit,
    ClassicalCondition,
    ConversionError,
    Hamiltonian,
    LayerSpec,
    Operation,
    Parameter,
    PauliString,
    ResourceError,
    ValidationError,
    convert,
    expectations,
    parameter_shift_jacobian,
    plan_conversion,
    plan_readout,
)


def sparse_spec():
    x, w = Parameter("x"), Parameter("w")
    circuit = (
        Circuit(5)
        .ry(Parameter("unused_input"), 1)
        .rx(Parameter("unused_weight"), 3)
        .ry(x * w, 4)
        .with_global_phase(Parameter("phase") + x / 3)
    )
    return LayerSpec(
        circuit,
        ("unused_input", "x"),
        ("phase", "unused_weight", "w"),
        (PauliString("IIIIZ"), PauliString("IIIII", 0.3)),
    )


def test_time_order_drops_late_local_gates_on_an_earlier_dependency():
    c = (
        Circuit(5)
        .ry(Parameter("a"), 4)
        .rx(Parameter("b"), 1)
        .cx(4, 1)
        .ry(Parameter("dead"), 4)
        .barrier(0, 1, 2, 3, 4)
        .with_global_phase(Parameter("phase"))
    )
    plan = plan_readout(c, [PauliString("IZIII")])
    assert plan.qubits == (1, 4) and plan.operation_indices == (0, 1, 2)
    assert [op.qubits for op in plan.circuit.operations] == [(1,), (0,), (1, 0)]
    assert plan.removed_parameters == ("dead", "phase")
    assert plan.circuit.global_phase.evaluate({}) == 0
    assert plan.observables == (PauliString("ZI"),)
    assert plan.source_circuit is c and c.num_qubits == 5
    for a, b in ((0.2, -0.7), (-0.6, 0.4)):
        values = dict(a=a, b=b, dead=1.3, phase=0.4)
        np.testing.assert_allclose(
            expectations(c, plan.source_observables, values),
            expectations(plan.circuit, plan.observables, dict(a=a, b=b)),
            atol=2e-14,
        )
    report = json.loads(plan.to_json())
    assert report["semantics"] == "declared_expectations"
    assert report["qubits"] == [1, 4] and report["operation_indices"] == [0, 1, 2]
    assert report["before"]["num_qubits"] == 5 and report["after"]["num_qubits"] == 2
    assert report["diagnostics"][0]["code"] == "DECLARED_EXPECTATIONS_ONLY"
    with pytest.raises(FrozenInstanceError):
        plan.circuit = c


@pytest.mark.parametrize("seed", range(6))
def test_random_unitaries_noncontiguous_wires_and_mixed_paulis(seed):
    rng = np.random.default_rng(seed)
    # The late spectator gates are removed even after earlier entanglement.
    c = Circuit(5)
    for _ in range(3):
        for q in (4, 1, 3):
            c = c.u(*rng.normal(size=3), q)
        c = c.cx(4, 1).rxx(float(rng.normal()), 3, 4)
    matrix, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    c = c.unitary(matrix, 4, 1).ry(0.8, 3).rx(-0.6, 0).ry(0.3, 2)
    obs = (
        PauliString("IXIIY", -0.3),
        Hamiltonian((PauliString("IZIII", 0.7), PauliString("IIIII", 0.2))),
        PauliString("YXYZX", 0),
    )
    plan = plan_readout(c, iter(obs))
    assert plan.qubits == (1, 3, 4)
    np.testing.assert_allclose(
        expectations(c, obs), expectations(plan.circuit, plan.observables), atol=2e-13
    )
    assert plan.observables[2] == PauliString("III", 0)
    assert len(plan.circuit.operations) == len(c.operations) - 3


def test_union_cone_preserves_term_order_coefficients_and_shared_symbol_gradients():
    a, b = Parameter("a"), Parameter("b")
    c = Circuit(5).ry(a * b, 4).rx(a.sin(), 1).cx(4, 1).ry(b**2, 2).with_global_phase(a + b)
    obs = (
        Hamiltonian((PauliString("IZIII", 0.3), PauliString("IIIIZ", -0.4))),
        PauliString("IIXII"),
    )
    plan = plan_readout(c, obs)
    assert plan.qubits == (1, 2, 4)
    assert plan.observables[0] == Hamiltonian((PauliString("ZII", 0.3), PauliString("IIZ", -0.4)))
    rows = [dict(a=0.3, b=-0.2), dict(a=-0.4, b=0.5)]
    np.testing.assert_allclose(
        parameter_shift_jacobian(c, obs, rows),
        parameter_shift_jacobian(plan.circuit, plan.observables, rows),
        atol=3e-13,
    )


def test_identity_and_zero_outputs_keep_one_wire_and_no_parameters():
    spec = sparse_spec()
    obs = (
        PauliString("IIIII", -0.7),
        Hamiltonian((PauliString("XYZYX", 0), PauliString("IIIII", 0.2))),
    )
    plan = plan_readout(spec.circuit, obs)
    assert plan.qubits == (0,) and plan.circuit == Circuit(1)
    assert plan.operation_indices == () and plan.removed_parameters == spec.circuit.parameters
    np.testing.assert_array_equal(expectations(plan.circuit, plan.observables), [-0.7, 0.2])


def test_unchanged_operations_retain_identity_and_classical_storage_is_not_retained():
    c = Circuit(2, (Operation("ry", (0,), (Parameter("w"),)),), num_clbits=8)
    plan = plan_readout(c, [PauliString("ZI")])
    assert plan.circuit.operations[0] is c.operations[0]
    assert plan.circuit.num_clbits == 0 and c.num_clbits == 8


@pytest.mark.parametrize(
    "obs", [None, [], "Z", {PauliString("Z")}, {"x": PauliString("Z")}, [1], [PauliString("ZZ")]]
)
def test_invalid_readouts_fail_explicitly(obs):
    with pytest.raises(ValidationError):
        plan_readout(Circuit(1), obs)


@pytest.mark.parametrize(
    "circuit",
    [
        Circuit(2).measure(1, 0),
        Circuit(2).amplitude_damping(0.2, 1),
        Circuit(2).reset(1),
        Circuit(2).if_else(ClassicalCondition((0,), 1), Circuit(2).x(1)),
    ],
)
def test_effects_are_rejected_even_when_disjoint_from_readouts(circuit):
    with pytest.raises(ConversionError) as error:
        plan_readout(circuit, [PauliString("ZI")])
    assert error.value.diagnostics[0].code == "READOUT_EFFECTS_UNSUPPORTED"


def test_operation_budget_precedes_analysis_and_conversion_remains_full_program():
    c = sparse_spec().circuit
    with pytest.raises(ConversionError) as error:
        plan_readout(c, sparse_spec().observables, max_operations=2)
    assert error.value.diagnostics[0].code == "DECOMPOSITION_LIMIT"
    assert plan_conversion(c, target="ir").export().value == c
    assert convert(c, target="json").value == c.to_json()


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "pennylane", "openqasm3"])
def test_import_reduce_export_and_execute_keep_wire_provenance(framework):
    if framework != "openqasm3":
        pytest.importorskip(framework)
    c = Circuit(5).ry(0.43, 4).cx(4, 1).rx(-0.3, 0).with_global_phase(0.2)
    obs = (PauliString("IZIII"), PauliString("IXIIX"))
    imported = convert(convert(c, target=framework).value, source=framework, target="ir").value
    plan = plan_readout(imported, obs)
    exported = convert(plan.circuit, target=framework).value
    restored = convert(exported, source=framework, target="ir").value
    assert plan.qubits == (1, 4) and restored.num_qubits == 2
    np.testing.assert_allclose(
        expectations(c, obs), expectations(restored, plan.observables), atol=3e-12
    )


@pytest.mark.parametrize(
    "backend,method", [("torch", "auto"), ("numpy", "auto"), ("numpy", "adjoint")]
)
@pytest.mark.parametrize("chunk", [None, 2])
def test_torch_broadcast_training_derivatives_and_source_checkpoint(backend, method, chunk):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    spec = sparse_spec()
    layer = QuantumLayer.from_spec(
        spec,
        reduce_readout=True,
        backend=backend,
        gradient_method=method,
        dtype=torch.float64,
        batch_chunk_size=chunk,
    )
    assert layer.circuit.num_qubits == 1 and layer.readout_plan.qubits == (4,)
    assert layer.to_spec() == spec and layer.source_observables == spec.observables
    x = torch.tensor(
        [[0.8, 0.3], [-0.4, 0.7], [0.2, -0.5]], dtype=torch.float64, requires_grad=True
    )
    w = torch.tensor(
        [[[0.4, -0.2, 0.6]], [[-0.1, 0.5, -0.7]]], dtype=torch.float64, requires_grad=True
    )
    out = layer.functional(w, x)
    reference = torch.cos(w[..., 2:] * x[..., 1:])
    expected = torch.cat((reference, torch.full_like(reference, 0.3)), dim=-1)
    torch.testing.assert_close(out, expected)
    actual_grad = torch.autograd.grad(out.square().sum(), (w, x))
    expected_grad = torch.autograd.grad(expected.square().sum(), (w, x))
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, atol=3e-12, rtol=3e-12)
    with torch.autograd.forward_ad.dual_level():
        dual_w = torch.autograd.forward_ad.make_dual(w.detach(), torch.ones_like(w))
        dual_x = torch.autograd.forward_ad.make_dual(x.detach(), torch.ones_like(x))
        _, tangent = torch.autograd.forward_ad.unpack_dual(layer.functional(dual_w, dual_x))
    expected_direction = -torch.sin(w[..., 2:] * x[..., 1:]) * (w[..., 2:] + x[..., 1:])
    torch.testing.assert_close(tangent[..., :1], expected_direction, atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(tangent[..., 1:], torch.zeros_like(expected_direction))
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    layer(x.detach()).square().sum().backward()
    optimizer.step()
    torch.testing.assert_close(layer.weights.grad[:2], torch.zeros_like(layer.weights.grad[:2]))
    checkpoint = layer.checkpoint()
    assert checkpoint.spec == spec
    for reduced in (False, True):
        restored = QuantumLayer.from_checkpoint(checkpoint, reduce_readout=reduced)
        torch.testing.assert_close(restored(x), layer(x))
    assert layer(torch.empty((0, 2), dtype=torch.float64)).shape == (0, 2)


def test_torch_higher_derivatives_compile_and_checkpoint_recomputation():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    layer = QuantumLayer.from_spec(
        sparse_spec(), reduce_readout=True, dtype=torch.float64, checkpoint_interval=1
    )
    x = torch.tensor([[0.1, 0.3]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([0.5, -0.1, 0.4], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradgradcheck(layer.functional, (w, x))
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), layer(x))


@pytest.mark.parametrize(
    "backend,method", [("jax", "auto"), ("numpy", "auto"), ("numpy", "adjoint")]
)
def test_jax_jit_vmap_directional_products_and_checkpoint(backend, method):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    layer = QuantumLayer.from_spec(
        sparse_spec(),
        reduce_readout=True,
        backend=backend,
        gradient_method=method,
        batch_chunk_size=2,
    )
    x = jnp.array([[0.8, 0.3], [-0.4, 0.7], [0.2, -0.5]], dtype=jnp.float32)
    w = jnp.array([[[0.4, -0.2, 0.6]], [[-0.1, 0.5, -0.7]]], dtype=jnp.float32)

    def reference(a, b):
        value = jnp.cos(a[..., 2:] * b[..., 1:])
        return jnp.concatenate((value, jnp.full_like(value, 0.3)), axis=-1)

    np.testing.assert_allclose(jax.jit(layer)(w, x), reference(w, x), atol=3e-7)
    actual = jax.jit(jax.grad(lambda a, b: jnp.square(layer(a, b)).sum(), argnums=(0, 1)))(w, x)
    expected = jax.grad(lambda a, b: jnp.square(reference(a, b)).sum(), argnums=(0, 1))(w, x)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=5e-7)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(layer, in_axes=(0, None)))(w, x), reference(w, x), atol=3e-7
    )
    _, tangent = jax.jit(layer.jvp)(w, x, jnp.ones_like(w), jnp.ones_like(x))
    _, expected_tangent = jax.jvp(reference, (w, x), (jnp.ones_like(w), jnp.ones_like(x)))
    np.testing.assert_allclose(tangent, expected_tangent, atol=5e-7)
    checkpoint = layer.checkpoint(w[0, 0])
    assert checkpoint.spec == sparse_spec()
    restored, values = QuantumLayer.from_checkpoint(checkpoint)
    np.testing.assert_allclose(restored(values, x), layer(w[0, 0], x), atol=3e-7)
    assert jax.jit(layer)(w[:0], x).shape == (0, 3, 2)


@pytest.mark.parametrize("backend,method", [("jax", "auto"), ("numpy", "adjoint")])
def test_flax_owned_and_generated_weights_keep_one_optimizer_tree(backend, method):
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp
    import optax

    from openqml.interfaces.flax import QuantumLayer

    module = QuantumLayer.from_spec(
        sparse_spec(), reduce_readout=True, backend=backend, gradient_method=method
    )
    x = jnp.array([[0.2, 0.4], [0.3, -0.5]], dtype=jnp.float32)
    variables = module.init(jax.random.key(7), x)
    assert variables["params"]["weights"].shape == (3,)
    gradients = jax.jit(jax.grad(lambda v: jnp.square(module.apply(v, x)).sum()))(variables)
    np.testing.assert_array_equal(gradients["params"]["weights"][:2], [0, 0])
    optimizer = optax.adam(0.01)
    updates, _ = optimizer.update(gradients, optimizer.init(variables), variables)
    trained = optax.apply_updates(variables, updates)
    checkpoint = module.checkpoint(trained["params"]["weights"])
    assert checkpoint.spec == sparse_spec()
    restored, restored_variables = QuantumLayer.from_checkpoint(checkpoint, reduce_readout=True)
    np.testing.assert_allclose(
        restored.apply(restored_variables, x), module.apply(trained, x), atol=3e-7
    )
    generated = jnp.array([0.1, 0.2, 0.6], dtype=x.dtype)
    assert module.init(jax.random.key(8), x, weights=generated) == {}
    gradient = jax.grad(lambda w: module.apply({}, x, weights=w).sum())(generated)
    np.testing.assert_array_equal(gradient[:2], [0, 0])


@pytest.mark.parametrize("interface", ["torch", "jax"])
def test_reduction_runs_before_state_allocation_limits_and_rejects_bad_configuration(interface):
    pytest.importorskip(interface)
    if interface == "torch":
        from openqml.interfaces.torch import QuantumLayer
    else:
        from openqml.interfaces.jax import QuantumLayer
    spec = sparse_spec()
    with pytest.raises(ResourceError):
        QuantumLayer.from_spec(spec, max_qubits=1)
    reduced = QuantumLayer.from_spec(spec, reduce_readout=True, max_qubits=1, max_memory_mb=0.01)
    assert reduced.circuit.num_qubits == 1
    for value in (1, "yes", None):
        with pytest.raises(ValidationError, match="boolean"):
            QuantumLayer.from_spec(spec, reduce_readout=value)
    with pytest.raises(ValidationError, match="requires backend"):
        QuantumLayer.from_spec(spec, reduce_readout=True, backend="qiskit")
    optimized = QuantumLayer.from_spec(spec, reduce_readout=True, optimize=True)
    assert optimized.readout_plan.source_circuit is optimized.optimization.circuit
    assert optimized.to_spec() == spec


def test_identity_only_torch_preserves_zero_gradients_and_second_derivatives():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    spec = sparse_spec()
    layer = QuantumLayer(
        spec.circuit,
        spec.input_parameters,
        spec.weight_parameters,
        [PauliString("IIIII")],
        reduce_readout=True,
        dtype=torch.float64,
    )
    x = torch.full((2, 2), 1e200, dtype=torch.float64, requires_grad=True)
    w = torch.full((3,), 1e200, dtype=torch.float64, requires_grad=True)
    output = layer.functional(w, x)
    torch.testing.assert_close(output, torch.ones_like(output))
    first = torch.autograd.grad(output.sum(), (w, x), create_graph=True)
    second = torch.autograd.grad(sum(v.sum() for v in first), (w, x))
    assert all(torch.count_nonzero(t) == 0 for t in (*first, *second))


def test_jax_hessian_survives_readout_reduction_and_recomputation():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    layer = QuantumLayer.from_spec(sparse_spec(), reduce_readout=True, checkpoint_interval=1)
    x, w = jnp.array([0.2, 0.3], dtype=jnp.float32), jnp.array([0.1, 0.4, 0.5], dtype=jnp.float32)
    actual = jax.jit(jax.hessian(lambda a: layer(a, x).sum()))(w)
    expected = jax.hessian(lambda a: jnp.cos(a[2] * x[1]))(w)
    np.testing.assert_allclose(actual, expected, atol=3e-7)


@pytest.mark.parametrize("interface", ["torch", "jax"])
@pytest.mark.parametrize("method", ["auto", "adjoint"])
def test_constant_external_outputs_scatter_zeros_without_executable_parameters(interface, method):
    spec = sparse_spec()
    options = dict(
        reduce_readout=True,
        backend="numpy",
        gradient_method=method,
        observables=[PauliString("IIIII", 0.4)],
    )
    if interface == "torch":
        torch = pytest.importorskip("torch")
        from openqml.interfaces.torch import QuantumLayer

        layer = QuantumLayer(
            spec.circuit,
            spec.input_parameters,
            spec.weight_parameters,
            dtype=torch.float64,
            **options,
        )
        x = torch.ones((3, 2), dtype=torch.float64, requires_grad=True)
        w = torch.ones((3,), dtype=torch.float64, requires_grad=True)
        output = layer.functional(w, x)
        torch.testing.assert_close(output, torch.full_like(output, 0.4))
        gradients = torch.autograd.grad(output.sum(), (w, x))
        assert all(torch.count_nonzero(t) == 0 for t in gradients)
    else:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        layer = QuantumLayer(spec.circuit, spec.input_parameters, spec.weight_parameters, **options)
        x, w = jnp.ones((3, 2), dtype=jnp.float32), jnp.ones((3,), dtype=jnp.float32)
        gradients = jax.jit(jax.grad(lambda a, b: layer(a, b).sum(), argnums=(0, 1)))(w, x)
        assert all(jnp.count_nonzero(t) == 0 for t in gradients)
        primal, tangent = jax.jit(layer.jvp)(w, x, w, x)
        np.testing.assert_allclose(primal, 0.4, atol=3e-8)
        np.testing.assert_array_equal(tangent, np.zeros((3, 1)))
    assert layer.circuit.parameters == ()
