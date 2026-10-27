"""Exact portfolio selection, factor reuse, provenance and native hybrid training."""

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openqml import (
    Circuit,
    ClassicalCondition,
    ConversionError,
    ExpectationPlan,
    Expression,
    Hamiltonian,
    LayerSpec,
    Parameter,
    PauliPlan,
    PauliString,
    ResourceError,
    ValidationError,
    convert,
    expectations,
)


def mixed_spec(width=6):
    x, w = Parameter("x"), Parameter("w")
    circuit = Circuit(width).ry(x * w, 0).cx(0, 2).ry(w.sin(), 3).rx(x, 4)
    matrix = np.kron(np.array([[1, 1j], [1j, 1]]) / np.sqrt(2), np.eye(2))
    circuit = circuit.unitary(matrix, 4, 3).rx(Parameter("unused"), 1)
    circuit = circuit.with_global_phase(Parameter("phase"))
    z = "Z" + "I" * (width - 1)
    combined = "ZIIXX" + "I" * (width - 5)
    return LayerSpec(
        circuit,
        ("unused", "x"),
        ("phase", "w"),
        (
            PauliString(z),
            PauliString(combined, -0.3),
            Hamiltonian(
                (PauliString(combined, 0.4), PauliString(z, 0.7), PauliString("I" * width, 0.2))
            ),
        ),
    )


def test_mixed_engines_share_factors_and_keep_source_provenance():
    spec = mixed_spec()
    plan = ExpectationPlan(spec.circuit, spec.observables)
    assert {g.method for g in plan.groups} == {"pauli", "statevector"}
    assert {g.source_qubits for g in plan.groups} == {(0, 2), (3, 4)}
    # Conservative cones keep the CNOT's ancestor wire. Repeated Hamiltonian
    # terms reuse the same local quantum readouts and final product.
    assert plan.factor_count == 2 and len(plan.groups) == 2
    assert plan.active_parameters == ("w", "x")
    assert plan.circuit is spec.circuit
    for group in plan.groups:
        for op, index in zip(group.circuit.operations, group.source_operations, strict=True):
            original = spec.circuit.operations[index]
            assert tuple(group.source_qubits[q] for q in op.qubits) == original.qubits
        valid = [v for v in (group.pauli_work, group.dense_work) if v is not None]
        assert group.estimated_work == min(valid)
    values = dict(x=np.array([0.2, -0.4]), w=0.6, phase=0.3, unused=0.8)
    np.testing.assert_allclose(
        plan.evaluate(values), expectations(spec.circuit, spec.observables, values), atol=5e-13
    )
    report = json.loads(plan.to_json())
    assert report["cost_model"] == "static_scalar_work"
    assert report["removed_parameters"] == ["phase", "unused"]
    assert report["diagnostics"][0]["code"] == "DECLARED_EXPECTATIONS_ONLY"
    dense = next(g for g in plan.groups if g.method == "statevector")
    assert any(d.code == "DECOMPOSITION_UNAVAILABLE" for d in dense.diagnostics)
    with pytest.raises(FrozenInstanceError):
        plan.groups = ()
    assert convert(plan.circuit, target="ir").value is spec.circuit


def test_pauli_frontier_exhaustion_falls_back_to_prepared_dense_execution():
    c = Circuit(2).ry(Parameter("x"), 0).cx(0, 1).rx(Parameter("w"), 1)
    obs = (PauliString("ZZ"), PauliString("XX"))
    with pytest.raises(ConversionError, match="PAULI_TERM_LIMIT"):
        PauliPlan(c, obs, max_terms=1)
    plan = ExpectationPlan(c, obs, max_terms=1)
    assert all(g.method == "statevector" for g in plan.groups)
    assert any(d.code == "PAULI_TERM_LIMIT" for g in plan.groups for d in g.diagnostics)
    np.testing.assert_allclose(
        plan.evaluate(dict(x=0.4, w=0.3)), expectations(c, obs, dict(x=0.4, w=0.3)), atol=2e-14
    )


@pytest.mark.parametrize("seed", range(8))
def test_random_product_circuits_and_mixed_hamiltonians_match_unfactored_state(seed):
    rng = np.random.default_rng(seed)
    c = Circuit(6)
    for _ in range(4):
        for wires in ((4, 1), (5, 0), (3, 2)):
            a = Parameter("x") * float(rng.normal()) + Parameter("w").sin()
            c = c.ry(a, wires[0]).crx(a / 3, *wires).rx(0.2, wires[1])
    obs = tuple(
        PauliString("".join(rng.choice(list("IXYZ"), 6)), float(rng.normal())) for _ in range(5)
    )
    obs += (Hamiltonian((obs[0], obs[2], PauliString("IIIIII", 0.3))),)
    plan = ExpectationPlan(c, obs)
    for x, w in ((0.3, -0.6), (-0.4, 0.7)):
        np.testing.assert_allclose(
            plan.evaluate(dict(x=x, w=w)), expectations(c, obs, dict(x=x, w=w)), atol=2e-12
        )


def test_time_order_keeps_entangled_ancestors_and_discards_late_unobserved_gates():
    c = Circuit(4).ry(Parameter("x"), 3).cx(3, 1).ry(Parameter("late"), 3).rx(Parameter("dead"), 0)
    plan = ExpectationPlan(c, [PauliString("IZII")])
    assert len(plan.groups) == 1 and plan.groups[0].source_qubits == (1, 3)
    assert plan.groups[0].source_operations == (0, 1)
    assert plan.active_parameters == ("x",)
    np.testing.assert_allclose(plan.evaluate(dict(x=0.7)), [np.cos(0.7)], atol=2e-14)


def test_idle_zero_and_identity_readouts_need_no_execution_groups():
    c = Circuit(4).ry(Parameter("x"), 1).with_global_phase(Parameter("phase"))
    obs = (
        PauliString("XIII"),
        PauliString("ZIII", -0.3),
        Hamiltonian((PauliString("IYII", 0), PauliString("IIII", 0.2))),
    )
    plan = ExpectationPlan(c, obs)
    assert plan.groups == () and plan.active_parameters == ()
    np.testing.assert_allclose(
        plan.evaluate(batch_shape=(2, 3)), np.broadcast_to([0, -0.3, 0.2], (2, 3, 3))
    )


def test_cross_group_parameter_arithmetic_is_interned_and_never_recompiled(monkeypatch):
    c = Circuit(6)
    for q in range(0, 6, 2):
        c = c.ry(Parameter("x") * Parameter("w"), q).unitary(np.eye(4), q, q + 1)
    plan = ExpectationPlan(c, [PauliString("XIXIXI")])
    assert len(plan.groups) == 3
    assert {g.program_indices for g in plan.groups} == {(0,)}
    assert plan.to_dict()["shared_coefficient_nodes"] == 3
    import openqml.execution as compiler

    monkeypatch.setattr(compiler, "_compile_dense", lambda *a: pytest.fail("Repeated preparation"))
    original = Expression.evaluate

    def guarded(self, bindings):
        if self.op is not None:
            pytest.fail("Runtime interpretation of source parameter graphs")
        return original(self, bindings)

    monkeypatch.setattr(Expression, "evaluate", guarded)
    for w in (0.3, -0.6):
        np.testing.assert_allclose(
            plan.evaluate(dict(x=0.4, w=w)), [np.sin(0.4 * w) ** 3], atol=3e-14
        )


def test_common_parameter_nodes_are_shared_across_different_execution_methods():
    angle = Parameter("x") * Parameter("w")
    c = Circuit(5).ry(angle, 0).cx(0, 1).cx(1, 2).ry(angle, 3).unitary(np.eye(4), 3, 4)
    plan = ExpectationPlan(c, [PauliString("IIZZI")])
    assert {g.method for g in plan.groups} == {"pauli", "statevector"}
    assert plan.to_dict()["shared_coefficient_nodes"] == 4
    x = np.array([-0.3, 0.2, 0.7])
    np.testing.assert_allclose(
        plan.evaluate(dict(x=x, w=0.4))[:, 0], np.cos(x * 0.4) ** 2, atol=3e-14
    )


def test_full_reference_depth_is_bounded_before_recursive_source_execution():
    angle = Expression(terms=(("x", 1),))
    for _ in range(140):
        angle = Expression(op="sin", args=(angle,))
    plan = ExpectationPlan(Circuit(1).ry(angle, 0), [PauliString("Z")])
    with pytest.raises(ResourceError, match="nesting ceiling"):
        plan.verify(dict(x=0.2))


def test_large_opaque_components_allocate_only_local_states(monkeypatch):
    import openqml.simulator as simulator

    width, c = 128, Circuit(128)
    for q in range(0, width, 2):
        c = c.ry(Parameter("x") * Parameter("w"), q).unitary(np.eye(4), q, q + 1)
    plan = ExpectationPlan(c, [PauliString("Z" * width)], max_dense_qubits=2, max_memory_mb=0.03)
    assert len(plan.groups) == 64 and all(g.method == "statevector" for g in plan.groups)
    assert plan.planning_work < 100_000
    apply, sizes = simulator._apply_matrix, []

    def checked(state, *args, **kwargs):
        sizes.append(state.shape[-1])
        return apply(state, *args, **kwargs)

    monkeypatch.setattr(simulator, "_apply_matrix", checked)
    x = np.array([0.1, 0.2, 0.3])
    np.testing.assert_allclose(
        plan.evaluate(dict(x=x, w=0.4))[:, 0], np.cos(x * 0.4) ** 64, atol=3e-13
    )
    assert sizes and max(sizes) == 4


def test_dense_parameter_dag_handles_aliasing_and_verification_bounds_the_reference():
    angle = Parameter("x").sin()
    for _ in range(72):
        angle = Expression(op="add", args=(angle, angle))
    c = Circuit(2).ry(angle, 0).unitary(np.eye(4), 0, 1)
    plan = ExpectationPlan(c, [PauliString("ZI")])
    assert plan.groups[0].method == "statevector"
    np.testing.assert_allclose(
        plan.evaluate(dict(x=0.0001)), [np.cos(2.0**72 * np.sin(0.0001))], atol=2e-14
    )
    with pytest.raises(ResourceError, match="max_reference_work"):
        plan.verify(dict(x=0.0001))


@pytest.mark.parametrize(
    "options,code",
    [
        ({"max_groups": 1}, "EXPECTATION_GROUP_LIMIT"),
        ({"max_planning_work": 1}, "EXPECTATION_PLANNING_LIMIT"),
        ({"max_operations": 1}, "DECOMPOSITION_LIMIT"),
        ({"max_dense_qubits": 0}, "NO_EXACT_EXECUTION"),
        ({"max_memory_mb": 0.00001}, "EXPECTATION_WORKING_LIMIT"),
    ],
)
def test_plan_limits_fail_with_structured_diagnostics(options, code):
    c = (
        Circuit(4)
        .ry(Parameter("x"), 0)
        .unitary(np.eye(4), 0, 1)
        .ry(Parameter("x"), 2)
        .unitary(np.eye(4), 2, 3)
    )
    with pytest.raises(ConversionError) as error:
        ExpectationPlan(c, [PauliString("XIXI")], **options)
    assert error.value.diagnostics[0].code == code


@pytest.mark.parametrize(
    "options",
    [
        {"max_groups": 0},
        {"max_nodes": 1},
        {"max_work": True},
        {"max_dense_qubits": -1},
        {"max_planning_work": 0},
        {"max_memory_mb": 0},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValidationError):
        ExpectationPlan(Circuit(1), [PauliString("Z")], **options)


@pytest.mark.parametrize("obs", [None, [], "Z", {PauliString("Z")}, [1], [PauliString("ZZ")]])
def test_invalid_readouts(obs):
    with pytest.raises(ValidationError):
        ExpectationPlan(Circuit(1), obs)


@pytest.mark.parametrize(
    "circuit",
    [
        Circuit(1).measure(0, 0),
        Circuit(1).reset(0),
        Circuit(1).amplitude_damping(0.2, 0),
        Circuit(1).if_else(ClassicalCondition((0,), 1), Circuit(1).x(0)),
    ],
)
def test_effects_are_not_implicitly_removed(circuit):
    with pytest.raises(ConversionError, match="EXPECTATION_EFFECTS_UNSUPPORTED"):
        ExpectationPlan(circuit, [PauliString("I")])


def test_broadcast_empty_and_working_batch_limits():
    spec = mixed_spec()
    plan = ExpectationPlan(spec.circuit, spec.observables, max_memory_mb=0.02)
    x = np.array([[0.2], [0.4]])
    w = np.array([0.3, -0.5, 0.7])
    expected = expectations(spec.circuit, spec.observables, dict(x=x, w=w, phase=0.2, unused=-0.4))
    np.testing.assert_allclose(plan.evaluate(dict(x=x, w=w)), expected, atol=3e-13)
    assert plan.evaluate(dict(x=np.empty((0, 1)), w=w)).shape == (0, 3, 3)
    with pytest.raises(ResourceError):
        plan.evaluate(dict(x=np.ones(100), w=0.3))


@pytest.mark.parametrize("backend", ["numpy", "qiskit", "cirq", "pennylane"])
def test_numerical_verification_against_independent_unfactored_engines(backend):
    if backend != "numpy":
        pytest.importorskip(backend)
    spec = mixed_spec()
    plan = ExpectationPlan(spec.circuit, spec.observables)
    report = plan.verify(dict(x=0.4, w=0.7, phase=0.3, unused=-0.2), reference_backend=backend)
    assert report.matched and report.max_absolute_error < 3e-12
    assert report.mismatched_outputs == ()
    assert json.loads(report.to_json())["scope"] == "declared_expectations_at_bindings"


def test_verification_reports_mismatch_and_rejects_unbounded_references(monkeypatch):
    plan = ExpectationPlan(Circuit(1).ry(Parameter("x"), 0), [PauliString("Z")])
    monkeypatch.setattr(ExpectationPlan, "evaluate", lambda *a, **k: np.array([0.0]))
    report = plan.verify(dict(x=0.2))
    assert not report.matched and report.mismatched_outputs == (0,)
    with pytest.raises(ResourceError):
        plan.verify(dict(x=0.2), max_reference_work=1)
    with pytest.raises(ValidationError):
        plan.verify({}, atol=-1)
    with pytest.raises(ValidationError):
        plan.verify(dict(x=[0.2]))
    many = ExpectationPlan(Circuit(1), [Hamiltonian((PauliString("Z"),) * 1000)])
    with pytest.raises(ResourceError, match="max_reference_work"):
        many.verify(max_reference_work=1000)


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "pennylane", "openqasm3"])
def test_shared_ir_import_plan_and_full_program_roundtrip(framework):
    if framework != "openqasm3":
        pytest.importorskip(framework)
    c = Circuit(4).ry(0.4, 3).cx(3, 1).rx(-0.2, 0).with_global_phase(0.3)
    native = convert(c, target=framework).value
    source = convert(native, source=framework, target="ir").value
    plan = ExpectationPlan(source, [PauliString("XZII"), PauliString("IZII")])
    assert plan.verify().matched
    restored = convert(
        convert(plan.circuit, target=framework).value, source=framework, target="ir"
    ).value
    from openqml import statevector

    np.testing.assert_allclose(statevector(restored), statevector(c), atol=3e-13)


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "openqasm3"])
def test_symbolic_imports_train_through_the_same_portfolio(framework):
    torch = pytest.importorskip("torch")
    if framework != "openqasm3":
        pytest.importorskip(framework)
    from openqml.interfaces.torch import QuantumLayer

    feature, weight = Parameter("feature"), Parameter("weight")
    c = Circuit(4).ry(feature * weight, 0).cx(0, 1).ry(weight, 2).rx(feature, 3)
    c = c.with_global_phase(Parameter("source_phase"))
    result = convert(c, target=framework)
    layer = QuantumLayer.from_program(
        result.value,
        ("feature",),
        ("source_phase", "weight"),
        framework=framework,
        execution_method="planned",
        observables=(PauliString("ZIII"), PauliString("IIZI")),
        dtype=torch.float64,
    )
    x = torch.tensor([[0.3], [0.6]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([0.7, 0.4], dtype=torch.float64, requires_grad=True)
    value = layer.functional(w, x)
    expected = torch.cat((torch.cos(x * w[1]), torch.cos(w[1]).expand_as(x)), dim=-1)
    torch.testing.assert_close(value, expected, atol=3e-12, rtol=3e-12)
    got = torch.autograd.grad(value.sum(), (w, x))
    reference = torch.autograd.grad(expected.sum(), (w, x))
    for a, b in zip(got, reference, strict=True):
        torch.testing.assert_close(a, b, atol=4e-12, rtol=4e-12)


@pytest.mark.parametrize("chunk", [None, 2])
def test_torch_native_training_jvp_hessian_and_checkpoint_ownership(chunk):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    spec = mixed_spec()
    layer = QuantumLayer.from_spec(
        spec, execution_method="planned", dtype=torch.float64, batch_chunk_size=chunk
    )
    dense = QuantumLayer.from_spec(spec, dtype=torch.float64)
    assert layer.pauli_plan is None and layer.execution_plan is not None
    x = torch.tensor(
        [[0.2, 0.4], [-0.1, 0.7], [0.5, -0.3]], dtype=torch.float64, requires_grad=True
    )
    w = torch.tensor([[[0.3, 0.5]], [[-0.2, 0.6]]], dtype=torch.float64, requires_grad=True)
    actual, expected = layer.functional(w, x), dense.functional(w, x)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    first = torch.autograd.grad(actual.square().sum(), (w, x), create_graph=True)
    reference = torch.autograd.grad(expected.square().sum(), (w, x), create_graph=True)
    for a, b in zip(first, reference, strict=True):
        torch.testing.assert_close(a, b, atol=5e-12, rtol=5e-12)
    for a, b in zip(
        torch.autograd.grad(sum(t.sum() for t in first), (w, x)),
        torch.autograd.grad(sum(t.sum() for t in reference), (w, x)),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=8e-12, rtol=8e-12)
    _, tangent = torch.func.jvp(layer.functional, (w, x), (torch.ones_like(w), torch.ones_like(x)))
    _, ref = torch.func.jvp(dense.functional, (w, x), (torch.ones_like(w), torch.ones_like(x)))
    torch.testing.assert_close(tangent, ref, atol=5e-12, rtol=5e-12)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    layer(x.detach()).square().sum().backward()
    optimizer.step()
    assert layer.weights.grad[0] == 0
    checkpoint = layer.checkpoint()
    assert checkpoint.spec == spec
    for method in ("auto", "planned"):
        restored = QuantumLayer.from_checkpoint(checkpoint, execution_method=method)
        torch.testing.assert_close(restored(x), layer(x), atol=5e-12, rtol=5e-12)
    assert layer(torch.empty((0, 2), dtype=torch.float64)).shape == (0, 3)


def test_zero_factors_retain_mixed_second_derivatives_and_fullgraph_compile():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    c = (
        Circuit(4)
        .ry(Parameter("x"), 0)
        .unitary(np.eye(4), 0, 1)
        .ry(Parameter("w"), 2)
        .unitary(np.eye(4), 2, 3)
    )
    layer = QuantumLayer(
        c, ("x",), ("w",), [PauliString("XIXI")], execution_method="planned", dtype=torch.float64
    )
    x = torch.zeros((1,), dtype=torch.float64, requires_grad=True)
    w = torch.zeros((1,), dtype=torch.float64, requires_grad=True)
    out = layer.functional(w, x)
    dx = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
    cross = torch.autograd.grad(dx.sum(), w)[0]
    torch.testing.assert_close(cross, torch.ones_like(w))
    compiled = torch.compile(layer.functional, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(w, x), out)
    assert torch.autograd.gradgradcheck(layer.functional, (w, x))


def test_jax_mixed_portfolio_jit_vmap_hessian_and_source_checkpoint():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    spec = mixed_spec()
    layer = QuantumLayer.from_spec(spec, execution_method="planned", batch_chunk_size=2)
    dense = QuantumLayer.from_spec(spec)
    x = jnp.array([[0.2, 0.4], [-0.1, 0.7], [0.5, -0.3]], dtype=jnp.float32)
    w = jnp.array([[[0.3, 0.5]], [[-0.2, 0.6]]], dtype=jnp.float32)
    np.testing.assert_allclose(jax.jit(layer)(w, x), dense(w, x), atol=6e-7)
    first = jax.jit(jax.grad(lambda a, b: jnp.square(layer(a, b)).sum(), argnums=(0, 1)))(w, x)
    reference = jax.grad(lambda a, b: jnp.square(dense(a, b)).sum(), argnums=(0, 1))(w, x)
    for a, b in zip(first, reference, strict=True):
        np.testing.assert_allclose(a, b, atol=2e-6)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(layer, in_axes=(0, None)))(w, x), dense(w, x), atol=6e-7
    )
    _, got = jax.jit(layer.jvp)(w, x, jnp.ones_like(w), jnp.ones_like(x))
    _, expected = jax.jvp(dense, (w, x), (jnp.ones_like(w), jnp.ones_like(x)))
    np.testing.assert_allclose(got, expected, atol=2e-6)
    got = jax.jit(jax.hessian(lambda a: layer(a, x[0]).sum()))(w[0, 0])
    expected = jax.hessian(lambda a: dense(a, x[0]).sum())(w[0, 0])
    np.testing.assert_allclose(got, expected, atol=2e-6)
    checkpoint = layer.checkpoint(w[0, 0])
    restored, values = QuantumLayer.from_checkpoint(checkpoint, execution_method="planned")
    assert checkpoint.spec == spec
    np.testing.assert_allclose(restored(values, x), layer(w[0, 0], x), atol=5e-7)


def test_flax_owned_and_generated_weights_with_portfolio_execution():
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp
    import optax

    from openqml.interfaces.flax import QuantumLayer

    module = QuantumLayer.from_spec(mixed_spec(), execution_method="planned")
    x = jnp.array([[0.2, 0.4], [0.3, -0.2]], dtype=jnp.float32)
    variables = module.init(jax.random.key(4), x)
    gradients = jax.jit(jax.grad(lambda p: jnp.square(module.apply(p, x)).sum()))(variables)
    assert gradients["params"]["weights"][0] == 0
    optimizer = optax.adam(0.01)
    updates, _ = optimizer.update(gradients, optimizer.init(variables), variables)
    variables = optax.apply_updates(variables, updates)
    checkpoint = module.checkpoint(variables["params"]["weights"])
    restored, saved = QuantumLayer.from_checkpoint(checkpoint, execution_method="planned")
    np.testing.assert_allclose(restored.apply(saved, x), module.apply(variables, x), atol=5e-7)
    supplied = jnp.array([0.2, 0.7], dtype=jnp.float32)
    assert module.init(jax.random.key(5), x, weights=supplied) == {}
    gradient = jax.grad(lambda w: module.apply({}, x, weights=w).sum())(supplied)
    assert gradient[0] == 0 and gradient[1] != 0


@pytest.mark.parametrize("interface", ["torch", "jax"])
@pytest.mark.parametrize(
    "options",
    [
        {"planning_options": {}},
        {"execution_method": "pauli", "planning_options": {}},
        {"execution_method": "planned", "planning_options": []},
        {"execution_method": "planned", "planning_options": {"unknown": 1}},
        {"execution_method": "planned", "backend": "numpy"},
        {"execution_method": "planned", "checkpoint_interval": 1},
    ],
)
def test_layer_option_contracts(interface, options):
    pytest.importorskip(interface)
    if interface == "torch":
        from openqml.interfaces.torch import QuantumLayer
    else:
        from openqml.interfaces.jax import QuantumLayer
    with pytest.raises(ValidationError):
        QuantumLayer.from_spec(mixed_spec(), **options)
