"""Exact Heisenberg arithmetic against dense simulation and native autodiff."""

import json
from dataclasses import FrozenInstanceError
from itertools import product

import numpy as np
import pytest

from openqml import (
    Circuit,
    ClassicalCondition,
    ConversionError,
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
    parameter_shift_jacobian,
)
from openqml.ir import GATES, UNITARY_GATES


@pytest.mark.parametrize("gate", sorted(UNITARY_GATES - {"unitary"}))
def test_all_standard_gates_and_every_three_wire_pauli_match_dense_conjugation(gate):
    arity, angles = GATES[gate]
    order = (2, 0, 1)[:arity]
    circuit = Circuit(3).ry(0.2, 0).rx(-0.6, 1).ry(0.4, 2).cx(2, 1)
    circuit = circuit.gate(
        gate, *order, params=tuple((i + 1) * Parameter("w") for i in range(angles))
    )
    obs = tuple(PauliString("".join(p)) for p in product("IXYZ", repeat=3))
    plan = PauliPlan(circuit, obs)
    for value in (-0.71, 0.0, 0.42):
        bindings = {"w": value} if angles else {}
        np.testing.assert_allclose(
            plan.evaluate(bindings), expectations(circuit, obs, bindings), atol=4e-13
        )


def nonlinear_spec():
    x, w = Parameter("x"), Parameter("w")
    circuit = (
        Circuit(3)
        .ry(x * w, 0)
        .cx(0, 2)
        .rx(w.sin(), 1)
        .rzz(x, 2, 1)
        .u(x, w, x * w, 0)
        .with_global_phase(Parameter("phase") + Parameter("unused"))
    )
    return LayerSpec(
        circuit,
        ("unused", "x"),
        ("phase", "w"),
        (
            PauliString("XYZ", -0.7),
            Hamiltonian((PauliString("ZIX", 0.4), PauliString("III", 0.2))),
            PauliString("III", 0),
        ),
    )


@pytest.mark.parametrize("seed", range(6))
def test_random_mixed_circuits_and_hamiltonians_match_independent_dense_execution(seed):
    rng = np.random.default_rng(seed)
    c = Circuit(4)
    for i in range(12):
        q, r = rng.choice(4, 2, replace=False)
        angle = Parameter("a") * float(rng.normal()) + Parameter("b").sin()
        c = c.ry(angle, int(q)).gate(("cx", "cy", "cz", "iswap")[i % 4], int(q), int(r))
    obs = tuple(
        PauliString("".join(rng.choice(list("IXYZ"), 4)), float(rng.normal())) for _ in range(3)
    )
    obs += (Hamiltonian((obs[0], obs[1], PauliString("IIII", 0.3))),)
    plan = PauliPlan(c, obs)
    for a, b in ((0.2, -0.3), (-0.4, 0.5)):
        np.testing.assert_allclose(
            plan.evaluate(dict(a=a, b=b)), expectations(c, obs, dict(a=a, b=b)), atol=8e-12
        )


def test_broadcast_reuse_dead_path_elimination_and_source_preservation(monkeypatch):
    spec = nonlinear_spec()
    plan = PauliPlan(spec.circuit, spec.observables)
    assert plan.active_parameters == ("w", "x")
    assert plan.circuit is spec.circuit
    assert plan.to_dict()["construction_nodes"] > plan.to_dict()["arithmetic_nodes"]
    report = json.loads(plan.to_json())
    assert report["semantics"] == "declared_expectations"
    assert report["removed_parameters"] == ["phase", "unused"]
    assert report["diagnostics"][1]["code"] == "PAULI_EXACT_COMPILATION"
    x, w = np.array([[0.2], [0.4]]), np.array([0.3, -0.5, 0.7])
    expected = np.array(
        [
            [
                expectations(spec.circuit, spec.observables, dict(x=a, w=b, phase=0.2, unused=-0.5))
                for b in w
            ]
            for a in x[:, 0]
        ]
    )
    import openqml.pauli as compiler

    monkeypatch.setattr(compiler, "_compile", lambda *a, **k: pytest.fail("Repeated compilation"))
    for _ in range(2):
        np.testing.assert_allclose(plan.evaluate(dict(x=x, w=w)), expected, atol=3e-13)
    w *= -1
    changed = np.array(
        [
            [
                expectations(spec.circuit, spec.observables, dict(x=a, w=b, phase=0, unused=0))
                for b in w
            ]
            for a in x[:, 0]
        ]
    )
    np.testing.assert_allclose(plan.evaluate(dict(x=x, w=w)), changed, atol=3e-13)
    assert plan.evaluate(dict(x=np.empty((0, 1)), w=w)).shape == (0, 3, 3)
    with pytest.raises(FrozenInstanceError):
        plan.circuit = Circuit(1)
    assert convert(plan.circuit, target="ir").value is spec.circuit


def ghz_spec(width):
    circuit = Circuit(width).ry(Parameter("x") * Parameter("w"), 0)
    for q in range(width - 1):
        circuit = circuit.cx(q, q + 1)
    circuit = circuit.with_global_phase(Parameter("phase"))
    return LayerSpec(
        circuit,
        ("x",),
        ("phase", "w"),
        (
            PauliString("I" * (width - 1) + "Z"),
            PauliString("X" * width),
            Hamiltonian(
                (PauliString("Z" + "I" * (width - 2) + "Z", 0.4), PauliString("I" * width, 0.2))
            ),
        ),
    )


def test_large_connected_circuit_never_allocates_quantum_states(monkeypatch):
    spec = ghz_spec(512)
    import openqml.adjoint as adjoint
    import openqml.simulator as dense

    monkeypatch.setattr(dense, "statevector", lambda *a, **k: pytest.fail("Dense state allocation"))
    monkeypatch.setattr(dense, "_apply_matrix", lambda *a, **k: pytest.fail("Dense gate execution"))
    monkeypatch.setattr(
        adjoint, "_forward", lambda *a, **k: pytest.fail("Adjoint state allocation")
    )
    plan = PauliPlan(
        spec.circuit, spec.observables, max_terms=2, max_nodes=64, max_work=2000, max_memory_mb=0.02
    )
    x, w = np.linspace(-0.5, 0.5, 12), 0.7
    expected = np.stack((np.cos(x * w), np.sin(x * w), np.full_like(x, 0.6)), axis=-1)
    np.testing.assert_allclose(plan.evaluate(dict(x=x, w=w)), expected, atol=2e-15)
    assert plan.peak_terms == 2 and plan.to_dict()["arithmetic_nodes"] <= 8


@pytest.mark.parametrize(
    "options,code",
    [
        ({"max_terms": 1}, "PAULI_TERM_LIMIT"),
        ({"max_nodes": 2}, "PAULI_NODE_LIMIT"),
        ({"max_work": 1}, "PAULI_WORK_LIMIT"),
        ({"max_operations": 1}, "DECOMPOSITION_LIMIT"),
    ],
)
def test_growth_limits_return_diagnostics_without_an_approximate_plan(options, code):
    circuit = Circuit(2).ry(Parameter("x"), 0).cx(0, 1)
    with pytest.raises(ConversionError) as error:
        PauliPlan(circuit, [PauliString("IZ")], **options)
    assert error.value.diagnostics[0].code == code
    assert error.value.diagnostics[0].location


def test_observable_stream_and_source_hamiltonian_terms_are_bounded_before_collection():
    def readouts():
        for _ in range(4):
            yield PauliString("Z")
        pytest.fail("Readout iterator consumed beyond its output budget")

    with pytest.raises(ConversionError, match="PAULI_NODE_LIMIT"):
        PauliPlan(Circuit(1), readouts(), max_nodes=3)
    many = Hamiltonian((PauliString("Z"),) * 4)
    with pytest.raises(ConversionError, match="PAULI_NODE_LIMIT"):
        PauliPlan(Circuit(1), [many], max_nodes=3)


@pytest.mark.parametrize(
    "options",
    [
        {"max_nodes": 1},
        {"max_terms": 0},
        {"max_work": True},
        {"max_operations": -1},
        {"max_memory_mb": 0},
        {"max_memory_mb": float("nan")},
    ],
)
def test_invalid_limits_fail_before_compilation(options):
    with pytest.raises(ValidationError):
        PauliPlan(Circuit(1), [PauliString("Z")], **options)


@pytest.mark.parametrize("obs", [None, [], "Z", {PauliString("Z")}, [1], [PauliString("ZZ")]])
def test_invalid_observables(obs):
    with pytest.raises(ValidationError):
        PauliPlan(Circuit(1), obs)


@pytest.mark.parametrize(
    "circuit",
    [
        Circuit(1).measure(0, 0),
        Circuit(1).reset(0),
        Circuit(1).bit_flip(0.2, 0),
        Circuit(1).if_else(ClassicalCondition((0,), 1), Circuit(1).x(0)),
    ],
)
def test_effects_are_explicitly_rejected(circuit):
    with pytest.raises(ConversionError) as error:
        PauliPlan(circuit, [PauliString("Z")])
    assert error.value.diagnostics[0].code == "PAULI_EFFECTS_UNSUPPORTED"


def test_general_matrix_lowering_is_explicit_and_single_wire_matrices_keep_expectations():
    matrix = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    c = Circuit(1).ry(0.7, 0).unitary(matrix, 0)
    obs = [PauliString(p) for p in "XYZ"]
    np.testing.assert_allclose(PauliPlan(c, obs).evaluate(), expectations(c, obs), atol=2e-14)
    with pytest.raises(ConversionError) as error:
        PauliPlan(Circuit(2).unitary(np.eye(4), 1, 0), [PauliString("ZZ")])
    assert error.value.diagnostics[0].code == "DECOMPOSITION_UNAVAILABLE"


@pytest.mark.parametrize(
    "bindings",
    [
        {},
        {"x": 0.2, "missing": 1},
        {"x": True},
        {"x": 0.2j},
        {"x": float("nan")},
        {"x": "0.2"},
        {"x": np.ones((2, 3)), "phase": np.ones(5)},
    ],
)
def test_invalid_bindings_are_rejected(bindings):
    c = Circuit(1).ry(Parameter("x"), 0).with_global_phase(Parameter("phase"))
    plan = PauliPlan(c, [PauliString("Z")])
    with pytest.raises(ValidationError):
        plan.evaluate(bindings)


def test_memory_limits_are_checked_against_real_arithmetic_batch_work():
    plan = PauliPlan(Circuit(1).ry(Parameter("x"), 0), [PauliString("Z")], max_memory_mb=0.001)
    assert plan.evaluate(dict(x=0.2)).shape == (1,)
    with pytest.raises(ResourceError):
        plan.evaluate(dict(x=np.ones(100)))


def test_parameter_expressions_compile_into_shared_nodes_without_runtime_interpretation(
    monkeypatch,
):
    x, p, n = Parameter("x"), Parameter("power"), Parameter("numerator")
    angle = x.sin() + x.cos() + x.tan() + x.exp() + x.log() + x.sqrt() + x**p + n / (x + 1)
    circuit = Circuit(1).ry(angle, 0)
    plan = PauliPlan(circuit, [PauliString("Z"), PauliString("X")])
    values = dict(x=np.array([0.4, 0.7]), power=1.3, numerator=0.6)
    expected = expectations(circuit, plan.observables, values)
    monkeypatch.setattr(
        Expression, "evaluate", lambda *a, **k: pytest.fail("Runtime expression interpretation")
    )
    np.testing.assert_allclose(plan.evaluate(values), expected, atol=3e-14)
    assert all(node[0] != "angle" for node in plan._nodes)


def test_deeply_shared_parameter_dag_has_linear_discovery_compilation_and_evaluation():
    angle = Parameter("x").sin()
    for _ in range(72):
        angle = Expression(op="add", args=(angle, angle))
    assert angle.parameters == {"x"}
    circuit = Circuit(1).ry(angle, 0)
    plan = PauliPlan(circuit, [PauliString("Z")], max_nodes=100, max_memory_mb=0.01)
    assert circuit.parameters == ("x",)
    assert plan.to_dict()["arithmetic_nodes"] < 80
    np.testing.assert_allclose(
        plan.evaluate(dict(x=0.0001)), [np.cos(2.0**72 * np.sin(0.0001))], atol=1e-14
    )
    with pytest.raises(ConversionError, match="PAULI_NODE_LIMIT"):
        PauliPlan(circuit, [PauliString("Z")], max_nodes=8)


@pytest.mark.parametrize("operation", ["div", "log"])
def test_invalid_domains_discovered_by_compiled_constant_folding_are_validation_errors(operation):
    zero = Expression(op="mul", args=(Expression(terms=(("x", 1),)), Expression()))
    angle = Expression(
        op=operation, args=(Expression(constant=1), zero) if operation == "div" else (zero,)
    )
    with pytest.raises(ValidationError, match="undefined"):
        PauliPlan(Circuit(1).ry(angle, 0), [PauliString("Z")])


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "pennylane", "openqasm3"])
def test_native_import_to_shared_pauli_compiler_and_full_source_export(framework):
    if framework != "openqasm3":
        pytest.importorskip(framework)
    c = Circuit(3).ry(0.3, 2).cx(2, 0).u(0.2, -0.4, 0.7, 1).with_global_phase(0.2)
    obs = (PauliString("ZIX"), PauliString("IXY"))
    native = convert(c, target=framework).value
    imported = convert(native, source=framework, target="ir").value
    plan = PauliPlan(imported, obs)
    np.testing.assert_allclose(plan.evaluate(), expectations(c, obs), atol=3e-13)
    restored = convert(
        convert(plan.circuit, target=framework).value, source=framework, target="ir"
    ).value
    assert restored.num_qubits == c.num_qubits
    from openqml import statevector

    np.testing.assert_allclose(statevector(restored), statevector(c), atol=3e-13)


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "openqasm3"])
def test_symbolic_native_imports_reuse_active_names_and_nonlinear_angles(framework):
    if framework != "openqasm3":
        pytest.importorskip(framework)
    c = Circuit(3).ry(Parameter("feature") * Parameter("weight"), 2).cx(2, 0)
    c = c.with_global_phase(Parameter("source_phase"))
    result = convert(c, target=framework)
    restored = convert(result.value, source=framework, target="ir").value
    plan = PauliPlan(restored, [PauliString("ZII")])
    assert plan.active_parameters == ("feature", "weight")
    np.testing.assert_allclose(
        plan.evaluate(dict(feature=np.array([0.3, 0.6]), weight=0.4))[:, 0],
        np.cos(np.array([0.3, 0.6]) * 0.4),
        atol=3e-14,
    )


@pytest.mark.parametrize("interface", ["torch", "jax"])
def test_entirely_removed_parameter_groups_keep_zero_gradients_and_higher_derivatives(interface):
    c = Circuit(1).rz(Parameter("x"), 0).with_global_phase(Parameter("phase"))
    spec = LayerSpec(c, ("x",), ("phase",), (PauliString("X"), PauliString("I")))
    if interface == "torch":
        torch = pytest.importorskip("torch")
        from openqml.interfaces.torch import QuantumLayer

        layer = QuantumLayer.from_spec(spec, execution_method="pauli", dtype=torch.float64)
        x = torch.full((3, 1), 1e200, dtype=torch.float64, requires_grad=True)
        w = torch.full((1,), 1e200, dtype=torch.float64, requires_grad=True)
        out = layer.functional(w, x)
        torch.testing.assert_close(out, torch.tensor([[0.0, 1.0]] * 3, dtype=out.dtype))
        first = torch.autograd.grad(out.sum(), (w, x), create_graph=True)
        second = torch.autograd.grad(sum(a.sum() for a in first), (w, x))
        assert all(torch.count_nonzero(a) == 0 for a in (*first, *second))
    else:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        layer = QuantumLayer.from_spec(spec, execution_method="pauli")
        x, w = jnp.ones((3, 1), dtype=jnp.float32), jnp.ones((1,), dtype=jnp.float32)
        first = jax.jit(jax.grad(lambda a, b: layer(a, b).sum(), argnums=(0, 1)))(w, x)
        assert all(jnp.count_nonzero(a) == 0 for a in first)
        hessian = jax.jit(jax.hessian(lambda a: layer(a, x).sum()))(w)
        np.testing.assert_array_equal(hessian, np.zeros((1, 1)))
    assert layer.pauli_plan.active_parameters == ()
    assert layer.to_spec() == spec


@pytest.mark.parametrize("chunk", [None, 2])
@pytest.mark.parametrize("reduce", [False, True])
def test_torch_broadcast_forward_reverse_second_order_and_source_checkpoint(chunk, reduce):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    spec = nonlinear_spec()
    layer = QuantumLayer.from_spec(
        spec,
        execution_method="pauli",
        dtype=torch.float64,
        batch_chunk_size=chunk,
        reduce_readout=reduce,
    )
    dense = QuantumLayer.from_spec(spec, dtype=torch.float64)
    x = torch.tensor(
        [[0.1, 0.4], [-0.3, 0.7], [0.2, -0.6]], dtype=torch.float64, requires_grad=True
    )
    w = torch.tensor([[[0.8, 0.5]], [[0.4, -0.3]]], dtype=torch.float64, requires_grad=True)
    actual, expected = layer.functional(w, x), dense.functional(w, x)
    torch.testing.assert_close(actual, expected, atol=3e-12, rtol=3e-12)
    got = torch.autograd.grad(actual.square().sum(), (w, x), create_graph=True)
    reference = torch.autograd.grad(expected.square().sum(), (w, x), create_graph=True)
    for a, b in zip(got, reference, strict=True):
        torch.testing.assert_close(a, b, atol=4e-12, rtol=4e-12)
    second = torch.autograd.grad(sum(a.sum() for a in got), (w, x))
    second_reference = torch.autograd.grad(sum(a.sum() for a in reference), (w, x))
    for a, b in zip(second, second_reference, strict=True):
        torch.testing.assert_close(a, b, atol=8e-12, rtol=8e-12)
    _, tangent = torch.func.jvp(layer.functional, (w, x), (torch.ones_like(w), torch.ones_like(x)))
    _, tangent_reference = torch.func.jvp(
        dense.functional, (w, x), (torch.ones_like(w), torch.ones_like(x))
    )
    torch.testing.assert_close(tangent, tangent_reference, atol=4e-12, rtol=4e-12)
    layer.load_state_dict(dense.state_dict())
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    layer(x.detach()).square().sum().backward()
    optimizer.step()
    assert layer.weights.grad[0] == 0
    checkpoint = layer.checkpoint()
    assert checkpoint.spec == spec
    for method in ("auto", "pauli"):
        restored = QuantumLayer.from_checkpoint(checkpoint, execution_method=method)
        torch.testing.assert_close(restored(x), layer(x), atol=4e-12, rtol=4e-12)
    assert layer(torch.empty((0, 2), dtype=torch.float64)).shape == (0, 3)


def test_torch_large_width_and_fullgraph_compile_keep_real_native_gradients():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    layer = QuantumLayer.from_spec(
        ghz_spec(128),
        execution_method="pauli",
        max_qubits=128,
        max_memory_mb=0.01,
        dtype=torch.float64,
    )
    x = torch.tensor([[0.2], [0.4]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([0.7, 0.3], dtype=torch.float64, requires_grad=True)
    compiled = torch.compile(layer.functional, backend="aot_eager", fullgraph=True)
    out = compiled(w, x)
    torch.testing.assert_close(out[..., 0], torch.cos(x[..., 0] * w[1]))
    torch.testing.assert_close(out[..., 1], torch.sin(x[..., 0] * w[1]))
    assert torch.autograd.gradgradcheck(layer.functional, (w, x))


@pytest.mark.parametrize("chunk", [None, 2])
def test_jax_jit_vmap_jvp_and_hessians_match_dense(chunk):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    spec = nonlinear_spec()
    layer = QuantumLayer.from_spec(spec, execution_method="pauli", batch_chunk_size=chunk)
    dense = QuantumLayer.from_spec(spec)
    x = jnp.array([[0.1, 0.4], [-0.3, 0.7], [0.2, -0.6]], dtype=jnp.float32)
    w = jnp.array([[[0.8, 0.5]], [[0.4, -0.3]]], dtype=jnp.float32)
    np.testing.assert_allclose(jax.jit(layer)(w, x), dense(w, x), atol=5e-7)
    gradients = jax.jit(jax.grad(lambda a, b: jnp.square(layer(a, b)).sum(), argnums=(0, 1)))(w, x)
    expected = jax.grad(lambda a, b: jnp.square(dense(a, b)).sum(), argnums=(0, 1))(w, x)
    for a, b in zip(gradients, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=2e-6)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(layer, in_axes=(0, None)))(w, x), dense(w, x), atol=5e-7
    )
    direction = (jnp.ones_like(w), jnp.ones_like(x))
    _, actual = jax.jit(layer.jvp)(w, x, *direction)
    _, reference = jax.jvp(dense, (w, x), direction)
    np.testing.assert_allclose(actual, reference, atol=2e-6)
    hessian = jax.jit(jax.hessian(lambda a: layer(a, x[0]).sum()))(w[0, 0])
    reference = jax.hessian(lambda a: dense(a, x[0]).sum())(w[0, 0])
    np.testing.assert_allclose(hessian, reference, atol=2e-6)
    checkpoint = layer.checkpoint(w[0, 0])
    assert checkpoint.spec == spec
    restored, values = QuantumLayer.from_checkpoint(checkpoint)
    np.testing.assert_allclose(restored(values, x), layer(w[0, 0], x), atol=5e-7)


def test_flax_generated_and_owned_weights_share_optax_tree_and_source_checkpoint():
    jax = pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax.numpy as jnp
    import optax

    from openqml.interfaces.flax import QuantumLayer

    module = QuantumLayer.from_spec(ghz_spec(64), execution_method="pauli", max_qubits=64)
    x = jnp.array([[0.2], [0.5]], dtype=jnp.float32)
    variables = module.init(jax.random.key(4), x)
    gradients = jax.jit(jax.grad(lambda p: jnp.square(module.apply(p, x)).sum()))(variables)
    assert gradients["params"]["weights"][0] == 0
    optimizer = optax.adam(0.01)
    updates, _ = optimizer.update(gradients, optimizer.init(variables), variables)
    variables = optax.apply_updates(variables, updates)
    checkpoint = module.checkpoint(variables["params"]["weights"])
    restored, saved = QuantumLayer.from_checkpoint(
        checkpoint, execution_method="pauli", max_qubits=64
    )
    np.testing.assert_allclose(restored.apply(saved, x), module.apply(variables, x), atol=3e-7)
    supplied = jnp.array([0.1, 0.7], dtype=jnp.float32)
    assert module.init(jax.random.key(5), x, weights=supplied) == {}
    generated_grad = jax.grad(lambda w: module.apply({}, x, weights=w).sum())(supplied)
    assert generated_grad[0] == 0 and generated_grad[1] != 0


@pytest.mark.parametrize("interface", ["torch", "jax"])
@pytest.mark.parametrize(
    "options",
    [
        {"execution_method": "unknown"},
        {"execution_method": True},
        {"pauli_options": {}},
        {"execution_method": "pauli", "pauli_options": []},
        {"execution_method": "pauli", "pauli_options": {"unknown": 1}},
        {"execution_method": "pauli", "backend": "numpy"},
        {"execution_method": "pauli", "checkpoint_interval": 1},
        {"execution_method": "pauli", "max_qubits": 1},
    ],
)
def test_invalid_layer_execution_configuration(interface, options):
    pytest.importorskip(interface)
    if interface == "torch":
        from openqml.interfaces.torch import QuantumLayer
    else:
        from openqml.interfaces.jax import QuantumLayer
    with pytest.raises((ValidationError, ResourceError)):
        QuantumLayer.from_spec(ghz_spec(3), **options)


def test_torch_gradients_match_parameter_shift_for_nonlinear_shared_angles():
    torch = pytest.importorskip("torch")
    spec = nonlinear_spec()
    plan = PauliPlan(spec.circuit, spec.observables)
    x = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    w = torch.tensor(-0.6, dtype=torch.float64, requires_grad=True)
    value = plan.evaluate(dict(x=x, w=w), backend="torch", like=w)
    actual = torch.stack(
        [torch.stack(torch.autograd.grad(y, (x, w), retain_graph=True)) for y in value[:2]]
    )
    expected = parameter_shift_jacobian(
        spec.circuit,
        spec.observables,
        [dict(x=0.4, w=-0.6, phase=0, unused=0)],
        parameters=("x", "w"),
    )[0, :2]
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=2e-12)
