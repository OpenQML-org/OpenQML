"""Exact quantum function reuse, native lane packing and source provenance."""

import json
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from openqml import (
    Circuit,
    ExpectationPlan,
    Hamiltonian,
    LayerSpec,
    Parameter,
    PauliString,
    ResourceError,
    ValidationError,
    convert,
    expectations,
)


def block_spec(blocks=4, *, tied=False):
    circuit, terms = Circuit(2 * blocks), []
    inputs = ("x",) if tied else tuple(f"x{i}" for i in range(blocks))
    weights = ("w",) if tied else tuple(f"w{i}" for i in range(blocks))
    matrix = np.diag([1, 1j, -1j, 1])
    for i in range(blocks):
        x, w = Parameter(inputs[0 if tied else i]), Parameter(weights[0 if tied else i])
        circuit = circuit.ry(x * w, 2 * i).unitary(matrix, 2 * i, 2 * i + 1)
        terms.append(PauliString("I" * (2 * i) + "ZI" + "I" * (2 * (blocks - i - 1))))
    obs = (PauliString("ZI" * blocks), Hamiltonian(tuple(terms)))
    return LayerSpec(circuit, inputs, weights, obs)


def count_kernels(monkeypatch):
    import openqml.simulator as simulator

    shapes, original = [], simulator._apply_matrix

    def counted(state, *args, **kwargs):
        shapes.append(tuple(state.shape))
        return original(state, *args, **kwargs)

    monkeypatch.setattr(simulator, "_apply_matrix", counted)
    return shapes


@pytest.mark.parametrize("blocks", [2, 7, 64])
def test_tied_dense_functions_execute_once_and_retain_every_source_group(blocks, monkeypatch):
    spec = block_spec(blocks, tied=True)
    plan = ExpectationPlan(spec.circuit, spec.observables)
    assert len(plan.groups) == blocks and len(plan.dense_batches) == 1
    batch = plan.dense_batches[0]
    assert len(batch.parameter_slots) == 1
    assert batch.source_groups == (tuple(range(blocks)),)
    assert len(batch.factor_maps[0]) == blocks
    assert plan.factor_aliases == (0,) * blocks
    assert plan.to_dict()["unique_factors"] == 1
    assert len(plan._nodes) <= 3 * math.ceil(math.log2(blocks)) + 3
    shapes = count_kernels(monkeypatch)
    values = plan.evaluate({"x": np.array([0.2, -0.3]), "w": 0.7})
    angle = np.array([0.2, -0.3]) * 0.7
    np.testing.assert_allclose(
        values, np.stack((np.cos(angle) ** blocks, blocks * np.cos(angle)), -1)
    )
    assert shapes == [(2, 4), (2, 4)]
    report = json.loads(plan.to_json())
    assert report["dense_unique_lanes"] == 1
    assert report["dense_batches"][0]["source_groups"] == [list(range(blocks))]
    assert all(g.source_operations == (2 * i, 2 * i + 1) for i, g in enumerate(plan.groups))
    with pytest.raises(FrozenInstanceError):
        batch.parameter_slots = ()


@pytest.mark.parametrize("cap", [1, 2, 3, 32])
def test_independent_parameters_pack_without_aliasing_or_losing_batch_dimensions(cap, monkeypatch):
    spec = block_spec(5)
    plan = ExpectationPlan(spec.circuit, spec.observables, max_dense_batch=cap)
    values = {
        **{
            name: np.linspace(-0.3, 0.2, 6).reshape(2, 3) + i / 20
            for i, name in enumerate(spec.input_parameters)
        },
        **{name: i / 10 + 0.3 for i, name in enumerate(spec.weight_parameters)},
    }
    expected = expectations(spec.circuit, spec.observables, values)
    shapes = count_kernels(monkeypatch)
    np.testing.assert_allclose(plan.evaluate(values), expected, atol=3e-14)
    assert len(shapes) == 2 * math.ceil(5 / cap)
    assert len(plan.dense_batches) == 1
    assert len(plan.dense_batches[0].parameter_slots) == 5
    assert all(s[-3:] == (2, 3, 4) for s in shapes)
    assert all(s[0] <= cap for s in shapes if len(s) == 4)


def test_templates_union_readouts_and_reuse_a_state_with_different_source_outputs(monkeypatch):
    c = Circuit(6)
    for q in (0, 2, 4):
        c = c.ry(Parameter("x"), q).unitary(np.eye(4), q, q + 1)
    obs = (PauliString("XIIIII"), PauliString("IIZIII"), PauliString("IIIIYI"))
    plan = ExpectationPlan(c, obs)
    assert len(plan.groups) == 3
    assert len(plan.dense_batches) == 1
    assert len(plan.dense_batches[0].parameter_slots) == 1
    assert {p.paulis for p in plan.dense_batches[0].observables} == {"XI", "ZI", "YI"}
    expected = expectations(c, obs, {"x": 0.4})
    shapes = count_kernels(monkeypatch)
    np.testing.assert_allclose(plan.evaluate({"x": 0.4}), expected, atol=3e-14)
    assert shapes == [(4,), (4,)]


def test_tied_and_untied_angle_occurrences_share_a_template_but_not_a_lane(monkeypatch):
    x, y = Parameter("x"), Parameter("y")
    c = (
        Circuit(4)
        .ry(x, 0)
        .rx(x, 1)
        .unitary(np.eye(4), 0, 1)
        .ry(x, 2)
        .rx(y, 3)
        .unitary(np.eye(4), 2, 3)
    )
    obs = (PauliString("ZZZZ"),)
    plan = ExpectationPlan(c, obs)
    assert len(plan.dense_batches) == 1
    assert len(plan.dense_batches[0].parameter_slots) == 2
    expected = expectations(c, obs, {"x": 0.2, "y": -0.5})
    shapes = count_kernels(monkeypatch)
    np.testing.assert_allclose(plan.evaluate({"x": 0.2, "y": -0.5}), expected, atol=3e-14)
    assert shapes == [(2, 4)] * 3


@pytest.mark.parametrize("difference", ["matrix", "operands", "gate", "order"])
def test_exact_template_keys_never_merge_unequal_quantum_functions(difference):
    matrix = np.diag([1, 1, 1, np.exp(0.1j)])
    altered = np.diag([1, 1, 1, np.exp((0.1 + 1e-12) * 1j)])
    c = Circuit(4).ry(Parameter("x"), 0).rx(0.3, 1).unitary(matrix, 0, 1)
    if difference == "order":
        c = c.unitary(matrix, 2, 3).ry(Parameter("x"), 2).rx(0.3, 3)
    else:
        c = c.gate("rx" if difference == "gate" else "ry", 2, params=(Parameter("x"),))
        c = c.rx(0.3, 3).unitary(
            altered if difference == "matrix" else matrix,
            *((3, 2) if difference == "operands" else (2, 3)),
        )
    obs = (PauliString("YYYY"),)
    plan = ExpectationPlan(c, obs)
    assert len(plan.dense_batches) == 2
    assert len({b.template_id for b in plan.dense_batches}) == 2
    np.testing.assert_allclose(
        plan.evaluate({"x": 0.4}), expectations(c, obs, {"x": 0.4}), atol=2e-14
    )


def test_parameter_names_and_equal_numeric_bindings_do_not_define_function_identity():
    spec = block_spec(3)
    plan = ExpectationPlan(spec.circuit, spec.observables)
    before = plan.to_json()
    assert len(plan.dense_batches[0].parameter_slots) == 3
    for offset in (0, 0.3):
        bindings = {
            n: 0.2 + offset * i
            for i, n in enumerate(spec.input_parameters + spec.weight_parameters)
        }
        np.testing.assert_allclose(
            plan.evaluate(bindings),
            expectations(spec.circuit, spec.observables, bindings),
            atol=3e-14,
        )
    assert plan.to_json() == before


def test_memory_budget_reduces_pack_width_before_allocating(monkeypatch):
    spec = block_spec(6)
    probe = ExpectationPlan(spec.circuit, spec.observables)
    rows = 7
    # Fit retained arrays and just over two lanes for this runtime batch.
    budget = probe._retained_bytes_per_row + 2.1 * probe.dense_batches[0].working_bytes_per_lane
    plan = ExpectationPlan(spec.circuit, spec.observables, max_memory_mb=budget * rows / 2**20)
    values = {n: np.ones(rows) * 0.2 for n in spec.input_parameters + spec.weight_parameters}
    expected = expectations(spec.circuit, spec.observables, values)
    shapes = count_kernels(monkeypatch)
    np.testing.assert_allclose(plan.evaluate(values), expected, atol=3e-14)
    assert len(shapes) == 6 and all(s == (2, rows, 4) for s in shapes)
    shapes.clear()
    with pytest.raises(ResourceError, match="max_memory_mb"):
        plan.evaluate({n: np.ones(rows * 20) for n in values})
    assert not shapes


@pytest.mark.parametrize("cap", [0, -1, True, 1.5, "2"])
def test_dense_batch_cap_is_a_positive_integer(cap):
    spec = block_spec()
    with pytest.raises(ValidationError, match="max_dense_batch"):
        ExpectationPlan(spec.circuit, spec.observables, max_dense_batch=cap)


@pytest.mark.parametrize("framework", ["json", "qiskit", "cirq", "pennylane", "openqasm3"])
def test_imported_blocks_keep_structural_templates_and_source_exports(framework):
    if framework in {"qiskit", "cirq", "pennylane"}:
        pytest.importorskip(framework)
    if framework == "openqasm3":
        pytest.importorskip("qiskit")  # Opaque multi-qubit matrix synthesis is deferred to the SDK.
    spec = block_spec(3, tied=True)
    bound = spec.circuit.bind({"x": 0.3, "w": 0.4})
    value = convert(bound, target=framework).value
    restored = convert(value, source=framework, target="ir").value
    # QASM decomposes opaque matrices into portable gates, so force its
    # surviving local programs onto dense execution with a term cap.
    plan = ExpectationPlan(restored, spec.observables, max_terms=1)
    np.testing.assert_allclose(plan.evaluate(), expectations(bound, spec.observables), atol=3e-13)
    assert plan.circuit is restored
    assert len(plan.groups) == 3
    if plan.dense_batches:
        assert len(plan.dense_batches) == 1
        assert len(plan.dense_batches[0].parameter_slots) == 1
    if framework != "openqasm3":
        original = ExpectationPlan(bound, spec.observables)
        assert plan.dense_batches[0].template_id == original.dense_batches[0].template_id


@pytest.mark.parametrize("tied", [False, True])
def test_torch_packed_broadcast_training_hessians_jvp_and_portable_checkpoints(tied):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    spec = block_spec(3, tied=tied)
    layer = QuantumLayer.from_spec(
        spec,
        execution_method="planned",
        planning_options={"max_dense_batch": 2},
        batch_chunk_size=2,
        dtype=torch.float64,
    )
    reference = QuantumLayer.from_spec(spec, dtype=torch.float64)
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(
        2,
        3,
        len(spec.input_parameters),
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    w = torch.randn(
        2,
        1,
        len(spec.weight_parameters),
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    torch.testing.assert_close(layer.functional(w, x), reference.functional(w, x))
    for fn in (layer.functional, reference.functional):
        output = fn(w, x)
        gradient = torch.autograd.grad(output.square().sum(), (w, x), create_graph=True)
        hessian = torch.autograd.grad(sum(g.square().sum() for g in gradient), (w, x))
        if fn == layer.functional:
            actual = gradient, hessian
        else:
            for a, b in zip(actual, (gradient, hessian), strict=True):
                for first, second in zip(a, b, strict=True):
                    torch.testing.assert_close(first, second, atol=3e-12, rtol=3e-12)
    tangents = (torch.ones_like(w), torch.ones_like(x))
    for a, b in zip(
        torch.func.jvp(layer.functional, (w, x), tangents),
        torch.func.jvp(reference.functional, (w, x), tangents),
        strict=True,
    ):
        torch.testing.assert_close(a, b, atol=3e-12, rtol=3e-12)
    checkpoint = layer.checkpoint()
    assert checkpoint.spec == spec
    restored = QuantumLayer.from_checkpoint(checkpoint, execution_method="planned")
    torch.testing.assert_close(restored(x), layer(x))
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    layer(x.detach()).square().mean().backward()
    optimizer.step()
    assert torch.isfinite(layer.weights).all()


def test_torch_packed_zero_factors_keep_cross_derivatives_and_compile():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    c = (
        Circuit(4)
        .ry(Parameter("x0") * Parameter("w0"), 0)
        .unitary(np.eye(4), 0, 1)
        .ry(Parameter("x1") * Parameter("w1"), 2)
        .unitary(np.eye(4), 2, 3)
    )
    # X expectations vanish at zero; cross derivatives of their product do not.
    spec = LayerSpec(c, ("x0", "x1"), ("w0", "w1"), (PauliString("XIXI"),))
    layer = QuantumLayer.from_spec(spec, execution_method="planned", dtype=torch.float64)
    w = torch.ones(2, dtype=torch.float64, requires_grad=True)
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    hessian = torch.func.hessian(lambda a: layer.functional(w, a).sum())(x)
    torch.testing.assert_close(hessian, torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64))
    x = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    compiled = torch.compile(layer.functional, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(w, x), layer.functional(w, x))
    assert torch.autograd.gradgradcheck(layer.functional, (w, x))


def test_jax_packed_jit_vmap_hessian_and_flax_generated_weights():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    spec = block_spec(3)
    layer = QuantumLayer.from_spec(spec, execution_method="planned")
    reference = QuantumLayer.from_spec(spec)
    w = jnp.array([0.3, -0.2, 0.4], dtype=jnp.float32)
    x = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 20
    np.testing.assert_allclose(jax.jit(layer)(w, x), reference(w, x), atol=2e-6)
    got = jax.jit(jax.hessian(lambda a: layer(a, x).sum()))(w)
    expected = jax.hessian(lambda a: reference(a, x).sum())(w)
    np.testing.assert_allclose(got, expected, atol=3e-6)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(layer, in_axes=(0, None)))(jnp.stack((w, w * 2)), x),
        reference(jnp.stack((w, w * 2))[:, None, :], x),
        atol=3e-6,
    )
    pytest.importorskip("flax")
    from openqml.interfaces.flax import QuantumLayer as FlaxLayer

    module = FlaxLayer.from_spec(
        spec, execution_method="planned", planning_options={"max_dense_batch": 2}
    )
    variables = module.init(jax.random.key(3), x)
    np.testing.assert_allclose(
        jax.jit(module.apply)(variables, x, weights=w), reference(w, x), atol=2e-6
    )


def sparse_spec():
    base = block_spec(4)
    obs = tuple(
        PauliString("II" * i + label + "II" * (3 - i))
        for i, label in enumerate(("XI", "ZI", "YI", "XZ"))
    )
    return LayerSpec(base.circuit, base.input_parameters, base.weight_parameters, obs)


def test_sparse_lane_outputs_do_not_retain_unrequested_readout_storage(monkeypatch):
    import openqml.execution as execution

    spec = sparse_spec()
    plan = ExpectationPlan(spec.circuit, spec.observables)
    bindings = {n: np.ones(3) * 0.2 for n in spec.input_parameters + spec.weight_parameters}
    captured, original = [], execution._evaluate_program

    def inspect(nodes, roots, local, ops, shape, factors=()):
        if factors:
            storage = {}
            for value in factors:
                while isinstance(value.base, np.ndarray):
                    value = value.base
                storage[id(value)] = value.nbytes
            captured.append(sum(storage.values()))
        return original(nodes, roots, local, ops, shape, factors)

    monkeypatch.setattr(execution, "_evaluate_program", inspect)
    np.testing.assert_allclose(
        plan.evaluate(bindings), expectations(spec.circuit, spec.observables, bindings), atol=3e-14
    )
    assert captured == [plan.factor_count * 3 * 8]


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_sparse_readout_union_preserves_compiled_native_gradients(backend):
    spec = sparse_spec()
    if backend == "torch":
        torch = pytest.importorskip("torch")
        from openqml.interfaces.torch import QuantumLayer

        planned = QuantumLayer.from_spec(spec, execution_method="planned", dtype=torch.float64)
        reference = QuantumLayer.from_spec(spec, dtype=torch.float64)
        w = torch.ones(4, dtype=torch.float64, requires_grad=True) * 0.3
        x = torch.linspace(-0.2, 0.4, 12, dtype=torch.float64).reshape(3, 4).requires_grad_()
        compiled = torch.compile(planned.functional, backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(compiled(w, x), reference.functional(w, x))
        actual = torch.autograd.grad(compiled(w, x).square().sum(), (w, x))
        expected = torch.autograd.grad(reference.functional(w, x).square().sum(), (w, x))
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b)
    else:
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        planned = QuantumLayer.from_spec(spec, execution_method="planned")
        reference = QuantumLayer.from_spec(spec)
        w = jnp.ones(4, dtype=jnp.float32) * 0.3
        x = jnp.linspace(-0.2, 0.4, 12, dtype=jnp.float32).reshape(3, 4)
        actual = jax.jit(jax.grad(lambda a, b: jnp.square(planned(a, b)).sum(), argnums=(0, 1)))(
            w, x
        )
        expected = jax.grad(lambda a, b: jnp.square(reference(a, b)).sum(), argnums=(0, 1))(w, x)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, atol=2e-6)


def test_repeated_zero_factor_keeps_its_second_derivative():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    angle = Parameter("x") * Parameter("w")
    c = Circuit(4).ry(angle, 0).unitary(np.eye(4), 0, 1).ry(angle, 2).unitary(np.eye(4), 2, 3)
    layer = QuantumLayer(
        c, ("x",), ("w",), (PauliString("XIXI"),), execution_method="planned", dtype=torch.float64
    )
    assert layer.execution_plan.factor_aliases == (0, 0)
    w, x = torch.ones(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
    hessian = torch.func.hessian(lambda a: layer.functional(w, a).sum())(x)
    torch.testing.assert_close(hessian, torch.full((1, 1), 2.0, dtype=torch.float64))


def test_signed_matrix_zeros_have_stable_exact_template_identity():
    first, second = np.eye(4, dtype=complex), np.eye(4, dtype=complex)
    second[0, 1] = complex(-0.0, -0.0)
    c = Circuit(4).unitary(first, 0, 1).unitary(second, 2, 3)
    plan = ExpectationPlan(c, (PauliString("ZZZZ"),), max_memory_mb=1e308)
    assert len(plan.dense_batches) == 1
    assert plan.dense_batches[0].source_groups == ((0, 1),)
    np.testing.assert_allclose(plan.evaluate(), [1.0])
