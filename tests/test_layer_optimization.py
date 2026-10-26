"""Source ownership survives logical optimization and cross-framework training."""

import numpy as np
import pytest

from openqml import Circuit, ClassicalCondition, Parameter, PauliString, ValidationError, convert
from openqml.interfaces._spec import numeric_rows, prepare_layer

INPUTS = ("unused_input", "feature")
WEIGHTS = ("unused_weight", "weight")


def source_program(*, constant=False, effects=False):
    x, w = Parameter("feature"), Parameter("weight")
    a, b = Parameter("unused_input"), Parameter("unused_weight")
    circuit = Circuit(1).ry(a, 0).ry(-a, 0).rx(b, 0).rx(-b, 0)
    if not constant:
        circuit = circuit.ry(x * w, 0)
    if effects:
        circuit = circuit.measure(0, 0).if_else(ClassicalCondition((0,), 1), Circuit(1).x(0))
    return circuit


@pytest.mark.parametrize(
    "options",
    [
        {"optimize": 1},
        {"optimize": "yes"},
        {"optimize": False, "optimization_options": {}},
        {"optimize": True, "optimization_options": []},
        {"optimize": True, "optimization_options": {"unknown": 1}},
        {"optimize": True, "optimization_options": {"max_passes": 0}},
    ],
)
def test_invalid_optimization_configuration(options):
    with pytest.raises(ValidationError):
        prepare_layer(
            source_program(),
            INPUTS,
            WEIGHTS,
            None,
            **({"optimize": False, "optimization_options": None} | options),
        )


def test_source_ownership_is_validated_before_symbols_can_be_removed():
    with pytest.raises(ValidationError, match="Every circuit parameter"):
        prepare_layer(
            source_program(),
            ["feature"],
            ["weight"],
            None,
            optimize=True,
            optimization_options=None,
        )
    with pytest.raises(ValidationError, match="unique"):
        prepare_layer(
            source_program(),
            INPUTS,
            (*WEIGHTS, "feature"),
            None,
            optimize=True,
            optimization_options=None,
        )


def test_preparation_preserves_source_and_reports_actual_executable_program():
    source = source_program()
    inputs, weights, obs, compiled, report = prepare_layer(
        source,
        INPUTS,
        WEIGHTS,
        None,
        optimize=True,
        optimization_options={"max_passes": 2},
    )
    assert inputs == INPUTS and weights == WEIGHTS and obs == (PauliString("Z"),)
    assert len(source.operations) == 5 and len(compiled.operations) == 1
    assert set(report.removed_parameters) == {"unused_input", "unused_weight"}
    assert report.circuit is compiled
    _, _, _, unchanged, report = prepare_layer(
        source,
        INPUTS,
        WEIGHTS,
        None,
        optimize=False,
        optimization_options=None,
    )
    assert unchanged is source and report is None


class CountingBackend:
    from openqml import BackendCapabilities

    name = "optimized-layer-counting"
    capabilities = BackendCapabilities(native_expectations=True)

    def __init__(self):
        self.jobs = []

    def expectations_batch(self, circuits, observables, *, options):
        from openqml import expectations_batch

        self.jobs.extend(circuits)
        return expectations_batch(circuits, observables)


@pytest.mark.parametrize("backend", ["torch", "numpy"])
@pytest.mark.parametrize("effects", [False, True])
def test_torch_keeps_shapes_outputs_and_gradient_columns(backend, effects):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    c = source_program(effects=effects)
    layer = QuantumLayer(c, INPUTS, WEIGHTS, optimize=True, backend=backend, dtype=torch.float64)
    x = torch.tensor([[[0.1, 0.4], [-0.2, 0.9]]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([-0.3, 0.7], dtype=torch.float64, requires_grad=True)
    output = torch.func.functional_call(layer, {"weights": w}, (x,))
    expected = torch.ones_like(x[..., 1:]) if effects else torch.cos(x[..., 1:] * w[1])
    torch.testing.assert_close(output, expected)
    actual = torch.autograd.grad(output.sum(), (x, w))
    if effects:
        torch.testing.assert_close(actual[0], torch.zeros_like(x))
        torch.testing.assert_close(actual[1], torch.zeros_like(w))
    else:
        reference = torch.autograd.grad(expected.sum(), (x, w))
        for a, b in zip(actual, reference, strict=True):
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
    assert layer.source_circuit is c and layer.weights.shape == (2,)
    assert layer.optimization.removed_parameters == ("unused_input", "unused_weight")
    torch.testing.assert_close(
        layer(torch.empty(0, 2, dtype=torch.float64)), torch.empty(0, 1, dtype=torch.float64)
    )


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_torch_all_removed_and_large_unused_values_return_differentiable_zeros(backend):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    layer = QuantumLayer(
        source_program(constant=True),
        ["unused_input"],
        ["unused_weight"],
        optimize=True,
        backend=backend,
        dtype=torch.float64,
    )
    x = torch.full((3, 1), 1e300, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        layer.weights.fill_(1e300)
    out = layer(x)
    torch.testing.assert_close(out, torch.ones_like(out))
    dx, dw = torch.autograd.grad(out.sum(), (x, layer.weights), create_graph=backend == "torch")
    torch.testing.assert_close(dx, torch.zeros_like(x))
    torch.testing.assert_close(dw, torch.zeros_like(layer.weights))
    if backend == "torch":
        second = torch.autograd.grad(dx.sum() + dw.sum(), (x, layer.weights))
        assert all(torch.count_nonzero(t) == 0 for t in second)


def test_torch_optimized_native_gradgradcheck_compile_and_checkpoint():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    source = source_program()
    plain = QuantumLayer(source, INPUTS, WEIGHTS, dtype=torch.float64)
    layer = QuantumLayer(source, INPUTS, WEIGHTS, dtype=torch.float64, optimize=True)
    layer.load_state_dict(plain.state_dict())
    x = torch.tensor([[0.1, 0.5]], dtype=torch.float64, requires_grad=True)
    w = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)

    def function(x, w):
        return torch.func.functional_call(layer, {"weights": w}, (x,))

    assert torch.autograd.gradcheck(function, (x, w))
    assert torch.autograd.gradgradcheck(function, (x, w))
    compiled = torch.compile(layer, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), plain(x))
    assert compiled(x).shape == (1, 1)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        layer(x).sum().backward()
        optimizer.step()
    assert optimizer.state[layer.weights]["exp_avg"].shape == (2,)
    assert optimizer.state[layer.weights]["exp_avg"][0] == 0


def test_external_removed_symbols_submit_no_shifted_jobs():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    backend = CountingBackend()
    layer = QuantumLayer(
        source_program(), INPUTS, WEIGHTS, optimize=True, backend=backend, dtype=torch.float64
    )
    x = torch.tensor([[0.2, 0.4]], dtype=torch.float64, requires_grad=True)
    layer(x).sum().backward()
    assert len(backend.jobs) == 3  # one primal + two shared shifts of the active angle
    assert all(len(c.operations) == 1 for c in backend.jobs)
    assert numeric_rows(layer, np.array([0.3, 0.7]), np.array([[0.2, 0.4]])) == [
        {"feature": 0.4, "weight": 0.7}
    ]


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_torch_forward_ad_retains_source_tangent_order(backend):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    layer = QuantumLayer(
        source_program(), INPUTS, WEIGHTS, optimize=True, backend=backend, dtype=torch.float64
    )
    x = torch.tensor([[0.2, 0.4]], dtype=torch.float64)
    w = torch.tensor([0.5, 0.7], dtype=torch.float64)
    dx, dw = torch.tensor([[3.0, 0.2]], dtype=x.dtype), torch.tensor([7.0, -0.3], dtype=w.dtype)
    with torch.autograd.forward_ad.dual_level():
        dual_x = torch.autograd.forward_ad.make_dual(x, dx)
        dual_w = torch.autograd.forward_ad.make_dual(w, dw)
        result = torch.func.functional_call(layer, {"weights": dual_w}, (dual_x,))
        primal, tangent = torch.autograd.forward_ad.unpack_dual(result)
    torch.testing.assert_close(primal, torch.cos(x[:, 1:] * w[1]))
    expected = -torch.sin(x[:, 1:] * w[1]) * (dx[:, 1:] * w[1] + x[:, 1:] * dw[1])
    torch.testing.assert_close(tangent, expected, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
@pytest.mark.parametrize("constant", [False, True])
def test_jax_optimized_jit_vmap_vjp_jvp_and_native_hessian(backend, constant):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    names = (("unused_input",), ("unused_weight",)) if constant else (INPUTS, WEIGHTS)
    layer = QuantumLayer(source_program(constant=constant), *names, optimize=True, backend=backend)
    x = jnp.full((2, len(names[0])), 0.4, dtype=jnp.float32)
    w = jnp.full((len(names[1]),), 0.7, dtype=jnp.float32)

    def reference(w, x):
        return jnp.ones(x.shape[:-1] + (1,)) if constant else jnp.cos(x[..., 1:] * w[1])

    np.testing.assert_allclose(jax.jit(layer)(w, x), reference(w, x), atol=2e-7)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(layer, in_axes=(None, 0)))(w, x), reference(w, x), atol=2e-7
    )
    actual = jax.jit(jax.grad(lambda w, x: layer(w, x).sum(), argnums=(0, 1)))(w, x)
    expected = jax.grad(lambda w, x: reference(w, x).sum(), argnums=(0, 1))(w, x)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=3e-7)
    dw, dx = jnp.full_like(w, 0.2), jnp.full_like(x, 0.3)
    actual = jax.jit(layer.jvp)(w, x, dw, dx)
    expected = jax.jvp(reference, (w, x), (dw, dx))
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, atol=3e-7)
    if backend == "jax":
        actual = jax.jit(jax.hessian(lambda w: layer(w, x).sum()))(w)
        expected = jax.hessian(lambda w: reference(w, x).sum())(w)
        np.testing.assert_allclose(actual, expected, atol=3e-7)


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_flax_parameter_tree_and_optimizer_survive_elimination(backend):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    pytest.importorskip("flax")
    optax = pytest.importorskip("optax")
    from openqml.interfaces.flax import QuantumLayer

    x = jnp.array([[0.2, 0.4]], dtype=jnp.float32)
    plain = QuantumLayer(source_program(), INPUTS, WEIGHTS)
    module = QuantumLayer(
        source_program(),
        INPUTS,
        WEIGHTS,
        optimize=True,
        backend=backend,
        optimization_options={"max_passes": 2},
    )
    params = plain.init(jax.random.key(3), x)
    np.testing.assert_allclose(module.apply(params, x), plain.apply(params, x), atol=2e-7)
    optimizer = optax.adam(0.01)
    state = optimizer.init(params)
    gradient = jax.jit(jax.grad(lambda p: module.apply(p, x).sum()))(params)
    assert gradient["params"]["weights"].shape == (2,)
    assert gradient["params"]["weights"][0] == 0
    updates, _ = optimizer.update(gradient, state, params)
    updated = optax.apply_updates(params, updates)
    assert updated["params"]["weights"].shape == (2,)


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "openqasm3"])
def test_imported_program_optimizes_for_training_and_exports_back(framework):
    torch = pytest.importorskip("torch")
    if framework in {"qiskit", "cirq"}:
        pytest.importorskip(framework)
    from openqml import QuantumProgram
    from openqml.interfaces.torch import QuantumLayer

    native = convert(source_program(), target=framework).value
    program = (
        convert(native, source=framework, target="ir").value
        if framework == "openqasm3"
        else QuantumProgram.capture(native)
    )
    layer = QuantumLayer.from_program(program, INPUTS, optimize=True, dtype=torch.float64)
    assert set(layer.weight_parameters) == set(WEIGHTS)
    assert layer.optimization.after.gates == 1
    result = convert(layer.circuit, target=framework)
    assert result.value is not None
    x = torch.tensor([[0.2, 0.4]], dtype=torch.float64, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad[0, 0] == 0 and layer.weights.grad.shape == (2,)


def test_layer_optimization_limit_is_checked_before_training():
    torch = pytest.importorskip("torch")
    from openqml import ConversionError
    from openqml.interfaces.torch import QuantumLayer

    with pytest.raises(ConversionError) as exc:
        QuantumLayer(
            source_program(),
            INPUTS,
            WEIGHTS,
            dtype=torch.float64,
            optimize=True,
            optimization_options={"max_operations": 1},
        )
    assert exc.value.diagnostics[0].code == "OPTIMIZATION_LIMIT"
