"""Standard first-order transforms through external quantum execution bridges."""

import numpy as np
import pytest

from openqml import (
    BackendCapabilities,
    Circuit,
    ClassicalCondition,
    Hamiltonian,
    LayerSpec,
    Parameter,
    PauliString,
    ValidationError,
    expectations_batch,
    plan_conversion,
)


def spec():
    x, y = Parameter("x"), Parameter("y")
    a, b, c = Parameter("a"), Parameter("b"), Parameter("c")
    circuit = (
        Circuit(3)
        .ry(x * a, 0)
        .rx(y.sin() + b / 2, 1)
        .crx(c * x, 0, 1)
        .ry(Parameter("dead"), 2)
        .with_global_phase(Parameter("phi"))
    )
    return LayerSpec(
        circuit,
        ("dead", "x", "y"),
        ("phi", "a", "b", "c"),
        (
            PauliString("ZII"),
            Hamiltonian((PauliString("IXI", 0.3), PauliString("ZZI", 0.4))),
        ),
    )


def noisy_spec():
    x, w, p = Parameter("x"), Parameter("w"), Parameter("p")
    circuit = (
        Circuit(2)
        .ry(x * w, 0)
        .rx(x.sin(), 1)
        .measure(0, 0)
        .if_else(ClassicalCondition((0,), 1), Circuit(2).ry(w * w, 1), Circuit(2).rz(w, 1))
        .amplitude_damping(p, 1)
    )
    return LayerSpec(circuit, ("x",), ("p", "w"), (PauliString("IZ"), PauliString("IX")))


def tree_close(a, b, leaves, atol):
    first, second = leaves(a), leaves(b)
    assert len(first) == len(second)
    for x, y in zip(first, second, strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=atol, rtol=atol)


@pytest.mark.parametrize("method", ["adjoint", "parameter-shift"])
def test_torch_standard_products_jacobians_and_broadcast_reductions(method):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    configuration = spec()
    external = QuantumLayer.from_spec(
        configuration,
        backend="numpy",
        gradient_method=method,
        reduce_readout=True,
        dtype=torch.float64,
        batch_chunk_size=3,
    )
    native = QuantumLayer.from_spec(configuration, reduce_readout=True, dtype=torch.float64)
    w = torch.tensor([0.2, 0.6, -0.3, 0.4], dtype=torch.float64)
    x = torch.linspace(-0.3, 0.5, 6, dtype=torch.float64).reshape(2, 3)
    dw, dx = torch.ones_like(w) * 0.2, torch.ones_like(x) * -0.1

    def leaves(t):
        return [v.detach().numpy() for v in torch.utils._pytree.tree_leaves(t)]

    operations = [
        lambda f: torch.func.jvp(f, (w, x), (dw, dx)),
        lambda f: torch.func.grad(lambda a, b: f(a, b).square().sum(), argnums=(0, 1))(w, x),
        lambda f: torch.func.jacfwd(f, argnums=(0, 1))(w, x),
        lambda f: torch.func.jacrev(f, argnums=(0, 1))(w, x),
        lambda f: torch.func.vjp(f, w, x)[1](torch.ones((2, 2), dtype=torch.float64)),
    ]
    for operation in operations:
        tree_close(operation(external.functional), operation(native.functional), leaves, 4e-12)
    gradient = torch.func.grad(lambda a, b: external.functional(a, b).sum(), argnums=(0, 1))(w, x)
    assert gradient[0][0] == 0 and torch.count_nonzero(gradient[1][..., 0]) == 0
    assert external.to_spec() == configuration


@pytest.mark.parametrize("method", ["adjoint", "parameter-shift"])
def test_jax_standard_products_jacobians_and_broadcast_reductions(method):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    configuration = spec()
    external = QuantumLayer.from_spec(
        configuration,
        backend="numpy",
        gradient_method=method,
        reduce_readout=True,
        batch_chunk_size=3,
    )
    native = QuantumLayer.from_spec(configuration, reduce_readout=True)
    w = jnp.array([0.2, 0.6, -0.3, 0.4], dtype=jnp.float32)
    x = jnp.linspace(-0.3, 0.5, 6, dtype=jnp.float32).reshape(2, 3)
    dw, dx = jnp.ones_like(w) * 0.2, jnp.ones_like(x) * -0.1
    operations = [
        lambda f: jax.jit(lambda a, b: jax.jvp(f, (a, b), (dw, dx)))(w, x),
        lambda f: jax.jit(jax.grad(lambda a, b: jnp.square(f(a, b)).sum(), argnums=(0, 1)))(w, x),
        lambda f: jax.jit(jax.jacfwd(f, argnums=(0, 1)))(w, x),
        lambda f: jax.jit(jax.jacrev(f, argnums=(0, 1)))(w, x),
        lambda f: jax.vjp(f, w, x)[1](jnp.ones((2, 2), dtype=jnp.float32)),
    ]
    for operation in operations:
        tree_close(operation(external), operation(native), jax.tree_util.tree_leaves, 3e-6)
    gradient = jax.grad(lambda a, b: external(a, b).sum(), argnums=(0, 1))(w, x)
    assert gradient[0][0] == 0 and np.count_nonzero(gradient[1][..., 0]) == 0
    assert external.checkpoint(w).spec == configuration


@pytest.mark.parametrize("framework", ["torch", "jax"])
@pytest.mark.parametrize("mode", ["weights", "inputs", "both", "nested", "generated"])
def test_transform_axes_stay_independent_of_model_broadcast_dimensions(framework, mode):
    package = pytest.importorskip(framework)
    if framework == "torch":
        from openqml.interfaces.torch import QuantumLayer

        dtype = dict(dtype=package.float64)

        def arr(a):
            return package.tensor(a, dtype=package.float64)

        stack, vmap, move = package.stack, package.func.vmap, lambda a, d: a.movedim(0, d)

        def leaves(t):
            return [v.detach().numpy() for v in package.utils._pytree.tree_leaves(t)]
    else:
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        dtype = {}

        def arr(a):
            return jnp.array(a, dtype=jnp.float32)

        stack, vmap, move = jnp.stack, package.vmap, lambda a, d: jnp.moveaxis(a, 0, d)
        leaves = package.tree_util.tree_leaves
    external = QuantumLayer.from_spec(
        spec(),
        backend="numpy",
        gradient_method="adjoint",
        reduce_readout=True,
        batch_chunk_size=3,
        **dtype,
    )
    native = QuantumLayer.from_spec(spec(), reduce_readout=True, **dtype)
    f, g = (external.functional, native.functional) if framework == "torch" else (external, native)
    w, x = arr([0.2, 0.6, -0.3, 0.4]), arr(np.linspace(-0.2, 0.4, 6).reshape(2, 3))
    axis = "in_dims" if framework == "torch" else "in_axes"
    if mode == "weights":
        values = move(stack((w, w * 1.3, w * 0.7)), 1)

        def transform(fn):
            return vmap(fn, **{axis: (1, None)})(values, x)
    elif mode == "inputs":
        values = move(stack((x, x * 1.3, x * 0.7)), 2)

        def transform(fn):
            return vmap(fn, **{axis: (None, 2)})(w, values)
    elif mode == "both":
        ww, xx = stack((w, w * 1.3, w * 0.7)), move(stack((x, x * 1.1, x * 0.8)), 1)

        def transform(fn):
            return vmap(fn, **{axis: (0, 1)})(ww, xx)
    elif mode == "nested":
        ww = stack((stack((w, w * 1.3, w * 0.7)), stack((w * 0.9, w * 0.6, w * 1.1))))

        def transform(fn):
            return vmap(vmap(fn, **{axis: (0, None)}), **{axis: (0, None)})(ww, x)
    else:
        ww = stack((w, w * 1.3)).reshape(2, 1, 4)
        xx = arr(np.linspace(-0.2, 0.4, 9).reshape(1, 3, 3))
        ensembles = stack((ww, ww * 1.2, ww * 0.8))

        def transform(fn):
            return vmap(fn, **{axis: (0, None)})(ensembles, xx)

    tree_close(transform(f), transform(g), leaves, 4e-6)


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_vmap_of_grad_and_jvp_and_reverse_of_vmap(framework):
    lib = pytest.importorskip(framework)
    if framework == "torch":
        from openqml.interfaces.torch import QuantumLayer

        def arr(a):
            return lib.tensor(a, dtype=lib.float64)

        opts, api, axis = {"dtype": lib.float64}, lib.func, "in_dims"

        def leaves(t):
            return [v.detach().numpy() for v in lib.utils._pytree.tree_leaves(t)]

        def square(a):
            return a.square()
    else:
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        def arr(a):
            return jnp.array(a, dtype=jnp.float32)

        opts, api, axis = {}, lib, "in_axes"
        leaves, square = lib.tree_util.tree_leaves, jnp.square
    e = QuantumLayer.from_spec(spec(), backend="numpy", gradient_method="adjoint", **opts)
    n = QuantumLayer.from_spec(spec(), **opts)
    f, g = (e.functional, n.functional) if framework == "torch" else (e, n)
    w, x = arr(np.tile([0.2, 0.6, -0.3, 0.4], (3, 1))), arr(np.linspace(-0.2, 0.4, 6).reshape(2, 3))
    tangent = w * 0 + 0.2
    operations = [
        lambda fn: api.vmap(
            api.grad(lambda a, b: square(fn(a, b)).sum(), argnums=(0, 1)), **{axis: (0, None)}
        )(w, x),
        lambda fn: api.vmap(lambda a, d: api.jvp(fn, (a, x), (d, x * 0 + 0.1)), **{axis: (0, 0)})(
            w, tangent
        ),
        lambda fn: api.grad(
            lambda a, b: square(api.vmap(fn, **{axis: (0, None)})(a, b)).sum(), argnums=(0, 1)
        )(w, x),
    ]
    for op in operations:
        tree_close(op(f), op(g), leaves, 4e-6)


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_noise_measurement_feedback_standard_first_order_transforms(framework):
    lib = pytest.importorskip(framework)
    if framework == "torch":
        from openqml.interfaces.torch import QuantumLayer

        opts, api = {"dtype": lib.float64}, lib.func

        def arr(a):
            return lib.tensor(a, dtype=lib.float64)

        def leaves(t):
            return [v.detach().numpy() for v in lib.utils._pytree.tree_leaves(t)]
    else:
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        opts, api = {}, lib

        def arr(a):
            return jnp.array(a, dtype=jnp.float32)

        leaves = lib.tree_util.tree_leaves
    external, native = (
        QuantumLayer.from_spec(noisy_spec(), backend="numpy", **opts),
        QuantumLayer.from_spec(noisy_spec(), **opts),
    )
    f, g = (external.functional, native.functional) if framework == "torch" else (external, native)
    w, x = arr([0.2, 0.6]), arr([[0.3], [-0.4]])
    for op in (
        lambda fn: api.jacfwd(fn, argnums=(0, 1))(w, x),
        lambda fn: api.jacrev(fn, argnums=(0, 1))(w, x),
    ):
        tree_close(op(f), op(g), leaves, 5e-6)


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_requested_nonlinear_higher_derivatives_fail_explicitly(framework):
    lib = pytest.importorskip(framework)
    if framework == "torch":
        from openqml.interfaces.torch import QuantumLayer

        layer = QuantumLayer.from_spec(
            spec(), backend="numpy", gradient_method="adjoint", dtype=lib.float64
        )
        w, x = lib.ones(4, dtype=lib.float64), lib.ones(3, dtype=lib.float64)
        with pytest.raises(ValidationError, match="first-order"):
            lib.func.hessian(lambda a: layer.functional(a, x).square().sum())(w)
    else:
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        layer = QuantumLayer.from_spec(spec(), backend="numpy", gradient_method="adjoint")
        w, x = jnp.ones(4, dtype=jnp.float32), jnp.ones(3, dtype=jnp.float32)
        with pytest.raises(ValidationError, match="first-order"):
            lib.hessian(lambda a: jnp.square(layer(a, x)).sum())(w)


class Counting:
    name = "external-transform-counting"
    capabilities = BackendCapabilities(native_expectations=True)

    def __init__(self):
        self.sizes = []

    def expectations_batch(self, circuits, observables, *, options):
        self.sizes.append(len(circuits))
        return expectations_batch(circuits, observables)


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_vmap_preserves_primal_and_derivative_execution_chunk_limits(framework):
    lib = pytest.importorskip(framework)
    backend = Counting()
    if framework == "torch":
        from openqml.interfaces.torch import QuantumLayer

        opts, api, axis = {"dtype": lib.float64}, lib.func, "in_dims"
        w, x = lib.ones((5, 4), dtype=lib.float64) * 0.3, lib.ones((4, 3), dtype=lib.float64) * 0.2
    else:
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        opts, api, axis = {}, lib, "in_axes"
        w, x = jnp.ones((5, 4), dtype=jnp.float32) * 0.3, jnp.ones((4, 3), dtype=jnp.float32) * 0.2
    layer = QuantumLayer.from_spec(
        spec(), backend=backend, batch_chunk_size=3, gradient_chunk_size=3, **opts
    )
    fn = layer.functional if framework == "torch" else layer
    result = api.vmap(fn, **{axis: (0, None)})(w, x)
    if framework == "jax":
        result.block_until_ready()
    assert max(backend.sizes) <= 3 and sum(backend.sizes) == 20
    backend.sizes.clear()
    grad = api.grad(lambda a: api.vmap(fn, **{axis: (0, None)})(a, x).sum())(w)
    if framework == "jax":
        grad.block_until_ready()
    assert backend.sizes and max(backend.sizes) <= 3


def test_jax_linearize_transpose_normal_products_and_callback_batching(monkeypatch):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    import openqml.interfaces._spec as common
    from openqml.interfaces.jax import QuantumLayer

    layer = QuantumLayer.from_spec(spec(), backend="numpy", gradient_method="adjoint")
    native = QuantumLayer.from_spec(spec())
    w, x = (
        jnp.array([0.2, 0.6, -0.3, 0.4], dtype=jnp.float32),
        jnp.linspace(-0.2, 0.4, 6, dtype=jnp.float32).reshape(2, 3),
    )
    y, linear = jax.linearize(layer, w, x)
    pull = jax.linear_transpose(linear, jnp.zeros_like(w), jnp.zeros_like(x))
    _, expected_pull = jax.vjp(native, w, x)
    dw, dx = w * 0 + 0.1, x * 0 - 0.2
    derivative = linear(dw, dx)
    reference = jax.jvp(native, (w, x), (dw, dx))[1]
    tree_close(jax.jit(pull)(derivative), expected_pull(reference), jax.tree_util.tree_leaves, 4e-6)
    # Fixed-point linear maps can themselves be differentiated in vector inputs.
    tree_close(
        jax.jvp(pull, (y * 0.3,), (y * 0.2,))[1], pull(y * 0.2), jax.tree_util.tree_leaves, 4e-6
    )
    transpose_pull = jax.linear_transpose(pull, y * 0)
    tree_close(transpose_pull((dw, dx))[0], derivative, jax.tree_util.tree_leaves, 4e-6)
    counts, forward, reverse = [], common.numeric_expectations, common.numeric_vjp

    def f(*args):
        counts.append("forward")
        return forward(*args)

    def b(*args, **kwargs):
        counts.append(("reverse", kwargs["weight_grad"], kwargs["input_grad"]))
        return reverse(*args, **kwargs)

    monkeypatch.setattr(common, "numeric_expectations", f)
    monkeypatch.setattr(common, "numeric_vjp", b)
    values = jnp.stack((w, w * 1.2, w * 0.8))
    value, gradient = jax.jit(
        jax.value_and_grad(lambda a: jnp.square(jax.vmap(layer, in_axes=(0, None))(a, x)).sum())
    )(values)
    jax.block_until_ready((value, gradient))
    assert counts.count("forward") == 1
    assert counts.count(("reverse", True, False)) == 1


@pytest.mark.parametrize("framework", ["torch", "jax"])
@pytest.mark.parametrize("source", ["qiskit", "cirq", "openqasm3"])
def test_imported_native_programs_use_standard_transforms(framework, source):
    pytest.importorskip("qiskit" if source == "qiskit" else "cirq" if source == "cirq" else "numpy")
    lib = pytest.importorskip(framework)
    configuration = spec()
    conversion = plan_conversion(configuration.circuit, target=source)
    native_program = conversion.export().value
    inputs = tuple(conversion.parameter_map.get(n, n) for n in configuration.input_parameters)
    weights = tuple(conversion.parameter_map.get(n, n) for n in configuration.weight_parameters)
    if framework == "torch":
        from openqml.interfaces.torch import QuantumLayer

        opts, api = {"dtype": lib.float64}, lib.func
        w, x = lib.ones(4, dtype=lib.float64) * 0.3, lib.ones(3, dtype=lib.float64) * 0.2

        def leaves(t):
            return [v.detach().numpy() for v in lib.utils._pytree.tree_leaves(t)]
    else:
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        opts, api = {}, lib
        w, x = jnp.ones(4, dtype=jnp.float32) * 0.3, jnp.ones(3, dtype=jnp.float32) * 0.2
        leaves = lib.tree_util.tree_leaves
    layer = QuantumLayer.from_program(
        native_program,
        inputs,
        weights,
        framework=source,
        observables=configuration.observables,
        backend="numpy",
        gradient_method="adjoint",
        **opts,
    )
    reference = QuantumLayer.from_spec(configuration, **opts)
    fn, expected = (
        (layer.functional, reference.functional) if framework == "torch" else (layer, reference)
    )
    tree_close(
        api.jvp(fn, (w, x), (w * 0 + 0.1, x * 0 - 0.2)),
        api.jvp(expected, (w, x), (w * 0 + 0.1, x * 0 - 0.2)),
        leaves,
        4e-6,
    )


@pytest.mark.parametrize("framework", ["torch", "jax"])
@pytest.mark.parametrize("backend", ["qiskit", "cirq", "pennylane"])
def test_sdk_execution_backends_support_standard_jvp_and_vjp(framework, backend):
    pytest.importorskip(backend)
    lib = pytest.importorskip(framework)
    if framework == "torch":
        from openqml.interfaces.torch import QuantumLayer

        options, api = {"dtype": lib.float64}, lib.func
        w, x = lib.ones(4, dtype=lib.float64) * 0.3, lib.ones(3, dtype=lib.float64) * 0.2

        def leaves(value):
            return [v.detach().numpy() for v in lib.utils._pytree.tree_leaves(value)]
    else:
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        options, api = {}, lib
        w, x = jnp.ones(4, dtype=jnp.float32) * 0.3, jnp.ones(3, dtype=jnp.float32) * 0.2
        leaves = lib.tree_util.tree_leaves
    e = QuantumLayer.from_spec(spec(), backend=backend, gradient_chunk_size=3, **options)
    n = QuantumLayer.from_spec(spec(), **options)
    f, g = (e.functional, n.functional) if framework == "torch" else (e, n)
    tree_close(
        api.jvp(f, (w, x), (w * 0 + 0.1, x * 0 - 0.2)),
        api.jvp(g, (w, x), (w * 0 + 0.1, x * 0 - 0.2)),
        leaves,
        4e-6,
    )
    cotangent = f(w, x) * 0 + 0.3
    tree_close(api.vjp(f, w, x)[1](cotangent), api.vjp(g, w, x)[1](cotangent), leaves, 4e-6)


@pytest.mark.parametrize("framework", ["torch", "jax"])
@pytest.mark.parametrize(
    "case", ["no_weights", "no_inputs", "no_active", "empty_batch", "empty_map"]
)
def test_empty_dimensions_and_eliminated_symbols_keep_transform_shapes(framework, case):
    lib = pytest.importorskip(framework)
    if case == "no_weights":
        configuration = LayerSpec(Circuit(1).ry(Parameter("x"), 0), ("x",), ())
        ww, xx = np.empty((0,)), np.array([[0.2], [0.4]])
    elif case == "no_inputs":
        configuration = LayerSpec(Circuit(1).ry(Parameter("w"), 0), (), ("w",))
        ww, xx = np.array([0.3]), np.empty((2, 0))
    elif case == "no_active":
        configuration = LayerSpec(
            Circuit(2).ry(Parameter("x"), 1).with_global_phase(Parameter("w")),
            ("x",),
            ("w",),
            (PauliString("ZI"),),
        )
        ww, xx = np.array([0.3]), np.array([[0.2], [0.4]])
    else:
        configuration = LayerSpec(Circuit(1).ry(Parameter("x") * Parameter("w"), 0), ("x",), ("w",))
        ww, xx = (
            np.array([0.3]),
            np.empty((0, 1)) if case == "empty_batch" else np.array([[0.2], [0.4]]),
        )
    if framework == "torch":
        from openqml.interfaces.torch import QuantumLayer

        options, api, axis = {"dtype": lib.float64}, lib.func, "in_dims"
        w, x = lib.tensor(ww, dtype=lib.float64), lib.tensor(xx, dtype=lib.float64)

        def leaves(value):
            return [v.detach().numpy() for v in lib.utils._pytree.tree_leaves(value)]
    else:
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        options, api, axis = {}, lib, "in_axes"
        w, x = jnp.array(ww, dtype=jnp.float32), jnp.array(xx, dtype=jnp.float32)
        leaves = lib.tree_util.tree_leaves
    external = QuantumLayer.from_spec(
        configuration, backend="numpy", gradient_method="adjoint", reduce_readout=True, **options
    )
    native = QuantumLayer.from_spec(configuration, reduce_readout=True, **options)
    f, g = (external.functional, native.functional) if framework == "torch" else (external, native)
    if case == "empty_map":
        ensembles = w.reshape((1,) + tuple(w.shape))[:0]
        tree_close(
            api.vmap(f, **{axis: (0, None)})(ensembles, x),
            g(w, x)[None, ...][:0],
            leaves,
            4e-6,
        )
    else:
        tree_close(
            api.jacfwd(f, argnums=(0, 1))(w, x), api.jacfwd(g, argnums=(0, 1))(w, x), leaves, 4e-6
        )
        tree_close(
            api.jacrev(f, argnums=(0, 1))(w, x), api.jacrev(g, argnums=(0, 1))(w, x), leaves, 4e-6
        )


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_cancelling_directions_submit_only_primal_work(framework, monkeypatch):
    lib = pytest.importorskip(framework)
    import openqml.gradients as gradients

    monkeypatch.setattr(
        gradients,
        "parameter_shift_jacobian",
        lambda *a, **kw: pytest.fail("Materialized a Jacobian"),
    )
    backend = Counting()
    x, w = Parameter("x"), Parameter("w")
    configuration = LayerSpec(Circuit(1).ry(x - w, 0).rx((x - w) ** 2, 0), ("x",), ("w",))
    if framework == "torch":
        from openqml.interfaces.torch import QuantumLayer

        layer = QuantumLayer.from_spec(configuration, backend=backend, dtype=lib.float64)
        w, x = lib.tensor([0.3], dtype=lib.float64), lib.tensor([[0.2], [0.4]], dtype=lib.float64)
        _, tangent = lib.func.jvp(layer.functional, (w, x), (lib.ones_like(w), lib.ones_like(x)))
        assert lib.count_nonzero(tangent) == 0
    else:
        import jax.numpy as jnp

        from openqml.interfaces.jax import QuantumLayer

        layer = QuantumLayer.from_spec(configuration, backend=backend)
        w, x = jnp.array([0.3], dtype=jnp.float32), jnp.array([[0.2], [0.4]], dtype=jnp.float32)
        y, tangent = lib.jit(
            lambda w, x: lib.jvp(layer, (w, x), (jnp.ones_like(w), jnp.ones_like(x)))
        )(w, x)
        lib.block_until_ready((y, tangent))
        assert np.count_nonzero(tangent) == 0
    assert sum(backend.sizes) == 2


def test_jax_full_hybrid_model_linearization_matches_native_normal_operator():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from openqml.interfaces.jax import QuantumLayer

    external = QuantumLayer.from_spec(spec(), backend="numpy", gradient_method="adjoint")
    native = QuantumLayer.from_spec(spec())
    params = {
        "encoder": jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 30,
        "bias": jnp.array([0.1, -0.1, 0.2], dtype=jnp.float32),
        "quantum": jnp.array([0.2, 0.6, -0.3, 0.4], dtype=jnp.float32),
        "head": jnp.array([0.3, -0.4], dtype=jnp.float32),
    }
    inputs = jnp.arange(24, dtype=jnp.float32).reshape(6, 4) / 20 - 0.4

    def model(layer, p):
        encoded = jnp.tanh(inputs @ p["encoder"] + p["bias"])
        return layer(p["quantum"], encoded) @ p["head"]

    _, linear = jax.linearize(lambda p: model(external, p), params)
    _, expected_linear = jax.linearize(lambda p: model(native, p), params)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    transpose = jax.linear_transpose(linear, zeros)
    expected_transpose = jax.linear_transpose(expected_linear, zeros)
    direction = jax.tree_util.tree_map(lambda a: a * 0 + 0.2, params)
    actual = jax.jit(lambda v: transpose(linear(v))[0])(direction)
    expected = expected_transpose(expected_linear(direction))[0]
    tree_close(actual, expected, jax.tree_util.tree_leaves, 5e-6)
