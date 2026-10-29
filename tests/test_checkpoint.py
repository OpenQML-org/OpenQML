import copy
from dataclasses import replace

import numpy as np
import pytest

import openqml as oq
from openqml import (
    Circuit,
    Hamiltonian,
    LayerCheckpoint,
    LayerSpec,
    Parameter,
    PauliString,
    ValidationError,
    convert,
)


def spec():
    x, a, b, unused = (Parameter(n) for n in ("feature", "alpha", "beta", "unused"))
    c = Circuit(2).rx(unused, 0).rx(-unused, 0).ry(x, 0).rx(b, 1).ry(a, 1).cx(0, 1)
    obs = (PauliString("IZ"), Hamiltonian((PauliString("ZI", 0.3), PauliString("IX", -0.2))))
    return LayerSpec(c, ("feature",), ("beta", "unused", "alpha"), obs)


def test_named_checkpoint_roundtrip_canonicalizes_mapping_order_and_float_precision():
    s = spec()
    checkpoint = LayerCheckpoint(s, {"alpha": 0.2, "unused": 0.1, "beta": -0.4}, "float32")
    assert checkpoint.weights == tuple(
        float(v) for v in np.array([-0.4, 0.1, 0.2], dtype=np.float32)
    )
    assert tuple(checkpoint.bindings) == s.weight_parameters
    payload = checkpoint.to_dict()
    payload["weights"] = dict(reversed(list(payload["weights"].items())))
    assert LayerCheckpoint.from_dict(payload) == checkpoint
    assert LayerCheckpoint.from_json(checkpoint.to_json()) == checkpoint
    payload["weights"]["alpha"] = 9
    assert checkpoint.bindings["alpha"] != 9
    assert checkpoint.weights_for(s) == checkpoint.weights


@pytest.mark.parametrize(
    "weights",
    [
        None,
        "123",
        {1, 2, 3},
        [1, 2],
        [1, 2, 3, 4],
        [1, np.inf, 2],
        [1, np.nan, 2],
        [True, 1, 2],
        [1j, 1, 2],
        [[1], [2], [3]],
        {"alpha": 1},
        {"alpha": 1, "beta": 2, "other": 3},
    ],
)
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValidationError):
        LayerCheckpoint(spec(), weights)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "int64", None, {}, True])
def test_invalid_checkpoint_dtypes_rejected(dtype):
    with pytest.raises(ValidationError):
        LayerCheckpoint(spec(), [1, 2, 3], dtype)


def test_checkpoint_range_empty_vectors_and_spec_validation():
    with pytest.raises(ValidationError, match="dtype range"):
        LayerCheckpoint(spec(), [1e100, 2, 3], "float32")
    with pytest.raises(ValidationError, match="LayerSpec"):
        LayerCheckpoint({}, ())
    empty = LayerCheckpoint(LayerSpec(Circuit(1).h(0), (), ()), (), "float32")
    assert empty.bindings == {} and LayerCheckpoint.from_json(empty.to_json()) == empty


@pytest.mark.parametrize(
    "case",
    ["extra", "missing", "schema", "version", "boolean_version", "weight_list", "backend", "spec"],
)
def test_malformed_checkpoint_payloads_rejected(case):
    p = LayerCheckpoint(spec(), [0.2, 0.3, 0.4]).to_dict()
    if case == "extra":
        p["extra"] = 1
    elif case == "missing":
        p.pop("dtype")
    elif case == "schema":
        p["schema"] = "pickle"
    elif case == "version":
        p["version"] = 2
    elif case == "boolean_version":
        p["version"] = True
    elif case == "weight_list":
        p["weights"] = [0.2, 0.3, 0.4]
    elif case == "backend":
        p["backend"] = "arbitrary.module"
    else:
        p["spec"]["circuit"]["schema"] = "other"
    with pytest.raises(ValidationError):
        LayerCheckpoint.from_dict(p)


def test_checkpoint_json_limits_duplicate_names_and_nested_limits():
    checkpoint = LayerCheckpoint(spec(), [0.2, 0.3, 0.4])
    text = checkpoint.to_json()
    for value in [
        None,
        b"{}",
        "[",
        text + "\ud800",
        text.replace('"beta": 0.2', '"beta":0.2,"beta":0.7'),
    ]:
        with pytest.raises(ValidationError):
            LayerCheckpoint.from_json(value)
    with pytest.raises(ValidationError, match="max_bytes"):
        LayerCheckpoint.from_json(text, max_bytes=len(text.encode()) - 1)
    with pytest.raises(ValidationError, match="max_operations"):
        LayerCheckpoint.from_json(text, max_operations=1)
    assert LayerCheckpoint.from_json(text, max_bytes=len(text.encode())) == checkpoint


@pytest.mark.parametrize("change", ["weights", "ownership", "circuit", "observables"])
def test_equal_shape_configurations_cannot_load_each_others_weights(change):
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    s = spec()
    changed = (
        replace(s, weight_parameters=tuple(reversed(s.weight_parameters)))
        if change == "weights"
        else replace(
            s, input_parameters=("alpha",), weight_parameters=("beta", "unused", "feature")
        )
        if change == "ownership"
        else replace(s, circuit=s.circuit.h(1))
        if change == "circuit"
        else replace(s, observables=tuple(reversed(s.observables)))
    )
    checkpoint = LayerCheckpoint(s, [0.2, 0.3, 0.4], "float64")
    layer = QuantumLayer.from_spec(changed, dtype=torch.float64)
    before = layer.weights.detach().clone()
    with pytest.raises(ValidationError, match="configuration"):
        layer.load_checkpoint(checkpoint)
    torch.testing.assert_close(before, layer.weights)


def test_torch_checkpoint_preserves_parameter_identity_dtype_and_optimization():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    layer = QuantumLayer.from_spec(spec(), optimize=True, dtype=torch.float64)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.03)
    x = torch.tensor([[0.2], [0.5]], dtype=torch.float64, requires_grad=True)
    for _ in range(5):
        optimizer.zero_grad()
        layer(x).square().mean().backward()
        optimizer.step()
    checkpoint = LayerCheckpoint.from_json(layer.checkpoint().to_json())
    assert checkpoint.spec == spec() and checkpoint.dtype == "float64"
    assert "unused" in checkpoint.bindings and layer.optimization.removed_parameters == ("unused",)
    restored = QuantumLayer.from_checkpoint(checkpoint, optimize=True)
    torch.testing.assert_close(restored(x), layer(x))
    assert restored.weights.dtype == torch.float64
    assert QuantumLayer.from_checkpoint(checkpoint, dtype=None).weights.dtype == torch.float64
    original = layer.weights
    state = copy.deepcopy(optimizer.state_dict())
    layer.load_checkpoint(checkpoint)
    assert layer.weights is original and optimizer.param_groups[0]["params"][0] is original
    assert optimizer.state_dict()["param_groups"] == state["param_groups"]
    assert not restored.weights.data_ptr() == layer.weights.data_ptr()


def test_precision_mismatch_overflow_and_invalid_export_do_not_mutate_torch_weights():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    layer = QuantumLayer.from_spec(spec(), dtype=torch.float32)
    checkpoint = LayerCheckpoint(spec(), [0.2, 0.3, 0.4], "float64")
    with pytest.raises(ValidationError, match="precision"):
        layer.load_checkpoint(checkpoint)
    layer.load_checkpoint(checkpoint, allow_cast=True)
    before = layer.weights.detach().clone()
    with pytest.raises(ValidationError, match="dtype range"):
        layer.load_checkpoint(LayerCheckpoint(spec(), [1e100, 0, 0]), allow_cast=True)
    torch.testing.assert_close(layer.weights, before)
    cast = QuantumLayer.from_checkpoint(checkpoint, dtype=torch.float32)
    assert cast.weights.dtype == torch.float32
    with pytest.raises(ValidationError):
        layer.load_checkpoint(checkpoint, allow_cast=1)
    for value in [
        np.ones((1, 3)),
        np.ones(3, dtype=int),
        np.full(3, np.nan),
        torch.ones(3, dtype=torch.float16),
    ]:
        with pytest.raises(ValidationError):
            layer.checkpoint(value)
    for value in [None, {}]:
        with pytest.raises(ValidationError):
            QuantumLayer.from_checkpoint(value)


def test_trained_torch_weights_restore_in_jax_flax_and_back_with_gradients():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    pytest.importorskip("flax")
    from openqml.interfaces.flax import QuantumLayer as FlaxLayer
    from openqml.interfaces.jax import QuantumLayer as JaxLayer
    from openqml.interfaces.torch import QuantumLayer as TorchLayer

    layer = TorchLayer.from_spec(spec(), dtype=torch.float64, optimize=True)
    x = torch.tensor([[0.2], [0.5]], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.03)
    for _ in range(3):
        optimizer.zero_grad()
        layer(x).square().mean().backward()
        optimizer.step()
    checkpoint = LayerCheckpoint.from_json(layer.checkpoint().to_json())
    y = layer(x)
    dw, dx = torch.autograd.grad(y.sum(), (layer.weights, x))
    with oq.jax.enable_x64():
        jl, jw = JaxLayer.from_checkpoint(checkpoint, optimize=True)
        jx = jnp.asarray(x.detach().numpy())
        np.testing.assert_allclose(jax.jit(jl)(jw, jx), y.detach().numpy(), atol=1e-12)
        grads = jax.jit(jax.grad(lambda w, x: jl(w, x).sum(), argnums=(0, 1)))(jw, jx)
        np.testing.assert_allclose(grads[0], dw.numpy(), atol=1e-12)
        np.testing.assert_allclose(grads[1], dx.numpy(), atol=1e-12)
        fm, params = FlaxLayer.from_checkpoint(
            LayerCheckpoint.from_json(jl.checkpoint(jw).to_json()), optimize=True
        )
        np.testing.assert_allclose(jax.jit(fm.apply)(params, jx), y.detach().numpy(), atol=1e-12)
        fg = jax.grad(lambda p: fm.apply(p, jx).sum())(params)
        np.testing.assert_allclose(fg["params"]["weights"], dw.numpy(), atol=1e-12)
        back = TorchLayer.from_checkpoint(fm.checkpoint(params["params"]["weights"]))
        torch.testing.assert_close(back(x), y)
        with pytest.raises(ValidationError):
            jl.checkpoint()
        with pytest.raises(ValidationError):
            jax.jit(lambda w: jl.checkpoint(w).weights)(jw)


def test_jax_precision_requires_explicit_cast_and_flax_checks_destination():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    pytest.importorskip("flax")
    from openqml.interfaces.flax import QuantumLayer as FlaxLayer
    from openqml.interfaces.jax import QuantumLayer as JaxLayer

    checkpoint = LayerCheckpoint(spec(), [0.2, 0.3, 0.4], "float64")
    with oq.jax.enable_x64(False):
        with pytest.raises(ValidationError, match="x64"):
            JaxLayer.from_checkpoint(checkpoint)
        with pytest.raises(ValidationError, match="x64"):
            FlaxLayer.from_checkpoint(checkpoint)
        jl, jw = JaxLayer.from_checkpoint(checkpoint, dtype=jnp.float32)
        fm, params = FlaxLayer.from_checkpoint(checkpoint, dtype=jnp.float32)
        assert jw.dtype == params["params"]["weights"].dtype == jnp.float32
        with pytest.raises(ValidationError, match="precision"):
            fm.load_checkpoint(checkpoint)
        assert (
            fm.load_checkpoint(checkpoint, allow_cast=True)["params"]["weights"].dtype
            == jnp.float32
        )
        with pytest.raises(ValidationError):
            jl.load_checkpoint(checkpoint, dtype="not-a-dtype")
        with pytest.raises(ValidationError):
            jl.load_checkpoint(checkpoint, dtype=jnp.float16)
        with pytest.raises(ValidationError):
            jl.load_checkpoint({}, dtype=jnp.float32)


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "openqasm3"])
def test_imported_program_checkpoint_remains_exportable_after_restore(framework):
    torch = pytest.importorskip("torch")
    if framework in {"qiskit", "cirq"}:
        pytest.importorskip(framework)
    from openqml.interfaces.torch import QuantumLayer

    native = convert(spec().circuit, target=framework).value
    layer = QuantumLayer.from_program(
        native,
        spec().input_parameters,
        spec().weight_parameters,
        framework=framework,
        observables=spec().observables,
        dtype=torch.float64,
        optimize=True,
    )
    saved = LayerCheckpoint.from_json(layer.checkpoint().to_json())
    restored = QuantumLayer.from_checkpoint(saved, optimize=True)
    x = torch.tensor([[0.3]], dtype=torch.float64)
    torch.testing.assert_close(restored(x), layer(x))
    assert convert(restored.circuit, target=framework).value is not None
