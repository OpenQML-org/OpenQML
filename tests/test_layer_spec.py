import copy
import json

import numpy as np
import pytest

from openqml import (
    Circuit,
    ClassicalCondition,
    Hamiltonian,
    LayerSpec,
    Parameter,
    PauliString,
    ValidationError,
    convert,
    expectations,
    hardware_efficient,
)


def specification():
    x, w, a, b = (Parameter(n) for n in ("feature", "weight", "unused_input", "unused_weight"))
    circuit = Circuit(2).ry(a, 0).ry(-a, 0).rx(b, 1).rx(-b, 1).ry(x * w, 0).cx(0, 1)
    obs = (
        PauliString("ZZ", -0.5),
        Hamiltonian((PauliString("ZI", 0.7), PauliString("IX", -0.2))),
        PauliString("IZ", 0),
    )
    return LayerSpec(circuit, ("unused_input", "feature"), ("unused_weight", "weight"), obs)


def test_template_compatibility_and_immutable_normalization():
    from openqml.templates import LayerSpec as OldLayerSpec

    assert OldLayerSpec is LayerSpec
    spec = hardware_efficient(2, 1)
    assert spec.observables == (PauliString("ZI"), PauliString("IZ"))
    rebuilt = LayerSpec(
        spec.circuit, (Parameter(n) for n in spec.input_parameters), list(spec.weight_parameters)
    )
    assert rebuilt == spec and hash(rebuilt) == hash(spec)


def test_layer_json_roundtrip_preserves_order_coefficients_and_independent_copies():
    spec = specification()
    payload = spec.to_dict()
    assert LayerSpec.from_dict(payload) == LayerSpec.from_json(spec.to_json()) == spec
    payload["input_parameters"].reverse()
    assert spec.input_parameters == ("unused_input", "feature")
    restored = LayerSpec.from_dict(payload)
    assert restored.input_parameters == tuple(reversed(spec.input_parameters))
    rows = {"unused_input": 0.1, "feature": 0.4, "unused_weight": 0.2, "weight": 0.7}
    np.testing.assert_allclose(
        expectations(spec.circuit, spec.observables, rows),
        expectations(restored.circuit, restored.observables, rows),
        atol=1e-12,
    )


def test_effects_matrices_branch_phases_and_classical_storage_survive():
    x = Parameter("x")
    circuit = (
        Circuit(2)
        .unitary(np.array([[0, 1j], [1j, 0]]), 0)
        .ry(x, 1)
        .measure(0, 3)
        .if_else(
            ClassicalCondition((3,), 1), Circuit(2).amplitude_damping(0.2, 1).with_global_phase(x)
        )
    )
    spec = LayerSpec(circuit, ("x",), (), (PauliString("IZ"),))
    restored = LayerSpec.from_json(spec.to_json())
    assert restored == spec
    np.testing.assert_allclose(
        expectations(circuit, spec.observables, {"x": 0.4}),
        expectations(restored.circuit, restored.observables, {"x": 0.4}),
        atol=1e-12,
    )


@pytest.mark.parametrize("values", ["feature", {"feature"}, {"feature": 1}, None, [False], [[]]])
def test_parameter_ownership_requires_valid_ordered_names(values):
    with pytest.raises(ValidationError):
        LayerSpec(Circuit(1).ry(Parameter("feature"), 0), values, ())


@pytest.mark.parametrize(
    "value", [[], {PauliString("Z")}, "Z", {}, [PauliString("ZZ")], [object()], 1]
)
def test_invalid_readout_definitions_are_rejected(value):
    with pytest.raises(ValidationError):
        LayerSpec(Circuit(1), (), (), value)


@pytest.mark.parametrize(
    "case",
    [
        "schema",
        "version",
        "boolean_version",
        "extra",
        "missing",
        "input_order",
        "weights",
        "readout_type",
        "term_fields",
        "coefficient",
        "empty_hamiltonian",
        "wrong_width",
        "backend",
    ],
)
def test_malformed_layer_payloads_are_rejected(case):
    payload = specification().to_dict()
    if case == "schema":
        payload["schema"] = "other"
    elif case == "version":
        payload["version"] = 2
    elif case == "boolean_version":
        payload["version"] = True
    elif case == "extra":
        payload["extra"] = 1
    elif case == "missing":
        payload.pop("circuit")
    elif case == "input_order":
        payload["input_parameters"] = "feature"
    elif case == "weights":
        payload["weight_parameters"] = []
    elif case == "readout_type":
        payload["observables"][0]["type"] = "pickle"
    elif case == "term_fields":
        payload["observables"][1]["terms"][0]["extra"] = 1
    elif case == "coefficient":
        payload["observables"][0]["coefficient"] = float("nan")
    elif case == "empty_hamiltonian":
        payload["observables"][1]["terms"] = []
    elif case == "wrong_width":
        payload["observables"][0]["paulis"] = "Z"
    else:
        payload["backend"] = "arbitrary.module"
    with pytest.raises(ValidationError):
        LayerSpec.from_dict(payload)


def test_json_rejects_duplicate_nested_fields_invalid_text_and_byte_limits():
    text = specification().to_json()
    for value in [
        None,
        b"{}",
        "[",
        text.replace('"schema": "openqml.layer"', '"schema":"openqml.layer","schema":"other"'),
        text.replace('"coefficient": -0.5', '"coefficient": -0.5, "coefficient": 0.2'),
        text + "\ud800",
    ]:
        with pytest.raises(ValidationError):
            LayerSpec.from_json(value)
    with pytest.raises(ValidationError, match="max_bytes"):
        LayerSpec.from_json(text, max_bytes=len(text.encode()) - 1)
    assert LayerSpec.from_json(text, max_bytes=len(text.encode())) == specification()
    name = "\u89d2\u5ea6"
    unicode_spec = LayerSpec(Circuit(1).ry(Parameter(name), 0), (name,), ())
    raw = json.dumps(unicode_spec.to_dict(), ensure_ascii=False)
    with pytest.raises(ValidationError, match="max_bytes"):
        LayerSpec.from_json(raw, max_bytes=len(raw))


def test_decoding_budgets_precede_ir_construction(monkeypatch):
    from openqml import Circuit

    payload = specification().to_dict()
    nested = Circuit(2).if_else(ClassicalCondition((0,), 0), specification().circuit)
    deep = LayerSpec(
        nested, specification().input_parameters, specification().weight_parameters
    ).to_dict()

    def fail(*args, **kwargs):
        raise AssertionError("IR decoding ran before preflight")

    monkeypatch.setattr(Circuit, "from_dict", fail)
    with pytest.raises(ValidationError, match="max_operations"):
        LayerSpec.from_dict(payload, max_operations=1)
    with pytest.raises(ValidationError, match="max_depth"):
        LayerSpec.from_dict(deep, max_depth=0)


@pytest.mark.parametrize("framework", ["qiskit", "cirq", "openqasm3", "pennylane"])
def test_specifications_capture_native_frameworks(framework):
    if framework in {"qiskit", "cirq", "pennylane"}:
        pytest.importorskip(framework)
    c = specification().circuit
    if framework == "pennylane":
        c = c.bind(dict.fromkeys(c.parameters, 0.3))
    native = convert(c, target=framework).value
    spec = LayerSpec.from_program(
        native,
        () if framework == "pennylane" else specification().input_parameters,
        framework=framework,
        **({"parser": "builtin"} if framework == "openqasm3" else {}),
    )
    assert set(spec.weight_parameters) == (
        set() if framework == "pennylane" else set(specification().weight_parameters)
    )
    assert LayerSpec.from_json(spec.to_json()) == spec


def test_optimized_torch_spec_retains_source_ownership_and_checkpoint():
    torch = pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    spec = specification()
    layer = QuantumLayer.from_spec(spec, optimize=True, dtype=torch.float64)
    restored = QuantumLayer.from_spec(
        LayerSpec.from_json(layer.to_spec().to_json()), dtype=torch.float64
    )
    restored.load_state_dict(copy.deepcopy(layer.state_dict()))
    assert layer.to_spec() == restored.to_spec() == spec
    assert len(layer.circuit.operations) < len(restored.circuit.operations)
    x = torch.tensor([[0.1, 0.4], [-0.2, 0.8]], dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(layer(x), restored(x))
    actual = torch.autograd.grad(layer(x).sum(), (x, layer.weights))
    expected = torch.autograd.grad(restored(x).sum(), (x, restored.weights))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    with pytest.raises(ValidationError):
        QuantumLayer.from_spec(spec.to_dict())


def test_torch_jax_flax_share_spec_weights_outputs_and_derivatives():
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    pytest.importorskip("flax")
    from openqml.interfaces.flax import QuantumLayer as FlaxLayer
    from openqml.interfaces.jax import QuantumLayer as JaxLayer
    from openqml.interfaces.torch import QuantumLayer as TorchLayer

    spec = LayerSpec.from_json(specification().to_json())
    tx = torch.tensor([[0.1, 0.4], [-0.2, 0.8]], dtype=torch.float64, requires_grad=True)
    tw = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
    tl = TorchLayer.from_spec(spec, dtype=torch.float64, optimize=True)
    output = torch.func.functional_call(tl, {"weights": tw}, (tx,))
    dw, dx = torch.autograd.grad(output.sum(), (tw, tx))
    with jax.enable_x64():
        jl = JaxLayer.from_spec(spec, optimize=True)
        jx, jw = jnp.asarray(tx.detach().numpy()), jnp.asarray(tw.detach().numpy())
        np.testing.assert_allclose(jax.jit(jl)(jw, jx), output.detach().numpy(), atol=1e-12)
        grad = jax.jit(jax.grad(lambda w, x: jl(w, x).sum(), argnums=(0, 1)))(jw, jx)
        np.testing.assert_allclose(grad[0], dw.numpy(), atol=1e-12)
        np.testing.assert_allclose(grad[1], dx.numpy(), atol=1e-12)
        fl = FlaxLayer.from_spec(spec, optimize=True, dtype=jnp.float64)
        params = {"params": {"weights": jw}}
        np.testing.assert_allclose(
            jax.jit(fl.apply)(params, jx), output.detach().numpy(), atol=1e-12
        )
        fg = jax.grad(lambda p: fl.apply(p, jx).sum())(params)
        np.testing.assert_allclose(fg["params"]["weights"], dw.numpy(), atol=1e-12)
        assert fl.to_spec() == jl.to_spec() == tl.to_spec() == spec
        assert FlaxLayer.from_program(
            spec.circuit, spec.input_parameters
        ).weight_parameters == tuple(
            p for p in spec.circuit.parameters if p not in spec.input_parameters
        )


def test_shared_factories_validate_order_and_support_flax_templates():
    pytest.importorskip("torch")
    from openqml.interfaces.torch import QuantumLayer

    with pytest.raises(ValidationError):
        QuantumLayer.from_program(specification().circuit, set(specification().input_parameters))
    pytest.importorskip("flax")
    from openqml.interfaces.flax import QuantumLayer as FlaxLayer

    assert FlaxLayer.hardware_efficient(2, 1).to_spec() == hardware_efficient(2, 1)
