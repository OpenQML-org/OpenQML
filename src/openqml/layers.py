"""Portable quantum layer definitions."""

import json
from collections.abc import Mapping, Set
from dataclasses import dataclass

import numpy as np

from .errors import ValidationError
from .ir import Circuit, integer
from .observables import Hamiltonian, PauliString, z_observables
from .parameters import parameter_name, real


def _load_json(text, max_bytes):
    integer(max_bytes, "max_bytes", 1)
    if not isinstance(text, str):
        raise ValidationError("Layer JSON must be text.")
    try:
        if len(text) > max_bytes or len(text.encode("utf-8")) > max_bytes:
            raise ValidationError("Layer JSON exceeds max_bytes.")

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValidationError(f"Duplicate JSON field {key!r}.")
                result[key] = value
            return result

        return json.loads(text, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, RecursionError, UnicodeError) as exc:
        raise ValidationError("Invalid layer JSON.") from exc


def _parameter_names(values):
    if isinstance(values, (str, bytes, Set, Mapping)):
        raise ValidationError("Parameter names require an ordered iterable.")
    try:
        return tuple(parameter_name(getattr(p, "name", p)) for p in values)
    except TypeError as exc:
        raise ValidationError("Parameter names require an ordered iterable.") from exc


def validate_layer(circuit, inputs, weights, observables):
    if not isinstance(circuit, Circuit):
        raise ValidationError("A quantum layer requires an OpenQML Circuit; use from_program().")
    inputs, weights = _parameter_names(inputs), _parameter_names(weights)
    if len(set(inputs + weights)) != len(inputs) + len(weights):
        raise ValidationError("Input and weight parameter names must be distinct and unique.")
    if set(inputs + weights) != set(circuit.parameters):
        raise ValidationError("Every circuit parameter must belong to exactly one input or weight.")
    if isinstance(observables, (str, bytes, Set, Mapping)):
        raise ValidationError("Observables require an ordered iterable.")
    try:
        obs = z_observables(circuit.num_qubits) if observables is None else tuple(observables)
    except TypeError as exc:
        raise ValidationError("Observables require an ordered iterable.") from exc
    if not obs:
        raise ValidationError("A quantum layer needs at least one output observable.")
    for ob in obs:
        terms = ob.terms if isinstance(ob, Hamiltonian) else (ob,)
        if any(
            not isinstance(p, PauliString) or len(p.paulis) != circuit.num_qubits for p in terms
        ):
            raise ValidationError("Observable width must equal the circuit width.")
    return inputs, weights, obs


@dataclass(frozen=True)
class LayerSpec:
    """Portable circuit, ordered symbol ownership and readouts; excludes model weights/backends."""

    circuit: Circuit
    input_parameters: tuple[str, ...]
    weight_parameters: tuple[str, ...]
    observables: tuple[PauliString | Hamiltonian, ...] | None = None

    def __post_init__(self):
        values = validate_layer(
            self.circuit, self.input_parameters, self.weight_parameters, self.observables
        )
        for name, value in zip(
            ("input_parameters", "weight_parameters", "observables"), values, strict=True
        ):
            object.__setattr__(self, name, value)

    @classmethod
    def from_program(
        cls,
        program,
        input_parameters,
        weight_parameters=None,
        observables=None,
        *,
        framework=None,
        **import_options,
    ):
        from .program import as_circuit

        if framework in {"json", "openqasm2", "openqasm3", "qasm2", "qasm3"}:
            from .conversion import convert

            circuit = convert(
                program, source=framework, target="ir", source_options=import_options
            ).value
        else:
            circuit = as_circuit(program, framework=framework, **import_options)
        inputs = _parameter_names(input_parameters)
        weights = (
            tuple(p for p in circuit.parameters if p not in inputs)
            if weight_parameters is None
            else weight_parameters
        )
        return cls(circuit, inputs, weights, observables)

    def to_dict(self):
        def term(p):
            return {"paulis": p.paulis, "coefficient": p.coefficient}

        return {
            "schema": "openqml.layer",
            "version": 1,
            "circuit": self.circuit.to_dict(),
            "input_parameters": list(self.input_parameters),
            "weight_parameters": list(self.weight_parameters),
            "observables": [
                {"type": "hamiltonian", "terms": [term(p) for p in ob.terms]}
                if isinstance(ob, Hamiltonian)
                else {"type": "pauli", **term(ob)}
                for ob in self.observables
            ],
        }

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, payload, *, max_operations=100_000, max_depth=64):
        integer(max_operations, "max_operations")
        integer(max_depth, "max_depth")
        fields = {
            "schema",
            "version",
            "circuit",
            "input_parameters",
            "weight_parameters",
            "observables",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValidationError("Invalid layer specification fields.")
        if (
            payload["schema"] != "openqml.layer"
            or type(payload["version"]) is not int
            or payload["version"] != 1
        ):
            raise ValidationError("Unsupported layer specification schema or version.")
        if any(
            not isinstance(payload[k], list)
            for k in ("input_parameters", "weight_parameters", "observables")
        ):
            raise ValidationError("Layer parameters and observables must be JSON arrays.")
        pending, count = [(payload["circuit"], 0)], 0
        while pending:
            body, depth = pending.pop()
            if depth > min(max_depth, 128):
                raise ValidationError("Layer circuit exceeds max_depth.")
            if not isinstance(body, dict) or not isinstance(body.get("operations"), list):
                raise ValidationError("Invalid layer circuit.")
            for op in body["operations"]:
                count += 1
                if count > max_operations:
                    raise ValidationError("Layer circuit exceeds max_operations.")
                if not isinstance(op, dict) or not isinstance(op.get("blocks", []), list):
                    raise ValidationError("Invalid layer operation.")
                pending.extend((block, depth + 1) for block in op.get("blocks", []))

        def term(value):
            if not isinstance(value, dict) or set(value) != {"paulis", "coefficient"}:
                raise ValidationError("Invalid Pauli term fields.")
            return PauliString(value["paulis"], value["coefficient"])

        observables = []
        for ob in payload["observables"]:
            if not isinstance(ob, dict):
                raise ValidationError("Invalid observable fields.")
            if ob.get("type") == "pauli" and set(ob) == {"type", "paulis", "coefficient"}:
                observables.append(term({k: v for k, v in ob.items() if k != "type"}))
            elif (
                ob.get("type") == "hamiltonian"
                and set(ob) == {"type", "terms"}
                and isinstance(ob["terms"], list)
            ):
                observables.append(Hamiltonian(tuple(term(p) for p in ob["terms"])))
            else:
                raise ValidationError("Unsupported observable specification.")
        return cls(
            Circuit.from_dict(payload["circuit"]),
            payload["input_parameters"],
            payload["weight_parameters"],
            tuple(observables),
        )

    @classmethod
    def from_json(cls, text, *, max_bytes=2_000_000, max_operations=100_000, max_depth=64):
        return cls.from_dict(
            _load_json(text, max_bytes), max_operations=max_operations, max_depth=max_depth
        )


@dataclass(frozen=True)
class LayerCheckpoint:
    """Portable layer definition and named weights; excludes optimizer state and backends."""

    spec: LayerSpec
    weights: tuple[float, ...]
    dtype: str = "float64"

    def __post_init__(self):
        if not isinstance(self.spec, LayerSpec):
            raise ValidationError("A checkpoint requires a LayerSpec.")
        if not isinstance(self.dtype, str) or self.dtype not in {"float32", "float64"}:
            raise ValidationError("Checkpoint dtype must be float32 or float64.")
        values = self.weights
        if isinstance(values, Mapping):
            if set(values) != set(self.spec.weight_parameters):
                raise ValidationError("Checkpoint weights must match the named weight parameters.")
            values = tuple(values[name] for name in self.spec.weight_parameters)
        elif isinstance(values, (str, bytes, Set)):
            raise ValidationError("Checkpoint weights require an ordered vector or name mapping.")
        try:
            values = tuple(real(value) for value in values)
        except TypeError as exc:
            raise ValidationError("Checkpoint weights require a finite real vector.") from exc
        if len(values) != len(self.spec.weight_parameters):
            raise ValidationError("Checkpoint weight count differs from the layer specification.")
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(values, dtype=self.dtype)
        if not np.isfinite(array).all():
            raise ValidationError("Checkpoint weights exceed the declared dtype range.")
        object.__setattr__(self, "weights", tuple(float(value) for value in array))

    @property
    def bindings(self):
        return dict(zip(self.spec.weight_parameters, self.weights, strict=True))

    def weights_for(self, spec):
        """Reject topology, ownership or readout mismatches before loading any weights."""
        if not isinstance(spec, LayerSpec) or spec != self.spec:
            raise ValidationError("Checkpoint configuration differs from this layer.")
        return self.weights

    def to_dict(self):
        # Store values by name so JSON object ordering cannot reassign weights.
        return {
            "schema": "openqml.layer_checkpoint",
            "version": 1,
            "spec": self.spec.to_dict(),
            "weights": self.bindings,
            "dtype": self.dtype,
        }

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, payload, *, max_operations=100_000, max_depth=64):
        fields = {"schema", "version", "spec", "weights", "dtype"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValidationError("Invalid layer checkpoint fields.")
        if (
            payload["schema"] != "openqml.layer_checkpoint"
            or type(payload["version"]) is not int
            or payload["version"] != 1
        ):
            raise ValidationError("Unsupported layer checkpoint schema or version.")
        if not isinstance(payload["weights"], dict):
            raise ValidationError("Checkpoint weights must be a JSON name mapping.")
        return cls(
            LayerSpec.from_dict(
                payload["spec"], max_operations=max_operations, max_depth=max_depth
            ),
            payload["weights"],
            payload["dtype"],
        )

    @classmethod
    def from_json(cls, text, *, max_bytes=2_000_000, max_operations=100_000, max_depth=64):
        return cls.from_dict(
            _load_json(text, max_bytes), max_operations=max_operations, max_depth=max_depth
        )
