"""Composable structured operators for the shared hybrid-network program.

Only specifications are serialized. Custom operator implementations must be
registered explicitly in every process; loading a checkpoint never runs code.
"""

import itertools
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType

import numpy as np

from .errors import ResourceError, ValidationError
from .ir import integer

_OPERATORS = {}
_LOCK = RLock()


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(v) for key, v in value.items()})
    return tuple(_freeze(v) for v in value) if isinstance(value, list) else value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(v) for key, v in value.items()}
    return [_thaw(v) for v in value] if isinstance(value, tuple) else value


class NetworkOperator:
    """Backend-independent extension contract; use the supplied execution context.

    declare receives projection(name,in_features,out_features) and
    parameter(name,shape,role,initialization,scale) for additional parameters.
    apply receives context.kernel(features, role), project,
    array helpers, the block's parameters, masks and graph fields. Each role
    gets an independent encoder and quantum parameter vector.
    """

    version = 1
    terminal = False
    stateful = False
    native_state = False
    uses_decode = True

    def validate(self, config):
        if config:
            raise ValidationError("This operator has no configuration fields.")

    def roles(self, config):
        return ("",)

    def input_width(self, hidden, config):
        return hidden

    def declare(self, hidden, readouts, config, projection, parameter):
        pass

    def validate_metadata(self, metadata, config):
        pass

    def validate_cell(self, spec, config):
        pass

    def apply(self, context):
        return context.project("decode", context.kernel(context.inputs))

    def row_chunks(self, metadata, config):
        return (metadata["sites"],)


def register_network_operator(name, operator):
    """Register a namespaced operator without replacing an existing implementation."""
    if not isinstance(name, str) or not name or not isinstance(operator, NetworkOperator):
        raise ValidationError("Provide an operator name and NetworkOperator instance.")
    integer(operator.version, "operator version", 1)
    with _LOCK:
        if name in _OPERATORS:
            raise ValidationError(f"Network operator {name!r} is already registered.")
        _OPERATORS[name] = operator


def list_network_operators():
    with _LOCK:
        return tuple(sorted(_OPERATORS))


@dataclass(frozen=True)
class OperatorSpec:
    kind: str = "pointwise"
    config: object = field(default_factory=dict)
    version: int = 1

    def __post_init__(self):
        if not isinstance(self.kind, str):
            raise ValidationError("Operator kind must be a registered name.")
        object.__setattr__(self, "version", integer(self.version, "operator version", 1))
        with _LOCK:
            operator = _OPERATORS.get(self.kind)
        if operator is None or self.version != operator.version:
            raise ValidationError(
                f"Register network operator {self.kind!r} version {self.version}."
            )
        try:
            config = json.loads(json.dumps(dict(self.config), allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValidationError("Operator configuration must contain JSON values.") from exc
        operator.validate(config)
        object.__setattr__(self, "config", _freeze(config))

    @property
    def implementation(self):
        return _OPERATORS[self.kind]

    def to_dict(self):
        return {"kind": self.kind, "version": self.version, "config": _thaw(self.config)}

    @classmethod
    def convolution(
        cls, kernel_size=3, *, spatial_dims=1, dilation=1, padding="same", chunk_size=64
    ):
        return cls(
            "convolution",
            dict(
                kernel_size=kernel_size,
                spatial_dims=spatial_dims,
                dilation=dilation,
                padding=padding,
                chunk_size=chunk_size,
            ),
        )

    @classmethod
    def attention(
        cls,
        *,
        heads=1,
        key_features=8,
        mode="gaussian",
        causal=False,
        positional=True,
        chunk_size=64,
    ):
        return cls(
            "attention",
            dict(
                heads=heads,
                key_features=key_features,
                mode=mode,
                causal=causal,
                positional=positional,
                chunk_size=chunk_size,
            ),
        )

    @classmethod
    def graph(cls, *, edge_features=0, geometric=False, aggregation="mean", chunk_size=64):
        return cls(
            "graph",
            dict(
                edge_features=edge_features,
                geometric=geometric,
                aggregation=aggregation,
                chunk_size=chunk_size,
            ),
        )

    @classmethod
    def state_attention(
        cls, *, heads=1, causal=False, positional=True, chunk_size=64, depolarization=0.01
    ):
        """One quantum feature bank and a learned physical reduced-state kernel."""
        return cls(
            "state-attention",
            dict(
                heads=heads,
                causal=causal,
                positional=positional,
                chunk_size=chunk_size,
                depolarization=depolarization,
            ),
        )

    @classmethod
    def pair_graph(cls, *, edge_features=0, geometric=False, chunk_size=64):
        """Persistent ordered-pair states with a factorized higher-order update."""
        return cls(
            "pair-graph",
            dict(edge_features=edge_features, geometric=geometric, chunk_size=chunk_size),
        )

    @classmethod
    def quantum_kan(cls, *, chunk_size=64):
        return cls("quantum-kan", dict(chunk_size=chunk_size))

    @classmethod
    def memory(cls, *, gated=True, scan="sequential", max_work=1_000_000_000):
        return cls("quantum-memory", dict(gated=gated, scan=scan, max_work=max_work))

    @classmethod
    def born_readout(
        cls,
        visible_qubits=1,
        *,
        memory=False,
        gated=True,
        depolarization=0.001,
        chunk_size=64,
        measurement="basis",
        memory_qubits=1,
        povm_regularization=0.01,
        scan="sequential",
        max_work=1_000_000_000,
    ):
        config = dict(visible_qubits=visible_qubits, depolarization=depolarization)
        if memory:
            config.update(
                gated=gated,
                measurement=measurement,
                memory_qubits=memory_qubits,
                povm_regularization=povm_regularization,
                scan=scan,
                max_work=max_work,
            )
        else:
            config["chunk_size"] = chunk_size
        return cls("born-memory" if memory else "born", config)


def _fields(config, required):
    if set(config) != set(required):
        raise ValidationError(f"Operator fields must be {tuple(required)}.")


class _Convolution(NetworkOperator):
    def validate_metadata(self, metadata, c):
        if "shape" in metadata and len(metadata["shape"]) != c["spatial_dims"] + 2:
            raise ValidationError("Convolution expects channels-last batched spatial inputs.")

    def validate(self, c):
        _fields(c, ("kernel_size", "spatial_dims", "dilation", "padding", "chunk_size"))
        if integer(c["spatial_dims"], "spatial_dims", 1) not in (1, 2):
            raise ValidationError("Convolution supports one or two spatial dimensions.")
        for key in ("kernel_size", "dilation", "chunk_size"):
            integer(c[key], key, 1)
        if c["padding"] not in {"same", "circular", "causal"} or (
            c["padding"] == "causal" and c["spatial_dims"] != 1
        ):
            raise ValidationError("Use same/circular padding, or causal padding for 1D.")

    def input_width(self, hidden, c):
        return hidden * c["kernel_size"] ** c["spatial_dims"]

    def row_chunks(self, metadata, c):
        n = metadata["sites"]
        return tuple(min(c["chunk_size"], n - i) for i in range(0, n, c["chunk_size"]))

    def apply(self, ctx):
        x, c = ctx.inputs, ctx.spec.config
        spatial = x.shape[1:-1]
        if len(spatial) != c["spatial_dims"]:
            raise ValidationError("Convolution expects channels-last batched spatial inputs.")
        n, width = math.prod(spatial), x.shape[-1]
        ctx.check_workspace(
            min(n, c["chunk_size"])
            * c["kernel_size"] ** len(spatial)
            * (x.shape[0] * width + 3 * len(spatial))
        )
        flat = x.reshape((x.shape[0], n, width))
        shifts = np.arange(c["kernel_size"])
        shifts -= c["kernel_size"] - 1 if c["padding"] == "causal" else (c["kernel_size"] - 1) // 2
        offsets = np.asarray(list(itertools.product(shifts * c["dilation"], repeat=len(spatial))))
        results = []
        for start in range(0, n, c["chunk_size"]):
            sites = np.stack(
                np.unravel_index(np.arange(start, min(n, start + c["chunk_size"])), spatial),
                axis=-1,
            )
            neighbors = sites[:, None, :] + offsets[None, :, :]
            valid = np.all((neighbors >= 0) & (neighbors < np.asarray(spatial)), axis=-1)
            wrapped = neighbors % np.asarray(spatial)
            indices = np.ravel_multi_index(tuple(np.moveaxis(wrapped, -1, 0)), spatial)
            patches = flat[:, ctx.array(indices, integer=True), :]
            if c["padding"] != "circular":
                patches = patches * ctx.array(valid)[None, ..., None]
            patches = patches.reshape((x.shape[0], len(sites), -1))
            results.append(ctx.project("decode", ctx.kernel(patches)))
        return ctx.cat(results, axis=1).reshape(x.shape)


class _Attention(NetworkOperator):
    def validate_metadata(self, metadata, c):
        if "shape" in metadata and len(metadata["shape"]) < 3:
            raise ValidationError("Attention expects a batch and token dimensions.")

    def validate(self, c):
        _fields(c, ("heads", "key_features", "mode", "causal", "positional", "chunk_size"))
        for key in ("heads", "key_features", "chunk_size"):
            integer(c[key], key, 1)
        if c["mode"] not in {"gaussian", "linear"} or any(
            type(c[k]) is not bool for k in ("causal", "positional")
        ):
            raise ValidationError("Attention uses gaussian/linear kernels and boolean flags.")

    def roles(self, c):
        return ("", "q", "k")

    def declare(self, hidden, readouts, c, projection, parameter):
        if hidden % c["heads"]:
            raise ValidationError("Hidden width must be divisible by attention heads.")
        for role in ("q", "k"):
            projection(f"attention_{role}", readouts, c["heads"] * c["key_features"])

    def apply(self, ctx):
        from .network_attention import quantum_attention

        return quantum_attention(ctx)


class _Graph(NetworkOperator):
    def validate_metadata(self, metadata, c):
        if "shape" in metadata and "edges" not in metadata:
            raise ValidationError("Graph execution requires edge_index metadata.")

    def validate(self, c):
        _fields(c, ("edge_features", "geometric", "aggregation", "chunk_size"))
        integer(c["edge_features"], "edge_features", 0)
        integer(c["chunk_size"], "chunk_size", 1)
        if type(c["geometric"]) is not bool or c["aggregation"] not in {"sum", "mean"}:
            raise ValidationError(
                "Graph operators use sum/mean aggregation and geometric=True/False."
            )

    def input_width(self, hidden, c):
        return 2 * hidden + c["edge_features"] + int(c["geometric"])

    def row_chunks(self, metadata, c):
        n = metadata.get("edges", metadata["sites"])
        return tuple(min(c["chunk_size"], n - i) for i in range(0, n, c["chunk_size"]))

    def apply(self, ctx):
        from .network_graph import quantum_messages

        return quantum_messages(ctx)


class _StateAttention(NetworkOperator):
    def validate(self, c):
        from .parameters import real

        _fields(c, ("heads", "causal", "positional", "chunk_size", "depolarization"))
        for key in ("heads", "chunk_size"):
            integer(c[key], key, 1)
        if any(type(c[k]) is not bool for k in ("causal", "positional")):
            raise ValidationError("State attention flags must be boolean.")
        if not 0 < real(c["depolarization"]) < 1:
            raise ValidationError("depolarization must lie strictly between zero and one.")

    def validate_cell(self, spec, c):
        from .network_attention import reduced_state_groups

        reduced_state_groups(spec)

    def validate_metadata(self, metadata, c):
        _Attention().validate_metadata(metadata, c)

    def declare(self, hidden, readouts, c, projection, parameter):
        if hidden % c["heads"]:
            raise ValidationError("Hidden width must be divisible by attention heads.")
        parameter("state_channel", (c["heads"], readouts // 3, 4), scale=0.05)

    def row_chunks(self, metadata, c):
        return _Convolution().row_chunks(metadata, c)

    def apply(self, ctx):
        from .network_attention import state_attention

        return state_attention(ctx)


class _PairGraph(NetworkOperator):
    def validate(self, c):
        _fields(c, ("edge_features", "geometric", "chunk_size"))
        integer(c["edge_features"], "edge_features", 0)
        integer(c["chunk_size"], "chunk_size", 1)
        if type(c["geometric"]) is not bool:
            raise ValidationError("geometric must be boolean.")

    def validate_metadata(self, metadata, c):
        if "pair_nodes" not in metadata:
            raise ValidationError("pair-graph requires input_lift='graph-pairs'.")

    def declare(self, hidden, readouts, c, projection, parameter):
        projection("pair_left", hidden + readouts, hidden)
        projection("pair_right", hidden + readouts, hidden)
        projection("pair_contract", hidden, hidden)

    def row_chunks(self, metadata, c):
        return _Convolution().row_chunks(metadata, c)

    def apply(self, ctx):
        from .network_graph import quantum_pairs

        return quantum_pairs(ctx)


class _QuantumKAN(NetworkOperator):
    uses_decode = False

    def validate(self, c):
        _fields(c, ("chunk_size",))
        integer(c["chunk_size"], "chunk_size", 1)

    def input_width(self, hidden, c):
        return 1

    def declare(self, hidden, readouts, c, projection, parameter):
        projection("kan_mix", hidden * readouts, hidden)

    def row_chunks(self, metadata, c):
        return _Convolution().row_chunks(
            dict(metadata, sites=metadata["sites"] * metadata["hidden_features"]), c
        )

    def apply(self, ctx):
        x, size = ctx.inputs, ctx.spec.config["chunk_size"]
        flat = x.reshape((x.shape[0], -1, 1))
        ctx.check_workspace(x.shape[0] * x[0].numel() if ctx.backend == "torch" else x.size)
        values = [
            ctx.kernel(flat[:, start : start + size]) for start in range(0, flat.shape[1], size)
        ]
        basis = ctx.cat(values, axis=1).reshape(x.shape[:-1] + (-1,))
        return ctx.project("kan_mix", basis)


class _Memory(NetworkOperator):
    stateful = native_state = True

    def validate(self, c):
        _fields(c, ("gated", "scan", "max_work"))
        if type(c["gated"]) is not bool or c["scan"] not in {"sequential", "parallel"}:
            raise ValidationError("Memory requires a boolean gate and sequential/parallel scan.")
        integer(c["max_work"], "max_work", 1)

    def validate_metadata(self, metadata, c):
        if "shape" in metadata and len(metadata["shape"]) != 3:
            raise ValidationError("Quantum memory expects features[B,T,H].")

    def declare(self, hidden, readouts, c, projection, parameter):
        if c["gated"]:
            projection("memory_gate", hidden, 1)

    def row_chunks(self, metadata, c):
        return (metadata["sites"],) if c["scan"] == "parallel" else (1,) * metadata["sites"]

    def _scan(self, ctx, **options):
        c, x = ctx.spec.config, ctx.inputs
        if x.ndim != 3:
            raise ValidationError("Quantum memory expects features[B,T,H].")
        cell = ctx.architecture.cells[ctx.index]
        controls = math.pi * ctx.xp.tanh(ctx.project("encode", x))
        weights = ctx.parameters[f"{ctx.group}_quantum"]
        mixing = None
        if c["gated"]:
            configuration = ctx.architecture.quantum
            configuration = (
                configuration
                if hasattr(configuration, "initial_mixing")
                else configuration[ctx.index]
            )
            logit = math.log(configuration.initial_mixing / (1 - configuration.initial_mixing))
            mixing = 0.5 * (1 + ctx.xp.tanh((ctx.project("memory_gate", x)[..., 0] + logit) * 0.5))
        kwargs = dict(
            state=ctx.context.get("_state"),
            mask=ctx.context.get("mask"),
            reset_mask=ctx.context.get("reset_mask"),
            mixing=mixing,
            method=c["scan"],
            max_work=c["max_work"],
            **options,
        )
        return (
            cell.scan(weights, controls, backend=ctx.backend, **kwargs)
            if ctx.evaluator is None
            else ctx.evaluator(weights, controls, **kwargs)
        )

    def apply(self, ctx):
        output, state = self._scan(ctx)
        return ctx.project("decode", output), state


class _Born(NetworkOperator):
    terminal = True
    uses_decode = False

    def validate(self, c):
        from .parameters import real

        _fields(c, ("visible_qubits", "depolarization", "chunk_size"))
        integer(c["visible_qubits"], "visible_qubits", 1)
        integer(c["chunk_size"], "chunk_size", 1)
        if not 0 <= real(c["depolarization"]) < 1:
            raise ValidationError("depolarization must lie in [0,1).")

    def validate_cell(self, spec, c):
        k, n = c["visible_qubits"], spec.circuit.num_qubits
        if (
            k > n
            or len(spec.observables).bit_length() != k
            or len(spec.observables) != (1 << k) - 1
        ):
            raise ValidationError("Born readout requires the complete ordered visible Z basis.")
        for i, ob in enumerate(spec.observables, 1):
            expected = "".join("Z" if i & (1 << (k - q - 1)) else "I" for q in range(k)) + "I" * (
                n - k
            )
            if not hasattr(ob, "paulis") or ob.paulis != expected or ob.coefficient != 1:
                raise ValidationError("Born readout requires the complete ordered visible Z basis.")

    def row_chunks(self, metadata, c):
        return _Convolution().row_chunks(metadata, c)

    def apply(self, ctx):
        from .quantum_generative import born_probabilities

        x, size = ctx.inputs, ctx.spec.config["chunk_size"]
        flat = x.reshape((x.shape[0], -1, x.shape[-1]))
        rows = [ctx.kernel(flat[:, i : i + size]) for i in range(0, flat.shape[1], size)]
        moments = ctx.cat(rows, axis=1).reshape(x.shape[:-1] + (-1,))
        return ctx.architecture._masked(
            born_probabilities(
                moments, backend=ctx.backend, depolarization=ctx.spec.config["depolarization"]
            ),
            ctx.context,
            ctx.backend,
        )


class _BornMemory(_Memory):
    terminal = True
    uses_decode = False

    def validate(self, c):
        from .parameters import real

        _fields(
            c,
            (
                "visible_qubits",
                "depolarization",
                "gated",
                "measurement",
                "memory_qubits",
                "povm_regularization",
                "scan",
                "max_work",
            ),
        )
        _Memory().validate({k: c[k] for k in ("gated", "scan", "max_work")})
        for name in ("visible_qubits", "memory_qubits"):
            integer(c[name], name, 1)
        if (
            c["measurement"] not in {"basis", "povm"}
            or not 0 <= real(c["depolarization"]) < 1
            or real(c["povm_regularization"]) <= 0
        ):
            raise ValidationError("Invalid Born memory measurement configuration.")
        if c["measurement"] == "basis" and c["visible_qubits"] > c["memory_qubits"]:
            raise ValidationError("Basis readout cannot exceed the memory width.")

    def declare(self, hidden, readouts, c, projection, parameter):
        from .quantum_measurement import QuantumPOVMPlan

        super().declare(hidden, readouts, c, projection, parameter)
        if c["measurement"] == "povm":
            plan = QuantumPOVMPlan(
                c["memory_qubits"],
                1 << c["visible_qubits"],
                c["povm_regularization"],
                c["depolarization"],
            )
            parameter(
                "povm_frame",
                plan.parameter_shape,
                "quantum",
                scale=1 / math.sqrt(plan.outcomes * (1 << plan.memory_qubits)),
            )

    def apply(self, ctx):
        from .quantum_measurement import QuantumPOVMPlan

        c = ctx.spec.config
        cell = ctx.architecture.cells[ctx.index]
        if cell.memory_qubits != c["memory_qubits"]:
            raise ValidationError("Born measurement and memory cell widths must match.")
        povm = None
        if c["measurement"] == "povm":
            plan = QuantumPOVMPlan(
                c["memory_qubits"],
                1 << c["visible_qubits"],
                c["povm_regularization"],
                c["depolarization"],
                ctx.architecture.max_memory_mb,
            )
            povm = plan.prepare(ctx.parameters[f"{ctx.group}_povm_frame"], backend=ctx.backend)
        return self._scan(
            ctx,
            visible_qubits=c["visible_qubits"] if povm is None else 0,
            depolarization=c["depolarization"],
            povm=povm,
        )


def input_metadata(inputs, input_features):
    context = inputs if isinstance(inputs, Mapping) else {}
    x = context.get("nodes", context.get("x")) if context else inputs
    if x is None or x.ndim < 1 or x.shape[-1] != input_features:
        raise ValidationError("Batched input feature width differs from the architecture.")
    if any(size < 1 for size in x.shape):
        raise ValidationError("Input batch and feature/site dimensions must be nonempty.")
    meta = {
        "samples": x.shape[0] if x.ndim > 1 else 1,
        "sites": math.prod(x.shape[1:-1]),
        "shape": tuple(x.shape),
    }
    if "edge_index" in context:
        edge = context["edge_index"]
        if x.ndim != 3 or edge.ndim != 3 or edge.shape[0] != x.shape[0] or edge.shape[-1] != 2:
            raise ValidationError("Graph inputs require nodes[B,N,F] and edge_index[B,E,2].")
        meta["edges"] = edge.shape[1]
    mask = context.get("node_mask", context.get("mask"))
    if mask is not None and tuple(mask.shape) != tuple(x.shape[:-1]):
        raise ValidationError("Input mask shape must match batch and site dimensions.")
    return x, context, meta


class OperatorContext:
    def __init__(self, architecture, index, parameters, inputs, context, backend, evaluator):
        from .simulator import _TensorOps

        self.architecture, self.index, self.parameters = architecture, index, parameters
        self.inputs, self.context, self.backend = inputs, context, backend
        self.spec = architecture.operator_specs[index]
        self.group = f"block_{index}"
        self.ops = _TensorOps(backend, parameters[f"{self.group}_quantum"])
        self.xp, self.evaluator = self.ops.xp, evaluator

    def array(self, value, *, integer=False, boolean=False):
        if self.backend == "torch":
            return self.xp.as_tensor(
                value,
                device=self.inputs.device,
                dtype=self.xp.int64 if integer else self.xp.bool if boolean else self.inputs.dtype,
            )
        return self.xp.asarray(
            value,
            dtype="int64"
            if integer and self.backend == "numpy"
            else "int32"
            if integer
            else "bool"
            if boolean
            else self.inputs.dtype,
        )

    def cat(self, values, axis=-1):
        return (
            self.xp.cat(values, dim=axis)
            if self.backend == "torch"
            else self.xp.concatenate(values, axis=axis)
        )

    def sum(self, value, axis, keepdims=False):
        return (
            value.sum(dim=axis, keepdim=keepdims)
            if self.backend == "torch"
            else self.xp.sum(value, axis=axis, keepdims=keepdims)
        )

    def zeros(self, shape):
        if self.backend == "torch":
            return self.xp.zeros(shape, dtype=self.inputs.dtype, device=self.inputs.device)
        return self.xp.zeros(shape, dtype=self.inputs.dtype)

    def project(self, name, value):
        return self.architecture._linear(
            f"{self.group}_{name}", self.parameters, value, self.backend
        )

    def kernel(self, value, role=""):
        from .simulator import expectations

        suffix = "" if not role else f"_{role}"
        controls = math.pi * self.xp.tanh(self.project(f"encode{suffix}", value))
        weights = self.parameters[f"{self.group}_quantum{suffix}"]
        if self.evaluator is not None:
            return self.evaluator(weights, controls)
        spec = self.architecture.cells[self.index].to_spec()
        bindings = {name: controls[..., i] for i, name in enumerate(spec.input_parameters)}
        bindings.update({name: weights[i] for i, name in enumerate(spec.weight_parameters)})
        return expectations(
            spec.circuit,
            spec.observables,
            bindings,
            backend=self.backend,
            max_memory_mb=self.architecture.max_memory_mb,
        )

    def check_workspace(self, elements):
        if (
            elements
            * max(
                4,
                self.inputs.dtype.itemsize
                if self.backend != "torch"
                else self.inputs.element_size(),
            )
            > self.architecture.max_memory_mb * 2**20
        ):
            raise ResourceError(
                "Operator forward workspace exceeds max_memory_mb; reduce chunk size."
            )


for _name, _operator in (
    ("pointwise", NetworkOperator()),
    ("convolution", _Convolution()),
    ("attention", _Attention()),
    ("graph", _Graph()),
    ("state-attention", _StateAttention()),
    ("pair-graph", _PairGraph()),
    ("quantum-kan", _QuantumKAN()),
    ("quantum-memory", _Memory()),
    ("born", _Born()),
    ("born-memory", _BornMemory()),
):
    register_network_operator(_name, _operator)
