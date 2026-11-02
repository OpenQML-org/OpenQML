"""One model program and parameter layout for deep quantum-classical networks.

The architecture owns composition, parameter identity and resource planning.
Framework adapters own tensors and autodiff; execution adapters own hardware.
Increasing active depth changes execution, never the allocated parameter tree.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType

import numpy as np

from .deep import DeepQuantumPlan
from .errors import ResourceError, ValidationError
from .ir import integer
from .multiscale import MultiscalePlan
from .network_operators import OperatorContext, OperatorSpec, input_metadata
from .parameters import real
from .quantum_cells import CustomCircuitPlan, FeatureCircuitPlan
from .simulator import _TensorOps
from .tensor_train import TensorTrainPlan


@dataclass(frozen=True)
class QuantumCellSpec:
    kind: str = "multiscale"
    num_qubits: int = 4
    depth: int = 1
    bond_qubits: int = 1
    num_outputs: int = 3
    reupload: bool = True
    initial_mixing: float = 0.1
    max_operations: int = 100_000
    program_json: str | None = None
    _custom_spec: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.kind not in {
            "multiscale",
            "residual-channel",
            "qcnn",
            "hardware-efficient",
            "custom",
        }:
            raise ValidationError("Unknown quantum cell kind.")
        for name in ("num_qubits", "depth", "bond_qubits", "num_outputs", "max_operations"):
            object.__setattr__(self, name, integer(getattr(self, name), name, 1))
        if type(self.reupload) is not bool:
            raise ValidationError("reupload must be a boolean.")
        if not 0 < real(self.initial_mixing) < 1:
            raise ValidationError("initial_mixing must lie strictly between zero and one.")
        object.__setattr__(self, "initial_mixing", real(self.initial_mixing))
        if self.kind == "custom":
            from .layers import LayerSpec

            if not isinstance(self.program_json, str):
                raise ValidationError("Custom cells require a portable LayerSpec JSON document.")
            try:
                payload = json.loads(self.program_json)
            except json.JSONDecodeError as exc:
                raise ValidationError("Malformed custom quantum cell JSON.") from exc
            spec = LayerSpec.from_dict(payload, max_operations=self.max_operations)
            if not spec.input_parameters:
                raise ValidationError("A custom hybrid cell needs input parameters.")
            object.__setattr__(self, "_custom_spec", spec)
            object.__setattr__(self, "num_qubits", spec.circuit.num_qubits)
            object.__setattr__(self, "num_outputs", len(spec.observables))
            object.__setattr__(
                self,
                "program_json",
                json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":")),
            )
        elif self.program_json is not None:
            raise ValidationError("program_json is only valid for custom quantum cells.")

    def to_dict(self):
        value = {
            name: getattr(self, name)
            for name in (
                "kind",
                "num_qubits",
                "depth",
                "bond_qubits",
                "num_outputs",
                "reupload",
                "initial_mixing",
                "max_operations",
            )
        }
        if self.program_json is not None:
            value["program_json"] = self.program_json
        return value

    @classmethod
    def from_spec(cls, spec, *, max_operations=100_000):
        from .layers import LayerSpec

        if not isinstance(spec, LayerSpec):
            raise ValidationError("Provide a portable LayerSpec.")
        return cls(kind="custom", program_json=spec.to_json(), max_operations=max_operations)

    def compile(self):
        if self.kind == "custom":
            return CustomCircuitPlan(self._custom_spec)
        if self.kind in {"qcnn", "hardware-efficient"}:
            return FeatureCircuitPlan(
                self.num_qubits,
                self.depth,
                self.num_outputs,
                self.reupload,
                self.max_operations,
                self.kind == "qcnn",
                self.bond_qubits,
            )
        return (
            DeepQuantumPlan(
                self.num_qubits, self.depth, self.reupload, self.initial_mixing, self.max_operations
            )
            if self.kind == "residual-channel"
            else MultiscalePlan(
                self.num_qubits,
                self.bond_qubits,
                self.depth,
                self.num_outputs,
                reupload=self.reupload,
                max_operations=self.max_operations,
            )
        )


@dataclass(frozen=True)
class NetworkParameter:
    name: str
    shape: tuple
    role: str
    group: str
    initialization: str = "normal"
    scale: float = 0.1


@dataclass(frozen=True, eq=False)
class HybridArchitecture:
    input_features: int
    output_features: int
    hidden_features: int = 32
    depth: int = 6
    quantum: object = field(default_factory=QuantumCellSpec)
    expansion: int = 2
    bond_dimension: int = 4
    projection_policy: str = "auto"
    max_parameters: int = 1_000_000
    max_memory_mb: float = 256
    rematerialize: bool = True
    norm_epsilon: float = 1e-5
    operators: object = field(default_factory=OperatorSpec)
    readout_pooling: str = "none"
    input_lift: str = "none"
    native_execution: str = "auto"
    parameters: tuple = field(init=False, repr=False)
    cells: tuple = field(init=False, repr=False)
    operator_specs: tuple = field(init=False, repr=False)
    projections: object = field(init=False, repr=False)
    projection_shapes: object = field(init=False, repr=False)
    _by_name: object = field(init=False, repr=False)

    def __post_init__(self):
        for name in (
            "input_features",
            "output_features",
            "hidden_features",
            "depth",
            "expansion",
            "bond_dimension",
            "max_parameters",
        ):
            object.__setattr__(self, name, integer(getattr(self, name), name, 1))
        if real(self.max_memory_mb) <= 0 or real(self.norm_epsilon) <= 0:
            raise ValidationError("Memory and normalization epsilon must be positive.")
        object.__setattr__(self, "max_memory_mb", real(self.max_memory_mb))
        object.__setattr__(self, "norm_epsilon", real(self.norm_epsilon))
        if type(self.rematerialize) is not bool:
            raise ValidationError("rematerialize must be a boolean.")
        if self.projection_policy not in {"auto", "dense", "tensor-train"}:
            raise ValidationError("projection_policy must be auto, dense or tensor-train.")
        if self.native_execution not in {
            "auto",
            "pauli",
            "pauli-transfer",
            "planned",
            "tensor-network",
        }:
            raise ValidationError("Unknown native execution method.")
        if self.readout_pooling not in {"none", "mean", "sum", "first", "pair-invariant"}:
            raise ValidationError("Unknown readout pooling.")
        if self.input_lift not in {"none", "graph-pairs"}:
            raise ValidationError("input_lift must be none or graph-pairs.")
        if self.readout_pooling == "pair-invariant" and self.input_lift != "graph-pairs":
            raise ValidationError("pair-invariant readout requires graph-pairs input lifting.")
        operators = (
            (self.operators,) * self.depth
            if isinstance(self.operators, OperatorSpec)
            else tuple(self.operators)
        )
        if len(operators) != self.depth or any(not isinstance(o, OperatorSpec) for o in operators):
            raise ValidationError("Provide one OperatorSpec or one operator per block.")
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "operator_specs", operators)
        if self.input_lift == "graph-pairs" and (
            operators[0].kind != "pair-graph"
            or any(o.kind not in {"pair-graph", "pointwise"} for o in operators)
        ):
            raise ValidationError("graph-pairs uses pair-graph first, then pair/pointwise blocks.")
        if self.depth * (2 * self.hidden_features + 2) > self.max_parameters:
            raise ResourceError("Architecture depth exceeds the parameter admission budget.")
        configurations = (
            (self.quantum,) * self.depth
            if isinstance(self.quantum, QuantumCellSpec)
            else tuple(self.quantum)
        )
        if len(configurations) != self.depth or any(
            not isinstance(c, QuantumCellSpec) for c in configurations
        ):
            raise ValidationError("Provide one QuantumCellSpec or one spec per hybrid block.")
        cache = {}
        for configuration in configurations:
            if configuration not in cache:
                cache[configuration] = configuration.compile()
        cells = tuple(cache[c] for c in configurations)
        if not isinstance(self.quantum, QuantumCellSpec):
            object.__setattr__(self, "quantum", configurations)
        descriptors, projections, projection_shapes, parameter_names = [], {}, {}, set()
        count = 0

        def parameter(name, shape, role, group, initialization="normal", scale=0.1):
            nonlocal count
            if name in parameter_names:
                raise ValidationError(f"Duplicate network parameter {name!r}.")
            parameter_names.add(name)
            shape = tuple(
                integer(v, "parameter dimension", 0 if role == "quantum" else 1) for v in shape
            )
            if initialization not in {"normal", "zeros", "ones", "quantum"}:
                raise ValidationError("Unknown network parameter initializer.")
            scale = real(scale)
            count += math.prod(shape)
            if count > self.max_parameters or 8 * count > self.max_memory_mb * 2**20:
                raise ResourceError("Hybrid model parameters exceed the requested budget.")
            descriptors.append(
                NetworkParameter(name, tuple(shape), role, group, initialization, scale)
            )

        def projection(name, a, b, group):
            if name in projections:
                raise ValidationError(f"Duplicate network projection {name!r}.")
            a, b = integer(a, "in_features", 1), integer(b, "out_features", 1)
            plan = None
            if self.projection_policy != "dense":
                try:
                    plan = TensorTrainPlan.from_features(
                        a, b, bond_dimension=self.bond_dimension, max_memory_mb=self.max_memory_mb
                    )
                except ResourceError:
                    if self.projection_policy == "tensor-train":
                        raise
            if (
                self.projection_policy == "auto"
                and plan is not None
                and plan.num_parameters >= a * b
            ):
                plan = None
            projections[name] = plan
            projection_shapes[name] = (a, b)
            if plan is None:
                parameter(f"{name}_weight", (a, b), "projection", group, scale=1 / math.sqrt(a))
            else:
                power = math.prod(plan.ranks[1:-1]) ** (-1 / (2 * len(plan.core_shapes)))
                for i, shape in enumerate(plan.core_shapes):
                    parameter(
                        f"{name}_core_{i}",
                        shape,
                        "projection",
                        group,
                        scale=power / math.sqrt(shape[1]),
                    )
            parameter(f"{name}_bias", (b,), "bias", group, "zeros")

        h = self.hidden_features

        def declare_parameter(
            group, name, shape, role="classical", initialization="normal", scale=0.1
        ):
            parameter(f"{group}_{name}", shape, role, group, initialization, scale)

        stem_width = self.input_features
        if self.input_lift == "graph-pairs":
            config = operators[0].config
            stem_width = 2 * stem_width + 2 + config["edge_features"] + int(config["geometric"])
        parameter(
            "stem_weight",
            (stem_width, h),
            "classical",
            "shared",
            scale=1 / math.sqrt(stem_width),
        )
        parameter("stem_bias", (h,), "bias", "shared", "zeros")
        for i, (cell, operator) in enumerate(zip(cells, operators, strict=True)):
            group = f"block_{i}"
            parameter(f"{group}_norm_q", (h,), "norm", group, "ones")
            parameter(f"{group}_norm_ff", (h,), "norm", group, "ones")
            parameter(f"{group}_gain_q", (), "gate", group, "zeros")
            parameter(f"{group}_gain_ff", (), "gate", group, "zeros")
            spec = cell.to_spec()
            operator.implementation.validate_cell(spec, operator.config)
            kernel_width = integer(
                operator.implementation.input_width(h, operator.config), "kernel input width", 1
            )
            roles = tuple(operator.implementation.roles(operator.config))
            if (
                not roles
                or roles[0] != ""
                or len(set(roles)) != len(roles)
                or any(
                    not isinstance(role, str) or (role and not role.isidentifier())
                    for role in roles
                )
            ):
                raise ValidationError(
                    "Operator quantum roles start with '' and use unique identifiers."
                )
            for role in roles:
                suffix = "" if not role else f"_{role}"
                projection(
                    f"{group}_encode{suffix}", kernel_width, len(spec.input_parameters), group
                )
                parameter(
                    f"{group}_quantum{suffix}",
                    (len(spec.weight_parameters),),
                    "quantum",
                    group,
                    "quantum",
                )
            projection(f"{group}_decode", len(spec.observables), h, group)
            operator.implementation.declare(
                h,
                len(spec.observables),
                operator.config,
                lambda name, a, b, group=group: projection(f"{group}_{name}", a, b, group),
                partial(declare_parameter, group),
            )
            projection(f"{group}_ff_gate", h, h * self.expansion, group)
            projection(f"{group}_ff_up", h, h * self.expansion, group)
            projection(f"{group}_ff_down", h * self.expansion, h, group)
        parameter(
            "readout_weight",
            (h * (2 if self.readout_pooling == "pair-invariant" else 1), self.output_features),
            "classical",
            "shared",
            scale=1 / math.sqrt(h),
        )
        parameter("readout_bias", (self.output_features,), "bias", "shared", "zeros")
        for name, value in (
            ("parameters", tuple(descriptors)),
            ("cells", cells),
            ("projections", MappingProxyType(projections)),
            ("projection_shapes", MappingProxyType(projection_shapes)),
            ("_by_name", MappingProxyType({p.name: p for p in descriptors})),
        ):
            object.__setattr__(self, name, value)

    def active(self, depth):
        depth = self.depth if depth is None else integer(depth, "active_depth", 0)
        if depth > self.depth:
            raise ValidationError("Active depth exceeds the allocated architecture.")
        return depth

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def to_dict(self):
        value = {
            "format": "openqml.hybrid.v1",
            **{
                name: getattr(self, name)
                for name in (
                    "input_features",
                    "output_features",
                    "hidden_features",
                    "depth",
                    "expansion",
                    "bond_dimension",
                    "projection_policy",
                    "max_parameters",
                    "max_memory_mb",
                    "rematerialize",
                    "norm_epsilon",
                )
            },
            "quantum": [
                c.to_dict()
                for c in (
                    (self.quantum,) * self.depth
                    if isinstance(self.quantum, QuantumCellSpec)
                    else self.quantum
                )
            ],
        }
        # Default networks preserve the 0.9 serialization/fingerprint and can
        # resume their existing optimizer trees without migration.
        if any(o.kind != "pointwise" or o.config for o in self.operator_specs):
            value["operators"] = [o.to_dict() for o in self.operator_specs]
        if self.readout_pooling != "none":
            value["readout_pooling"] = self.readout_pooling
        if self.input_lift != "none":
            value["input_lift"] = self.input_lift
        if self.native_execution != "auto":
            value["native_execution"] = self.native_execution
        return value

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        if data.pop("format", None) != "openqml.hybrid.v1":
            raise ValidationError("Unsupported hybrid architecture format.")
        try:
            data["quantum"] = tuple(QuantumCellSpec(**c) for c in data["quantum"])
            if "operators" in data:
                data["operators"] = tuple(OperatorSpec(**o) for o in data["operators"])
            return cls(**data)
        except (TypeError, KeyError) as exc:
            raise ValidationError("Malformed hybrid architecture.") from exc

    def initialize_one(self, name, normal, full, *, backend="numpy"):
        p = self._by_name[name]
        if p.initialization in {"zeros", "ones"}:
            return full(p.shape, int(p.initialization == "ones"))
        value = normal(p.shape) * p.scale
        if p.role == "quantum":
            cell = self.cells[int(p.group.split("_")[1])]
            for a, b in getattr(cell, "initialization_pairs", ()):
                if backend == "jax":
                    value = value.at[b].set(-value[a])
                else:
                    value[b] = -value[a]
            for index, constant in getattr(cell, "initialization_values", ()):
                if backend == "jax":
                    value = value.at[index].set(constant)
                else:
                    value[index] = constant
        return value

    def initial_parameters(self, seed=None, *, dtype="float64"):
        if np.dtype(dtype) not in {np.dtype("float32"), np.dtype("float64")}:
            raise ValidationError("Hybrid parameters require float32 or float64.")
        rng = np.random.default_rng(seed)
        return {
            p.name: self.initialize_one(
                p.name,
                lambda s: rng.normal(size=s).astype(dtype),
                lambda s, v: np.full(s, v, dtype=dtype),
            )
            for p in self.parameters
        }

    def validate_parameters(self, parameters):
        if set(parameters) != set(self._by_name) or any(
            tuple(parameters[p.name].shape) != p.shape for p in self.parameters
        ):
            raise ValidationError(
                "Parameters must match the architecture's complete stable layout."
            )

    def _linear(self, name, parameters, inputs, backend):
        plan = self.projections[name]
        if plan is None:
            return inputs @ parameters[f"{name}_weight"] + parameters[f"{name}_bias"]
        cores = tuple(parameters[f"{name}_core_{i}"] for i in range(len(plan.core_shapes)))
        shape = inputs.shape[:-1]
        flat = inputs.reshape((-1, plan.in_features))
        size = plan.batch_capacity
        if len(flat) <= size:
            value = plan.apply(cores, inputs, backend=backend)
        else:
            values = [
                plan.apply(cores, flat[i : i + size], backend=backend)
                for i in range(0, len(flat), size)
            ]
            ops = _TensorOps(backend, cores[0])
            value = (
                ops.xp.cat(values, dim=0)
                if backend == "torch"
                else ops.xp.concatenate(values, axis=0)
            ).reshape(shape + (plan.out_features,))
        return value + parameters[f"{name}_bias"]

    def block(
        self, index, parameters, inputs, *, backend="numpy", quantum_evaluator=None, context=None
    ):
        ops = _TensorOps(backend, parameters[f"block_{index}_quantum"])
        group = f"block_{index}"

        def norm(value, name):
            mean = (
                value.square().mean(dim=-1, keepdim=True)
                if backend == "torch"
                else ops.xp.mean(value**2, axis=-1, keepdims=True)
            )
            return value / ops.xp.sqrt(mean + self.norm_epsilon) * parameters[name]

        normalized = norm(inputs, f"{group}_norm_q")
        ctx = OperatorContext(
            self, index, parameters, normalized, context or {}, backend, quantum_evaluator
        )
        quantum = self.operator_specs[index].implementation.apply(ctx)
        if tuple(quantum.shape) != tuple(inputs.shape):
            raise ValidationError("A network operator must preserve its hidden feature shape.")
        value = inputs + ops.xp.tanh(parameters[f"{group}_gain_q"]) * quantum
        value = self._masked(value, context or {}, backend)
        normalized = norm(value, f"{group}_norm_ff")
        gate = self._linear(f"{group}_ff_gate", parameters, normalized, backend)
        up = self._linear(f"{group}_ff_up", parameters, normalized, backend)
        silu = gate * (1 + ops.xp.tanh(gate * 0.5)) * 0.5
        return self._masked(
            value
            + ops.xp.tanh(parameters[f"{group}_gain_ff"])
            * self._linear(f"{group}_ff_down", parameters, silu * up, backend),
            context or {},
            backend,
        )

    @staticmethod
    def _masked(value, context, backend):
        mask = context.get("node_mask", context.get("mask"))
        if mask is None:
            return value
        ops = _TensorOps(backend, value)
        mask = (
            ops.xp.as_tensor(mask, device=value.device, dtype=ops.xp.bool)
            if backend == "torch"
            else ops.xp.asarray(mask, dtype=bool)
        )
        return ops.xp.where(mask[..., None], value, 0)

    def input_metadata(self, inputs):
        meta = input_metadata(inputs, self.input_features)[2]
        if self.input_lift == "graph-pairs":
            if "edges" not in meta:
                raise ValidationError("Pair graphs require edge_index and nodes[B,N,F].")
            meta = dict(meta, pair_nodes=meta["sites"], sites=meta["sites"] ** 2)
        return meta

    def execution_batches(self, metadata, microbatch_samples, index):
        samples = metadata["samples"]
        operator = self.operator_specs[index]
        chunks = operator.implementation.row_chunks(metadata, operator.config)
        roles = tuple(operator.implementation.roles(operator.config))
        return tuple(
            min(microbatch_samples, samples - start) * rows
            for start in range(0, samples, microbatch_samples)
            for _ in roles
            for rows in chunks
        )

    def evaluate(
        self, parameters, inputs, *, backend="numpy", active_depth=None, block_runner=None
    ):
        self.validate_parameters(parameters)
        depth = self.active(active_depth)
        inputs, context, metadata = input_metadata(inputs, self.input_features)
        inputs = self._masked(inputs, context, backend)
        if self.input_lift == "graph-pairs":
            from .network_graph import lift_graph_pairs

            metadata = self.input_metadata(context)
            ctx = OperatorContext(self, 0, parameters, inputs, context, backend, None)
            inputs, context = lift_graph_pairs(ctx)
        value = inputs @ parameters["stem_weight"] + parameters["stem_bias"]
        value = self._masked(value, context, backend)
        for index in range(depth):
            self.operator_specs[index].implementation.validate_metadata(
                metadata,
                self.operator_specs[index].config,
            )
            value = (
                self.block(index, parameters, value, backend=backend, context=context)
                if block_runner is None
                else block_runner(index, parameters, value, context)
            )
        if self.readout_pooling != "none" and value.ndim > 2:
            ops = _TensorOps(backend, value)
            axes = tuple(range(1, value.ndim - 1))
            if self.readout_pooling == "pair-invariant":
                from .network_graph import invariant_pair_readout

                value = invariant_pair_readout(value, context, backend)
            elif self.readout_pooling == "first":
                value = value.reshape((value.shape[0], -1, value.shape[-1]))[:, 0]
            else:
                summed = value.sum(dim=axes) if backend == "torch" else ops.xp.sum(value, axis=axes)
                if self.readout_pooling == "mean":
                    ones = ops.xp.ones_like(value[..., :1])
                    ones = self._masked(ones, context, backend)
                    denominator = (
                        ones.sum(dim=axes) if backend == "torch" else ops.xp.sum(ones, axis=axes)
                    )
                    denominator = (
                        denominator.clamp_min(1)
                        if backend == "torch"
                        else ops.xp.maximum(denominator, 1)
                    )
                    summed = summed / denominator
                value = summed
            return value @ parameters["readout_weight"] + parameters["readout_bias"]
        return self._masked(
            value @ parameters["readout_weight"] + parameters["readout_bias"], context, backend
        )

    def resource_profile(
        self, *, batch_size=1, active_depth=None, dtype_bytes=4, optimizer_slots=2
    ):
        integer(batch_size, "batch_size", 1)
        integer(dtype_bytes, "dtype_bytes", 1)
        integer(optimizer_slots, "optimizer_slots", 0)
        depth = self.active(active_depth)
        counts = {
            role: sum(math.prod(p.shape) for p in self.parameters if p.role == role)
            for role in {p.role for p in self.parameters}
        }
        total = sum(counts.values())
        return {
            "architecture": "gated_quantum_classical_residual_network",
            "depth": self.depth,
            "active_depth": depth,
            "parameters": total,
            "parameters_by_role": counts,
            "parameter_gradient_optimizer_bytes": total * dtype_bytes * (2 + optimizer_slots),
            "memory_scope": "excludes_activations_communication_and_allocator_reserves",
            "projection_parameters": sum(
                p.num_parameters if p is not None else math.prod(self.projection_shapes[name])
                for name, p in self.projections.items()
            ),
            "dense_projection_parameters": sum(
                math.prod(shape) for shape in self.projection_shapes.values()
            ),
            "projection_representations": {
                name: "dense" if p is None else "tensor-train"
                for name, p in self.projections.items()
            },
            "quantum_cells": [self.cells[i].to_dict() for i in range(depth)],
            "operators": [o.to_dict() for o in self.operator_specs[:depth]],
            "quantum_roles": [
                list(o.implementation.roles(o.config)) for o in self.operator_specs[:depth]
            ],
            "batch_size": batch_size,
        }

    def build(self, framework="torch", **options):
        if framework == "torch":
            from .interfaces._torch_network import HybridNetwork
        elif framework == "jax":
            from .interfaces._jax_network import HybridNetwork
        elif framework == "flax":
            from .interfaces._flax_network import HybridNetwork
        else:
            raise ValidationError("Architecture framework must be torch, jax or flax.")
        return HybridNetwork(self, **options)

    @classmethod
    def convolutional(
        cls,
        input_features,
        output_features,
        *,
        spatial_dims=2,
        kernel_size=3,
        dilation=1,
        padding="same",
        quantum=None,
        **options,
    ):
        return cls(
            input_features,
            output_features,
            quantum=QuantumCellSpec(kind="qcnn") if quantum is None else quantum,
            operators=OperatorSpec.convolution(
                kernel_size, spatial_dims=spatial_dims, dilation=dilation, padding=padding
            ),
            readout_pooling=options.pop("readout_pooling", "mean"),
            **options,
        )

    @classmethod
    def transformer(
        cls,
        input_features,
        output_features,
        *,
        heads=1,
        key_features=8,
        attention="gaussian",
        causal=False,
        positional=True,
        quantum=None,
        **options,
    ):
        return cls(
            input_features,
            output_features,
            quantum=QuantumCellSpec(kind="qcnn" if attention == "state" else "hardware-efficient")
            if quantum is None
            else quantum,
            operators=OperatorSpec.state_attention(
                heads=heads, causal=causal, positional=positional
            )
            if attention == "state"
            else OperatorSpec.attention(
                heads=heads,
                key_features=key_features,
                mode=attention,
                causal=causal,
                positional=positional,
            ),
            **options,
        )

    @classmethod
    def graph_network(
        cls,
        input_features,
        output_features,
        *,
        edge_features=0,
        geometric=False,
        aggregation="mean",
        order=1,
        quantum=None,
        **options,
    ):
        if integer(order, "order", 1) not in (1, 2):
            raise ValidationError("Graph order must be 1 (sparse messages) or 2 (ordered pairs).")
        return cls(
            input_features,
            output_features,
            quantum=QuantumCellSpec(kind="hardware-efficient") if quantum is None else quantum,
            operators=OperatorSpec.pair_graph(edge_features=edge_features, geometric=geometric)
            if order == 2
            else OperatorSpec.graph(
                edge_features=edge_features, geometric=geometric, aggregation=aggregation
            ),
            readout_pooling=options.pop(
                "readout_pooling", "pair-invariant" if order == 2 else "mean"
            ),
            input_lift="graph-pairs" if order == 2 else "none",
            **options,
        )

    @classmethod
    def deep(
        cls,
        input_features,
        output_features,
        *,
        num_qubits=4,
        depth=4,
        hidden_features=32,
        quantum_depth=2,
        reupload=True,
        initial_mixing=0.1,
        max_memory_mb=256,
        **options,
    ):
        return cls(
            input_features,
            output_features,
            hidden_features=hidden_features,
            depth=depth,
            quantum=QuantumCellSpec(
                kind="residual-channel",
                num_qubits=num_qubits,
                depth=quantum_depth,
                reupload=reupload,
                initial_mixing=initial_mixing,
            ),
            max_memory_mb=max_memory_mb,
            **options,
        )

    def schedule(self, **options):
        from .architecture_runtime import schedule_network

        return schedule_network(self, **options)

    def gradient_metrics(self, gradients, *, backend="jax"):
        ops = _TensorOps(backend, gradients["stem_weight"])
        values = {"quantum_grad_norm": [], "classical_grad_norm": []}
        for descriptor in self.parameters:
            key = "quantum_grad_norm" if descriptor.role == "quantum" else "classical_grad_norm"
            value = gradients[descriptor.name]
            values[key].append(ops.xp.sum(value**2))
        return {key: ops.xp.sqrt(sum(rows)) for key, rows in values.items()}
