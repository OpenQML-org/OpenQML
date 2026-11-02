"""One model program and parameter layout for deep quantum-classical networks.

The architecture owns composition, parameter identity and resource planning.
Framework adapters own tensors and autodiff; execution adapters own hardware.
Increasing active depth changes execution, never the allocated parameter tree.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from types import MappingProxyType

import numpy as np

from .deep import DeepQuantumPlan
from .errors import ResourceError, ValidationError
from .ir import integer
from .multiscale import MultiscalePlan
from .parameters import real
from .simulator import _TensorOps, expectations
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

    def __post_init__(self):
        if self.kind not in {"multiscale", "residual-channel"}:
            raise ValidationError("Quantum cell must be multiscale or residual-channel.")
        for name in ("num_qubits", "depth", "bond_qubits", "num_outputs", "max_operations"):
            object.__setattr__(self, name, integer(getattr(self, name), name, 1))
        if type(self.reupload) is not bool:
            raise ValidationError("reupload must be a boolean.")
        if not 0 < real(self.initial_mixing) < 1:
            raise ValidationError("initial_mixing must lie strictly between zero and one.")
        object.__setattr__(self, "initial_mixing", real(self.initial_mixing))

    def compile(self):
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
    parameters: tuple = field(init=False, repr=False)
    cells: tuple = field(init=False, repr=False)
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
        descriptors, projections, projection_shapes = [], {}, {}
        count = 0

        def parameter(name, shape, role, group, initialization="normal", scale=0.1):
            nonlocal count
            count += math.prod(shape)
            if count > self.max_parameters or 8 * count > self.max_memory_mb * 2**20:
                raise ResourceError("Hybrid model parameters exceed the requested budget.")
            descriptors.append(
                NetworkParameter(name, tuple(shape), role, group, initialization, scale)
            )

        def projection(name, a, b, group):
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
        parameter(
            "stem_weight",
            (self.input_features, h),
            "classical",
            "shared",
            scale=1 / math.sqrt(self.input_features),
        )
        parameter("stem_bias", (h,), "bias", "shared", "zeros")
        for i, cell in enumerate(cells):
            group = f"block_{i}"
            parameter(f"{group}_norm_q", (h,), "norm", group, "ones")
            parameter(f"{group}_norm_ff", (h,), "norm", group, "ones")
            parameter(f"{group}_gain_q", (), "gate", group, "zeros")
            parameter(f"{group}_gain_ff", (), "gate", group, "zeros")
            spec = cell.to_spec()
            projection(f"{group}_encode", h, len(spec.input_parameters), group)
            parameter(
                f"{group}_quantum", (len(spec.weight_parameters),), "quantum", group, "quantum"
            )
            projection(f"{group}_decode", len(spec.observables), h, group)
            projection(f"{group}_ff_gate", h, h * self.expansion, group)
            projection(f"{group}_ff_up", h, h * self.expansion, group)
            projection(f"{group}_ff_down", h * self.expansion, h, group)
        parameter(
            "readout_weight",
            (h, self.output_features),
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
        return {
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
                asdict(c)
                for c in (
                    (self.quantum,) * self.depth
                    if isinstance(self.quantum, QuantumCellSpec)
                    else self.quantum
                )
            ],
        }

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        if data.pop("format", None) != "openqml.hybrid.v1":
            raise ValidationError("Unsupported hybrid architecture format.")
        try:
            data["quantum"] = tuple(QuantumCellSpec(**c) for c in data["quantum"])
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

    def block(self, index, parameters, inputs, *, backend="numpy", quantum_evaluator=None):
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
        controls = math.pi * ops.xp.tanh(
            self._linear(f"{group}_encode", parameters, normalized, backend)
        )
        weights = parameters[f"{group}_quantum"]
        if quantum_evaluator is None:
            spec = self.cells[index].to_spec()
            bindings = {name: controls[..., i] for i, name in enumerate(spec.input_parameters)}
            bindings.update({name: weights[i] for i, name in enumerate(spec.weight_parameters)})
            quantum = expectations(
                spec.circuit,
                spec.observables,
                bindings,
                backend=backend,
                max_memory_mb=self.max_memory_mb,
            )
        else:
            quantum = quantum_evaluator(weights, controls)
        value = inputs + ops.xp.tanh(parameters[f"{group}_gain_q"]) * self._linear(
            f"{group}_decode", parameters, quantum, backend
        )
        normalized = norm(value, f"{group}_norm_ff")
        gate = self._linear(f"{group}_ff_gate", parameters, normalized, backend)
        up = self._linear(f"{group}_ff_up", parameters, normalized, backend)
        silu = gate * (1 + ops.xp.tanh(gate * 0.5)) * 0.5
        return value + ops.xp.tanh(parameters[f"{group}_gain_ff"]) * self._linear(
            f"{group}_ff_down", parameters, silu * up, backend
        )

    def evaluate(
        self, parameters, inputs, *, backend="numpy", active_depth=None, block_runner=None
    ):
        self.validate_parameters(parameters)
        depth = self.active(active_depth)
        if inputs.ndim < 1 or inputs.shape[-1] != self.input_features:
            raise ValidationError("Input feature width differs from the architecture.")
        value = inputs @ parameters["stem_weight"] + parameters["stem_bias"]
        for index in range(depth):
            value = (
                self.block(index, parameters, value, backend=backend)
                if block_runner is None
                else block_runner(index, parameters, value)
            )
        return value @ parameters["readout_weight"] + parameters["readout_bias"]

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
