"""PyTorch ownership and training for the shared hybrid architecture program."""

import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from ..architecture import HybridArchitecture
from ..architecture_runtime import reserve_native_cost
from ..backends import get_backend
from ..distributed_tensor import DistributedTensorPlan
from ..errors import ValidationError
from ..network_training import DepthSchedule, _batch_rows, select_network_input
from ..runtime import CostAwareBackend, PriorityBackend
from ._torch_training import HybridTrainer, _atomic_save, _leaves


class _QuantumExecution:
    def __init__(
        self,
        architecture,
        index,
        *,
        backend="torch",
        target=None,
        dtype=torch.float32,
        device=None,
        mesh=None,
        interconnect=None,
    ):
        from .torch import QuantumLayer

        self.spec = architecture.cells[index].to_spec()
        self.target = target
        self.native_state = architecture.operator_specs[index].implementation.native_state
        if self.native_state:
            from ._channel_execution import ChannelExecution

            if mesh is not None:
                raise ValidationError("Explicit density carries do not support tensor meshes.")
            if isinstance(backend, PriorityBackend):
                route = backend.plan(
                    self.spec.circuit,
                    observables=self.spec.observables,
                    allowed_devices=("GPU", "CPU"),
                )
                self.target = "cuda" if route.device == "GPU" else "cpu"
            elif not isinstance(backend, str) or backend not in {"torch", "cpu", "gpu"}:
                raise ValidationError(
                    "Density carry requires native Torch execution; "
                    "export an unroll for QPU execution."
                )
            elif backend in {"cpu", "gpu"}:
                self.target = "cpu" if backend == "cpu" else "cuda"
            self.layer = ChannelExecution(architecture.cells[index], "torch", self.target)
            self.external, self.distributed = False, False
            return
        self.distributed = mesh is not None and not self.spec.circuit.has_effects
        self.external = not (isinstance(backend, str) and backend == "torch")
        if self.distributed:
            if architecture.native_execution not in {"auto", "tensor-network"}:
                raise ValidationError("A pinned tensor mesh requires tensor-network execution.")
            self.plan = DistributedTensorPlan(
                self.spec.circuit,
                self.spec.observables,
                mesh.size(),
                interconnect,
                max_memory_mb=architecture.max_memory_mb,
                rematerialize=False,
            )
            self.mesh, self.layer = mesh, None
        else:
            # The network's ParameterDict is the sole parameter owner. The
            # execution view receives explicit weights and owns no duplicate
            # quantum Parameter or optimizer state. Preserve caller RNG state.
            execution_device = target if target is not None else device
            devices = (
                []
                if execution_device is None or torch.device(execution_device).type != "cuda"
                else [torch.device(execution_device).index]
            )
            with torch.random.fork_rng(devices=devices):
                self.layer = QuantumLayer.from_spec(
                    self.spec,
                    backend=backend,
                    dtype=dtype,
                    device=execution_device,
                    max_memory_mb=architecture.max_memory_mb,
                    execution_method="auto" if self.external else architecture.native_execution,
                    tensor_options={"reuse_intermediates": True}
                    if not self.external and architecture.native_execution == "tensor-network"
                    else None,
                )
            del self.layer._parameters["weights"]

    def __call__(self, weights, inputs, **scan_options):
        if self.native_state:
            return self.layer(weights, inputs, **scan_options)
        output_device, output_dtype = inputs.device, inputs.dtype
        if self.distributed:
            inputs = inputs.to(dtype=weights.dtype)
            bindings = {name: inputs[..., i] for i, name in enumerate(self.spec.input_parameters)}
            bindings.update(
                {name: weights[i] for i, name in enumerate(self.spec.weight_parameters)}
            )
            with torch.autocast(inputs.device.type, enabled=False):
                return self.plan.evaluate(bindings, mesh=self.mesh, like=weights).to(output_dtype)
        if self.target is not None:
            weights, inputs = weights.to(self.target), inputs.to(self.target)
        return self.layer.functional(weights, inputs).to(device=output_device, dtype=output_dtype)


class HybridNetwork(nn.Module):
    """One stable parameter tree, adaptive projections and executable quantum cells."""

    def __init__(
        self,
        architecture,
        *,
        dtype=None,
        device=None,
        active_depth=None,
        backend=None,
        tensor_mesh=None,
        interconnect=None,
    ):
        super().__init__()
        if not isinstance(architecture, HybridArchitecture):
            raise ValidationError("Provide a HybridArchitecture.")
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if dtype not in {torch.float32, torch.float64}:
            raise ValidationError("Hybrid model weights require float32 or float64.")
        device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        )
        self.architecture = architecture
        self.values = nn.ParameterDict(
            {
                p.name: nn.Parameter(
                    architecture.initialize_one(
                        p.name,
                        lambda s: torch.randn(s, dtype=dtype, device=device),
                        lambda s, v: torch.full(s, v, dtype=dtype, device=device),
                        backend="torch",
                    )
                )
                for p in architecture.parameters
            }
        )
        self.active_depth = architecture.active(active_depth)
        device = next(iter(self.values.values())).device
        self.runtime = get_backend("auto") if backend is None or backend == "auto" else backend
        self.tensor_mesh, self.interconnect = tensor_mesh, interconnect
        if tensor_mesh is not None:
            if isinstance(self.runtime, CostAwareBackend) and (
                self.runtime.ledger is not None
                or self.runtime.policy.require_profiles
                or any(
                    getattr(self.runtime.policy, name) is not None
                    for name in ("max_cost", "max_seconds", "max_rmse")
                )
            ):
                raise ValidationError(
                    "Pinned tensor meshes require caller-owned distributed cost accounting; "
                    "single-device automatic budget profiles cannot describe this execution."
                )
            if interconnect is None:
                raise ValidationError(
                    "Tensor-parallel construction requires calibrated interconnect rates."
                )
            import torch.distributed as dist

            coordinate = tensor_mesh.get_coordinate()
            for axis in range(tensor_mesh.ndim):
                source = list(coordinate)
                source[axis] = 0
                source = int(tensor_mesh.mesh[tuple(source)])
                for parameter in self.values.values():
                    dist.broadcast(parameter.data, src=source, group=tensor_mesh.get_group(axis))
        pinned = "torch" if isinstance(self.runtime, CostAwareBackend) else self.runtime
        self._executions = [
            _QuantumExecution(
                architecture,
                i,
                backend=pinned,
                dtype=dtype,
                device=device,
                mesh=tensor_mesh,
                interconnect=interconnect,
            )
            for i in range(architecture.depth)
        ]
        self.execution_plan = None
        self._route_keys = [None] * architecture.depth
        if isinstance(self.runtime, CostAwareBackend) and tensor_mesh is None:
            self.reschedule()

    def activate_depth(self, depth):
        self.active_depth = self.architecture.active(depth)
        return self

    def reschedule(self, *, workload=None, metadata=None, microbatch_samples=None):
        if not isinstance(self.runtime, CostAwareBackend) or self.tensor_mesh is not None:
            return self.execution_plan
        plan = self.architecture.schedule(
            scheduler=self.runtime,
            workload=workload,
            active_depth=self.active_depth,
            metadata=metadata,
            microbatch_samples=microbatch_samples,
        )
        first = next(iter(self.values.values()))
        replacements = []
        route_keys = []
        for index, stage in zip(
            self.architecture.execution_indices(self.active_depth), plan.stages, strict=True
        ):
            cost = next(c.estimate for c in stage.candidates if c.backend == stage.backend)
            key = (
                stage.backend,
                stage.device,
                id(get_backend(stage.backend)),
                0 if cost is None else cost["shots"],
                None if cost is None else cost["source"],
            )
            route_keys.append(key)
            if key == self._route_keys[index]:
                replacements.append(self._executions[index])
                continue
            backend = stage._backend if stage.device == "QPU" else "torch"
            target = None if stage.device == "QPU" else "cuda" if stage.device == "GPU" else "cpu"
            replacements.append(
                _QuantumExecution(
                    self.architecture,
                    index,
                    backend=backend,
                    target=target,
                    dtype=first.dtype,
                    device=first.device,
                )
            )
        # Install only after the entire network has a feasible plan and all
        # views pass admission. Canonical Parameter objects never change.
        for index, execution, key in zip(
            self.architecture.execution_indices(self.active_depth),
            replacements,
            route_keys,
            strict=True,
        ):
            self._executions[index] = execution
            self._route_keys[index] = key
        self.execution_plan = plan
        return plan

    def forward(self, inputs, *, state=None, return_state=False):
        def run(index, parameters, value, context):
            execution = self._executions[index]
            names = tuple(
                p.name for p in self.architecture.parameters if p.name.startswith(f"block_{index}_")
            )

            def block(x, *weights):
                local = dict(zip(names, weights, strict=True))
                return self.architecture.block(
                    index, local, x, backend="torch", quantum_evaluator=execution, context=context
                )

            weights = tuple(parameters[name] for name in names)
            if (
                self.architecture.rematerialize
                and not execution.external
                and torch.is_grad_enabled()
            ):
                from torch.utils.checkpoint import checkpoint

                return checkpoint(
                    block, value, *weights, use_reentrant=False, preserve_rng_state=False
                )
            return block(value, *weights)

        return self.architecture.evaluate(
            self.values,
            inputs,
            backend="torch",
            active_depth=self.active_depth,
            block_runner=run,
            state=state,
            return_state=return_state,
        )

    def parameter_groups(self, *, lr=1e-3, quantum_lr=None, weight_decay=0):
        quantum_lr = lr if quantum_lr is None else quantum_lr
        groups = {"quantum": [], "decay": [], "plain": []}
        for p in self.architecture.parameters:
            key = (
                "quantum"
                if p.role == "quantum"
                else "plain"
                if p.role in {"bias", "norm", "gate"}
                else "decay"
            )
            groups[key].append(self.values[p.name])
        return [
            {"params": groups["quantum"], "lr": quantum_lr, "weight_decay": 0},
            {"params": groups["decay"], "lr": lr, "weight_decay": weight_decay},
            {"params": groups["plain"], "lr": lr, "weight_decay": 0},
        ]

    def sample_tokens(self, steps, *, prompt=None, batch_size=1, seed=0):
        from ..autoregressive import sample_tokens

        self.architecture.validate_streaming_born()
        return sample_tokens(
            lambda x, state: self(x, state=state, return_state=True),
            self.architecture.output_features,
            steps,
            prompt=prompt,
            batch_size=batch_size,
            seed=seed,
            backend="torch",
            like=self.values["stem_weight"],
            max_memory_mb=self.architecture.max_memory_mb,
        )

    def trainer(self, optimizer=None, *, lr=1e-3, quantum_lr=None, weight_decay=0, **options):
        optimizer = (
            torch.optim.AdamW(
                self.parameter_groups(lr=lr, quantum_lr=quantum_lr, weight_decay=weight_decay)
            )
            if optimizer is None
            else optimizer
        )
        return NetworkTrainer(self, optimizer, **options)

    def get_extra_state(self):
        return {"architecture": self.architecture.fingerprint, "active_depth": self.active_depth}

    def set_extra_state(self, state):
        if state.get("architecture") != self.architecture.fingerprint:
            raise ValidationError("Hybrid architecture checkpoint differs from this model.")
        self.activate_depth(state["active_depth"])


class NetworkTrainer(HybridTrainer):
    def __init__(self, model, optimizer, *, depth_schedule=None, input_selector=None, **options):
        super().__init__(model, optimizer, **options)
        self.network = model.module if self.distributed else model
        if input_selector is not None and not callable(input_selector):
            raise ValidationError("input_selector must be callable.")
        self.input_selector = input_selector
        if not isinstance(self.network, HybridNetwork):
            raise ValidationError("NetworkTrainer requires a HybridNetwork.")
        self.depth_schedule = (
            DepthSchedule(((0, self.network.active_depth),))
            if depth_schedule is None
            else depth_schedule
        )
        if not isinstance(self.depth_schedule, DepthSchedule):
            raise ValidationError("depth_schedule must be DepthSchedule.")
        self.network.architecture.active(self.depth_schedule.stages[-1][1])
        if (
            self.distributed
            and not model.find_unused_parameters
            and not getattr(model, "static_graph", False)
            and self.depth_schedule.stages[0][1] < self.network.architecture.depth
        ):
            raise ValidationError("Progressive DDP requires find_unused_parameters=True.")
        if (
            self.distributed
            and getattr(model, "static_graph", False)
            and len({depth for _, depth in self.depth_schedule.stages}) > 1
        ):
            raise ValidationError("Progressive DDP requires static_graph=False.")

    def gradient_diagnostics(self):
        groups = {"quantum_grad_norm": [], "classical_grad_norm": []}
        for descriptor in self.network.architecture.parameters:
            gradient = self.network.values[descriptor.name].grad
            if gradient is not None:
                key = "quantum_grad_norm" if descriptor.role == "quantum" else "classical_grad_norm"
                groups[key].append(gradient.detach().square().sum())
        return {
            name: 0.0 if not values else float(torch.sqrt(sum(values)).cpu())
            for name, values in groups.items()
        }

    def step(self, batch, loss_function):
        start = time.perf_counter()
        self.network.activate_depth(self.depth_schedule.depth_at(self.updates))
        rows, samples = _batch_rows(_leaves(batch), self.network.architecture.input_features)
        runtime = self.network.runtime
        if isinstance(runtime, CostAwareBackend):
            work = replace(
                runtime.workload,
                batch_size=rows,
                steps=1,
                microbatch_size=self.microbatch_size * max(1, rows // samples),
                gradient_chunk_size=runtime.workload.gradient_chunk_size or 64,
                execution_batches=None,
            )
            inputs = select_network_input(
                batch, self.network.architecture.input_features, self.input_selector
            )
            metadata = None if inputs is None else self.network.architecture.input_metadata(inputs)
            plan = self.network.reschedule(
                workload=work, metadata=metadata, microbatch_samples=self.microbatch_size
            )
            reserve_native_cost(plan, runtime)
        result = super().step(batch, loss_function)
        result["active_depth"] = self.network.active_depth
        result["step_seconds"] = time.perf_counter() - start
        return result

    def state_dict(self):
        state = super().state_dict()
        state["network"] = {
            "fingerprint": self.network.architecture.fingerprint,
            "architecture": self.network.architecture.to_dict(),
            "optimizer_type": type(self.optimizer).__module__
            + "."
            + type(self.optimizer).__qualname__,
            "active_depth": self.network.active_depth,
            "depth_schedule": self.depth_schedule.stages,
        }
        return state

    def load_state_dict(self, state, *, restore_rng=True):
        network = state.get("network", {})
        if network.get("fingerprint") != self.network.architecture.fingerprint:
            raise ValidationError("Training checkpoint architecture differs.")
        optimizer_type = type(self.optimizer).__module__ + "." + type(self.optimizer).__qualname__
        if network.get("optimizer_type") != optimizer_type:
            raise ValidationError("Training checkpoint optimizer type differs.")
        schedule = DepthSchedule(network["depth_schedule"])
        self.network.architecture.active(schedule.stages[-1][1])
        self.network.architecture.active(network["active_depth"])
        current = self.network.state_dict()
        extra = state["model"].get("_extra_state", {})
        if extra.get("architecture") != self.network.architecture.fingerprint:
            raise ValidationError("Model checkpoint architecture differs.")
        self.network.architecture.active(extra["active_depth"])
        if set(current) != set(state["model"]) or any(
            isinstance(value, torch.Tensor)
            and (
                not isinstance(state["model"][name], torch.Tensor)
                or value.shape != state["model"][name].shape
            )
            for name, value in current.items()
        ):
            raise ValidationError("Training checkpoint parameter layout differs.")
        super().load_state_dict(state, restore_rng=restore_rng)
        self.depth_schedule = schedule
        self.network.activate_depth(network["active_depth"])
        return self

    def save(self, path):
        if self.network.tensor_mesh is None:
            return super().save(path)
        rank = torch.distributed.get_rank()
        return _atomic_save(self.state_dict(), Path(path) / f"rank-{rank}.pt")

    def load(self, path, *, restore_rng=True):
        if self.network.tensor_mesh is not None:
            path = Path(path) / f"rank-{torch.distributed.get_rank()}.pt"
        return super().load(path, restore_rng=restore_rng)
