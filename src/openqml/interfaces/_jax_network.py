"""JAX execution and stagewise optimization of the shared hybrid model program."""

import copy
from dataclasses import replace

import jax
import jax.numpy as jnp

from ..architecture import HybridArchitecture
from ..architecture_runtime import reserve_native_cost
from ..backends import get_backend
from ..errors import ValidationError
from ..network_training import DepthSchedule, StagewiseOptimizer, _batch_rows, select_network_input
from ..runtime import CostAwareBackend
from ._jax_backend import JAXNativeBackend
from ._jax_training import make_train_step


class HybridNetwork:
    def __init__(self, architecture, *, active_depth=None, backend=None):
        if not isinstance(architecture, HybridArchitecture):
            raise ValidationError("Provide a HybridArchitecture.")
        from .jax import QuantumLayer

        self.architecture = architecture
        self.active_depth = architecture.active(active_depth)
        self._native_candidate = JAXNativeBackend()
        self.runtime = get_backend("auto") if backend is None or backend == "auto" else backend
        pinned = "jax" if isinstance(self.runtime, CostAwareBackend) else self.runtime
        self._layers = tuple(
            QuantumLayer.from_spec(
                cell.to_spec(),
                backend=pinned,
                max_memory_mb=architecture.max_memory_mb,
                execution_method=architecture.native_execution if pinned == "jax" else "auto",
                tensor_options={"reuse_intermediates": True}
                if pinned == "jax" and architecture.native_execution == "tensor-network"
                else None,
            )
            for cell in architecture.cells
        )
        self.execution_plan, self._route_keys = None, None
        if isinstance(self.runtime, CostAwareBackend):
            routed = self.reschedule()
            self._layers, self.execution_plan, self._route_keys = (
                routed._layers,
                routed.execution_plan,
                routed._route_keys,
            )

    def init(self, key, *, dtype=jnp.float32):
        if jnp.dtype(dtype) not in {jnp.dtype("float32"), jnp.dtype("float64")}:
            raise ValidationError("Hybrid parameters require float32 or float64.")
        keys = jax.random.split(key, len(self.architecture.parameters))
        return {
            p.name: self.architecture.initialize_one(
                p.name,
                lambda s, key=k: jax.random.normal(key, s, dtype=dtype),
                lambda s, v: jnp.full(s, v, dtype=dtype),
                backend="jax",
            )
            for p, k in zip(self.architecture.parameters, keys, strict=True)
        }

    def with_depth(self, depth):
        depth = self.architecture.active(depth)
        if depth == self.active_depth:
            return self
        value = copy.copy(self)
        value.active_depth = depth
        return value

    def reschedule(self, *, workload=None, metadata=None, microbatch_samples=None):
        if not isinstance(self.runtime, CostAwareBackend):
            return self
        from .jax import QuantumLayer

        platform = "GPU" if jax.default_backend() == "gpu" else "CPU"
        plan = self.architecture.schedule(
            scheduler=self.runtime,
            workload=workload,
            active_depth=self.active_depth,
            native_devices=(platform,),
            metadata=metadata,
            microbatch_samples=microbatch_samples,
            native_candidates=(self._native_candidate,) if platform == "GPU" else (),
        )
        keys = tuple(
            (
                s.backend,
                s.device,
                id(self._native_candidate)
                if s.backend == self._native_candidate.name
                else id(get_backend(s.backend)),
                next(
                    c.estimate.get("shots", 0) if c.estimate else 0
                    for c in s.candidates
                    if c.backend == s.backend
                ),
            )
            for s in plan.stages
        )
        if keys == self._route_keys:
            self.execution_plan = plan
            return self
        value = copy.copy(self)
        layers = list(self._layers)
        for index, stage in enumerate(plan.stages):
            # A single JIT graph retains its native JAX platform. Other devices
            # use the explicit numeric bridge, whose shift cost was quoted.
            backend = "jax" if stage.device == platform else stage._backend
            layers[index] = QuantumLayer.from_spec(
                self.architecture.cells[index].to_spec(),
                backend=backend,
                max_memory_mb=self.architecture.max_memory_mb,
                execution_method=self.architecture.native_execution if backend == "jax" else "auto",
                tensor_options={"reuse_intermediates": True}
                if backend == "jax" and self.architecture.native_execution == "tensor-network"
                else None,
            )
        value._layers, value.execution_plan, value._route_keys = tuple(layers), plan, keys
        return value

    def __call__(self, parameters, inputs):
        def run(index, values, x, context):
            layer = self._layers[index]
            names = tuple(
                p.name for p in self.architecture.parameters if p.group == f"block_{index}"
            )
            local = {name: values[name] for name in names}

            def block(weights, features):
                return self.architecture.block(
                    index,
                    weights,
                    features,
                    backend="jax",
                    quantum_evaluator=layer,
                    context=context,
                )

            return (
                jax.checkpoint(block)(local, x)
                if self.architecture.rematerialize and layer.execution_backend == "jax"
                else block(local, x)
            )

        return self.architecture.evaluate(
            parameters,
            jax.tree_util.tree_map(jnp.asarray, inputs),
            backend="jax",
            active_depth=self.active_depth,
            block_runner=run,
        )

    def optimizer(self, optimizer):
        return StagewiseOptimizer(self.architecture, optimizer, self.active_depth)

    def make_train_step(self, optimizer, loss_function, **options):
        options.setdefault("gradient_metrics", self.architecture.gradient_metrics)
        return make_train_step(self.optimizer(optimizer), loss_function, **options)

    def trainer(
        self, optimizer, loss_function, *, parameters=None, key=None, depth_schedule=None, **options
    ):
        return NetworkTrainer(
            self,
            optimizer,
            loss_function,
            parameters=parameters,
            key=key,
            depth_schedule=depth_schedule,
            **options,
        )


class NetworkTrainer:
    """Host-side curriculum boundaries around pure, cached JAX update functions."""

    def __init__(
        self,
        model,
        optimizer,
        loss_function,
        *,
        parameters=None,
        key=None,
        depth_schedule=None,
        input_selector=None,
        **options,
    ):
        if not isinstance(model, HybridNetwork) or not callable(loss_function):
            raise ValidationError("Provide a HybridNetwork and loss(model, params, batch).")
        if parameters is None:
            if key is None:
                raise ValidationError("Provide initial parameters or a JAX key.")
            parameters = model.init(key)
        model.architecture.validate_parameters(parameters)
        self.model, self.optimizer, self.loss_function = model, optimizer, loss_function
        self.parameters = parameters
        self.optimizer_state = model.optimizer(optimizer).init(parameters)
        self.depth_schedule = (
            DepthSchedule(((0, model.active_depth),)) if depth_schedule is None else depth_schedule
        )
        if not isinstance(self.depth_schedule, DepthSchedule):
            raise ValidationError("depth_schedule must be DepthSchedule.")
        model.architecture.active(self.depth_schedule.stages[-1][1])
        self.options, self.updates, self.attempts = options, 0, 0
        if input_selector is not None and not callable(input_selector):
            raise ValidationError("input_selector must be callable.")
        self.input_selector = input_selector
        self._compiled, self._compiled_model = None, None

    def step(self, batch):
        model = self.model.with_depth(self.depth_schedule.depth_at(self.updates))
        rows, samples = _batch_rows(
            jax.tree_util.tree_leaves(batch), model.architecture.input_features
        )
        if isinstance(model.runtime, CostAwareBackend):
            work = replace(
                model.runtime.workload,
                batch_size=rows,
                steps=1,
                microbatch_size=self.options.get("microbatch_size", 32) * max(1, rows // samples),
                gradient_chunk_size=model.runtime.workload.gradient_chunk_size or 64,
                execution_batches=None,
            )
            inputs = select_network_input(
                batch, model.architecture.input_features, self.input_selector
            )
            metadata = None if inputs is None else model.architecture.input_metadata(inputs)
            model = model.reschedule(
                workload=work,
                metadata=metadata,
                microbatch_samples=self.options.get("microbatch_size", 32),
            )
            reserve_native_cost(
                model.execution_plan,
                model.runtime,
                native_devices=("GPU" if jax.default_backend() == "gpu" else "CPU",),
            )
        self.model = model
        if model is not self._compiled_model:
            self._compiled = model.make_train_step(
                self.optimizer, lambda p, b: self.loss_function(model, p, b), **self.options
            )
            self._compiled_model = model
        self.parameters, self.optimizer_state, metrics = self._compiled(
            self.parameters, self.optimizer_state, batch
        )
        self.attempts += 1
        self.updates += int(not bool(metrics["skipped"]))
        return {**metrics, "active_depth": model.active_depth, "updates": self.updates}

    @property
    def variables(self):
        return {"params": self.parameters}

    def save(self, path):
        from ._network_checkpoint import save_network

        return save_network(self, path)

    def load(self, path):
        from ._network_checkpoint import load_network

        return load_network(self, path)
