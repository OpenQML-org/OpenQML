"""AdamW/QNM optimization with bounded, backward-selected empirical geometry."""

import math
import threading
import weakref
from collections.abc import Mapping

import torch

from ...autodiff.distance_aware import DistanceAwareQNM, DistanceAwareState
from ...autodiff.fisher import quantum_fisher
from ...autodiff.geometry import quantum_metric
from ...compute.kernel import get_compute_kernel
from ...errors import ValidationError
from ...ir.parameters import real


def _remove_hooks(handles):
    for handle in handles:
        handle.remove()
    handles.clear()


class HybridQNM(torch.optim.AdamW):
    """AdamW classical groups and QFIM/momentum/polar quantum groups.

    Native QuantumLayer weight vectors use pure-state or noisy-state geometry.
    Backward-selected inputs span microbatches. DDP reduces sample-weighted
    metric products, without a dense parameter matrix. Input tensors must not
    mutate between forward and backward. Use state_dict for portable state.
    """

    def __init__(
        self,
        model,
        *,
        lr=1e-3,
        quantum_lr=0.01,
        block_shapes=None,
        quantum_options=None,
        fisher_options=None,
        weight_decay=0.01,
        **adam_options,
    ):
        from .torch import QuantumLayer

        if not isinstance(model, torch.nn.Module):
            raise ValidationError("HybridQNM requires a Torch module.")
        lr, quantum_lr = real(lr), real(quantum_lr)
        if lr < 0 or quantum_lr <= 0:
            raise ValidationError("Use nonnegative classical and positive quantum learning rates.")
        if any(
            adam_options.get(n, False)
            for n in ("capturable", "differentiable", "fused", "maximize")
        ):
            raise ValidationError("HybridQNM requires eager, minimizing optimizer steps.")
        if quantum_options is not None and not isinstance(quantum_options, Mapping):
            raise ValidationError("quantum_options must be a mapping.")
        options = dict(quantum_options or {})
        if fisher_options is not None and not isinstance(fisher_options, Mapping):
            raise ValidationError("fisher_options must be a mapping.")
        fisher = dict(fisher_options or {})
        if set(fisher) - {"state_damping", "max_branches", "max_work"}:
            raise ValidationError("Use fisher_options for state_damping, max_branches or max_work.")
        if set(options) & {"plan", "block_shapes", "learning_rate"}:
            raise ValidationError("Use block_shapes and quantum_lr for group layout and rates.")
        layers = [
            (name, layer)
            for name, layer in model.named_modules()
            if isinstance(layer, QuantumLayer)
        ]
        if not layers:
            raise ValidationError("HybridQNM needs at least one owned native QuantumLayer.")
        if block_shapes is not None and not isinstance(block_shapes, Mapping) and len(layers) != 1:
            raise ValidationError(
                "For multiple quantum layers, map module names to spectral shapes."
            )
        if isinstance(block_shapes, Mapping) and set(block_shapes) - {n for n, _ in layers}:
            raise ValidationError("Spectral block mapping contains unknown quantum module names.")
        self._quantum = []
        quantum_ids = set()
        for name, layer in layers:
            if layer.execution_backend != "torch":
                raise ValidationError("HybridQNM needs native Torch layers.")
            parameter = layer.weights
            if parameter.dtype not in (torch.float32, torch.float64):
                raise ValidationError("Use float32/float64 quantum master weights with AMP.")
            if id(parameter) in quantum_ids:
                raise ValidationError("Distinct quantum modules cannot share a weight Parameter.")
            names = tuple(n for n in layer.weight_parameters if n in layer.circuit.parameters)
            if not names:
                continue
            if parameter.ndim != 1 or parameter.shape[0] != len(layer.weight_parameters):
                raise ValidationError("Quantum weight vectors must preserve circuit ownership.")
            positions = tuple(layer.weight_parameters.index(n) for n in names)
            factory = quantum_fisher if layer.circuit.has_effects else quantum_metric
            plan = factory(
                layer.circuit,
                parameters=names,
                max_memory_mb=layer.limits["max_memory_mb"],
                max_qubits=layer.limits["max_qubits"],
                **(fisher if layer.circuit.has_effects else {}),
            )
            shapes = (
                block_shapes.get(name) if isinstance(block_shapes, Mapping) else block_shapes
            ) or ((1, len(names)),)
            geometry = DistanceAwareQNM(plan, shapes, learning_rate=quantum_lr, **options)
            self._quantum.append((name, layer, parameter, positions, geometry))
            quantum_ids.add(id(parameter))
        if not self._quantum:
            raise ValidationError("No active quantum weight coordinates were found.")
        classical = [p for p in model.parameters() if id(p) not in quantum_ids]
        groups = (
            [{"params": classical, "lr": lr, "weight_decay": weight_decay}] if classical else []
        )
        groups += [
            {"params": [p], "lr": quantum_lr, "weight_decay": 0, "openqml_quantum": name}
            for name, _, p, _, _ in self._quantum
        ]
        super().__init__(groups, lr=lr, weight_decay=weight_decay, **adam_options)
        self._distributed = isinstance(model, torch.nn.parallel.DistributedDataParallel)
        self._group = model.process_group if self._distributed else None
        self._lock = threading.RLock()
        self._generation = 0
        self._capture = {}
        self._captured_bytes = 0
        self._closed = False
        handles = []
        owner = weakref.ref(self)
        for name, layer, _, _, _ in self._quantum:

            def hook(module, args, kwargs, output, name=name):
                optimizer = owner()
                if (
                    optimizer is None
                    or optimizer._closed
                    or not isinstance(output, torch.Tensor)
                    or not output.requires_grad
                ):
                    return
                inputs = args[0] if args else kwargs.get("inputs")
                generation = optimizer._generation
                handle = []

                def capture(_gradient):
                    handle.pop().remove()
                    optimizer = owner()
                    if (
                        optimizer is None
                        or optimizer._closed
                        or generation != optimizer._generation
                    ):
                        return
                    if not isinstance(inputs, torch.Tensor):
                        raise ValidationError("Geometry capture requires Tensor inputs.")
                    count = math.prod(inputs.shape[:-1])
                    with optimizer._lock:
                        needed = count * inputs.shape[-1] * inputs.element_size()
                        get_compute_kernel().admit(
                            optimizer._captured_bytes + needed,
                            module.limits["max_memory_mb"],
                            itemsize=1,
                        )
                        values = inputs.detach().reshape(count, inputs.shape[-1]).clone()
                        optimizer._capture.setdefault(name, []).append(values)
                        optimizer._captured_bytes += needed

                handle.append(output.register_hook(capture))

            handles.append(layer.register_forward_hook(hook, with_kwargs=True))
        self._finalize = weakref.finalize(self, _remove_hooks, handles)

    def close(self):
        self._closed = True
        self._capture.clear()
        self._finalize()

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        if hasattr(self, "_capture"):
            self._capture.clear()
            self._captured_bytes = 0
            self._generation += 1

    @torch.no_grad()
    def step(self, closure=None):
        if self._closed:
            raise ValidationError("This HybridQNM optimizer is closed.")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        proposals = []
        for name, layer, p, positions, geometry in self._quantum:
            present = torch.tensor(int(p.grad is not None), device=p.device, dtype=torch.int64)
            gradient = torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
            if self._distributed:
                torch.distributed.all_reduce(present, group=self._group)
                torch.distributed.all_reduce(gradient, group=self._group)
                gradient = gradient / present.clamp_min(1)
            if not bool(present):
                continue
            if not torch.isfinite(gradient).all():
                raise ValidationError("Quantum gradients must be finite before updates.")
            captured = self._capture.get(name, [])
            count = sum(v.shape[0] for v in captured)
            total = torch.tensor(count, device=p.device, dtype=torch.int64)
            if self._distributed:
                torch.distributed.all_reduce(total, group=self._group)
            if not bool(total > 0):
                raise ValidationError(
                    "No geometry inputs were captured; call the owned layer's forward "
                    "before backward."
                )
            inputs = (
                torch.cat(captured)
                if count
                else torch.zeros((1, len(layer.input_parameters)), dtype=p.dtype, device=p.device)
            )
            bindings = {
                n: inputs[:, i]
                for i, n in enumerate(layer.input_parameters)
                if n in layer.circuit.parameters
            }
            bindings.update(
                (n, p.detach()[i])
                for i, n in enumerate(layer.weight_parameters)
                if n in layer.circuit.parameters
            )
            carry = self.state[p].get("openqml_qnm")
            state = (
                None
                if carry is None
                else DistanceAwareState(
                    carry["initial"],
                    carry["momentum"],
                    carry["distance"],
                    torch.tensor(carry["steps"], dtype=torch.int32, device=p.device),
                    torch.tensor(carry["signature"], dtype=torch.int32, device=p.device),
                )
            )
            group = next(g for g in self.param_groups if g.get("openqml_quantum") == name)

            def reducer(value, count=count, total=total):
                value = value * (count / total.to(dtype=value.dtype))
                torch.distributed.all_reduce(value, group=self._group)
                return value

            result = geometry.step(
                bindings,
                gradient[list(positions)],
                state=state,
                learning_rate=group["lr"],
                metric_reducer=reducer if self._distributed else None,
            )
            proposals.append(
                (p, positions, result, geometry, 0 if carry is None else carry["steps"])
            )
        saved = [(p, p.grad) for _, _, p, _, _ in self._quantum]
        try:
            for p, _ in saved:
                p.grad = None
            super().step()
        finally:
            for p, gradient in saved:
                p.grad = gradient
        for p, positions, result, geometry, steps in proposals:
            update = torch.zeros_like(p)
            update[list(positions)] = result.step
            p.sub_(update)
            self.state[p]["openqml_qnm"] = {
                "initial": result.state.initial_parameters,
                "momentum": result.state.momentum,
                "distance": result.state.max_distance,
                "steps": steps + 1,
                "signature": geometry._signature,
            }
            # AdamW's checkpoint loader expects the shared scalar counter for
            # every nonempty parameter state, including custom quantum groups.
            self.state[p]["step"] = torch.tensor(steps + 1, dtype=torch.float64)
        self._capture.clear()
        self._captured_bytes = 0
        self._generation += 1
        return loss

    def state_dict(self):
        state = super().state_dict()
        state["openqml_geometry"] = {
            "schema": "openqml.hybrid_qnm.v1",
            "groups": [(n, g._signature) for n, _, _, _, g in self._quantum],
        }
        return state

    def validate_state_dict(self, state_dict):
        expected = {
            "schema": "openqml.hybrid_qnm.v1",
            "groups": [(n, g._signature) for n, _, _, _, g in self._quantum],
        }
        if not isinstance(state_dict, Mapping) or state_dict.get("openqml_geometry") != expected:
            raise ValidationError("HybridQNM checkpoint circuit/configuration identity differs.")

    def load_state_dict(self, state_dict):
        self.validate_state_dict(state_dict)
        super().load_state_dict(state_dict)
        self.zero_grad()
