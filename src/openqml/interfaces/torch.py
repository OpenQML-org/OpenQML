"""PyTorch nn.Module quantum layer; gradients stay on the native device."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:
    from ..errors import MissingDependencyError

    raise MissingDependencyError("Install openqml[torch] for QuantumLayer.") from exc

from ..errors import ValidationError
from ..simulator import _resource_check, expectations
from ._spec import LayerSpecMixin, broadcast_batch_shape, prepare_layer


class QuantumLayer(LayerSpecMixin, nn.Module):
    """Trainable (..., n_inputs) -> (..., n_observables) circuit.

    Use matching input/weight devices. optimize=True preserves source ownership;
    circuit holds executable IR and optimization holds the pass report.
    """

    def __init__(
        self,
        circuit,
        input_parameters,
        weight_parameters,
        observables=None,
        *,
        dtype=None,
        device=None,
        init_scale=0.1,
        max_qubits=20,
        max_memory_mb=256,
        backend="torch",
        batch_chunk_size=None,
        gradient_chunk_size=64,
        max_branches=256,
        optimize=False,
        optimization_options=None,
    ):
        super().__init__()
        from ..ir import integer

        integer(max_branches, "max_branches", 1)
        self.source_circuit = circuit
        (
            self.input_parameters,
            self.weight_parameters,
            self.observables,
            circuit,
            self.optimization,
        ) = prepare_layer(
            circuit,
            input_parameters,
            weight_parameters,
            observables,
            optimize=optimize,
            optimization_options=optimization_options,
        )
        if backend == "torch":
            if circuit.has_effects:
                from ..density import _planning_limit
                from ..density import _resource_check as density_check
                from ..planning import plan_density

                density_check(
                    circuit,
                    max_qubits,
                    max_memory_mb,
                    branches=plan_density(
                        circuit,
                        max_branches=_planning_limit(
                            circuit, max_qubits, max_memory_mb, max_branches=max_branches
                        ),
                    ).peak_branches,
                )
            else:
                _resource_check(circuit, max_qubits, max_memory_mb)
        self.execution_backend = backend
        self.batch_chunk_size = (
            None if batch_chunk_size is None else integer(batch_chunk_size, "batch_chunk_size", 1)
        )

        self.gradient_chunk_size = integer(gradient_chunk_size, "gradient_chunk_size", 2)
        self.circuit = circuit
        self._active_parameters = frozenset(circuit.parameters)
        self.limits = {
            "max_qubits": max_qubits,
            "max_memory_mb": max_memory_mb,
            "max_branches": max_branches,
        }
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if dtype not in {torch.float32, torch.float64}:
            raise ValidationError("QuantumLayer requires float32 or float64 weights.")
        from ..parameters import real

        init_scale = real(init_scale)
        if init_scale < 0:
            raise ValidationError("init_scale must be nonnegative.")
        self.weights = nn.Parameter(
            torch.randn(len(self.weight_parameters), dtype=dtype, device=device) * init_scale
        )

    @classmethod
    def from_checkpoint(cls, checkpoint, **kwargs):
        """Restore a module; dtype/device overrides explicitly select the destination."""
        from ..layers import LayerCheckpoint

        if not isinstance(checkpoint, LayerCheckpoint):
            raise ValidationError("Expected a LayerCheckpoint.")
        if kwargs.get("dtype") is None:
            kwargs["dtype"] = getattr(torch, checkpoint.dtype)
        layer = cls.from_spec(checkpoint.spec, **kwargs)
        return layer.load_checkpoint(checkpoint, allow_cast=True)

    def load_checkpoint(self, checkpoint, *, allow_cast=False):
        """Copy weights in place after full configuration and precision checks."""
        from ..layers import LayerCheckpoint
        from ._spec import checkpoint_values

        if type(allow_cast) is not bool or not isinstance(checkpoint, LayerCheckpoint):
            raise ValidationError("Use a LayerCheckpoint and boolean allow_cast.")
        dtype = str(self.weights.dtype).removeprefix("torch.")
        if not allow_cast and dtype != checkpoint.dtype:
            raise ValidationError("Checkpoint precision differs; set allow_cast=True explicitly.")
        values = checkpoint_values(checkpoint, self.to_spec(), dtype)
        # Validate the complete checkpoint before changing a caller-owned parameter.
        with torch.no_grad():
            self.weights.copy_(
                torch.as_tensor(values, dtype=self.weights.dtype, device=self.weights.device)
            )
        return self

    def forward(self, inputs):
        return self.functional(self.weights, inputs)

    def functional(self, weights, inputs):
        """Evaluate explicit (..., n_weights) weights without changing module parameters.

        Leading weight/input dimensions broadcast. This supports classical weight
        generators, parameter sharing and ensembles in the same autograd graph.
        """
        if (
            not isinstance(weights, torch.Tensor)
            or weights.ndim < 1
            or weights.shape[-1] != len(self.weight_parameters)
        ):
            raise ValidationError(f"Expected weight shape (..., {len(self.weight_parameters)}).")
        if weights.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            raise ValidationError("Weights must use a floating-point dtype.")
        if (
            not isinstance(inputs, torch.Tensor)
            or inputs.ndim < 1
            or inputs.shape[-1] != len(self.input_parameters)
        ):
            raise ValidationError(f"Expected input shape (..., {len(self.input_parameters)}).")
        if inputs.device != weights.device:
            raise ValidationError("Inputs and layer weights must share a device.")
        if inputs.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            raise ValidationError("Inputs must use a floating-point dtype.")
        output_dtype = inputs.dtype
        computation_weights = (
            weights if weights.dtype in {torch.float32, torch.float64} else weights.float()
        )
        inputs = inputs.to(dtype=computation_weights.dtype)
        shape = broadcast_batch_shape(computation_weights, inputs)
        import math

        # reshape(-1, 0) is ambiguous; compute row count from the batch dimensions.
        count = math.prod(shape)
        if count == 0:
            return (
                torch.zeros(
                    (*shape, len(self.observables)), dtype=inputs.dtype, device=inputs.device
                )
                + (inputs * 0).square().sum()
                + (computation_weights * 0).square().sum()
            ).to(output_dtype)
        if self.batch_chunk_size is not None and count > self.batch_chunk_size:
            flat = inputs.expand(shape + (len(self.input_parameters),)).reshape(
                count, len(self.input_parameters)
            )
            flat_weights = computation_weights.expand(
                shape + (len(self.weight_parameters),)
            ).reshape(count, len(self.weight_parameters))
            # Each call owns at most one chunk of simulator states. Autograd
            # retains the required activations and accumulates shared weights.
            return (
                torch.cat(
                    [
                        self.functional(w, x)
                        for w, x in zip(
                            flat_weights.split(self.batch_chunk_size),
                            flat.split(self.batch_chunk_size),
                            strict=True,
                        )
                    ],
                    dim=0,
                )
                .reshape(*shape, len(self.observables))
                .to(output_dtype)
            )
        if self.execution_backend != "torch":
            return _ExternalQuantumFunction.apply(inputs, computation_weights, self).to(
                output_dtype
            )
        bindings = {
            name: inputs[..., i]
            for i, name in enumerate(self.input_parameters)
            if name in self._active_parameters
        }
        bindings.update(
            (name, computation_weights[..., i])
            for i, name in enumerate(self.weight_parameters)
            if name in self._active_parameters
        )
        result = expectations(
            self.circuit,
            self.observables,
            bindings,
            backend="torch",
            like=computation_weights,
            batch_shape=shape,
            **self.limits,
        )
        removed = self.optimization is not None and self.optimization.removed_parameters
        if removed or not self.input_parameters:
            # Multiply before squaring: finite unused values cannot overflow.
            # Quadratic zero retains a graph for second-order autodiff as well.
            result = result + (inputs * 0).square().sum(-1, keepdim=True)
        if removed or not self.weight_parameters:
            result = result + (computation_weights * 0).square().sum(-1, keepdim=True)
        return result.to(output_dtype)

    def extra_repr(self):
        return (
            f"qubits={self.circuit.num_qubits}, inputs={len(self.input_parameters)}, "
            f"weights={len(self.weight_parameters)}, outputs={len(self.observables)}"
        )


class _ExternalQuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, layer):
        import numpy as np

        from ..batch import expectations_batch

        ctx.layer = layer
        ctx.save_for_backward(inputs, weights)
        ctx.save_for_forward(inputs, weights)
        rows = _binding_rows(layer, inputs, weights)
        output = expectations_batch(
            layer.circuit, layer.observables, rows, backend=layer.execution_backend, **layer.limits
        )
        shape = broadcast_batch_shape(weights, inputs)
        output = np.asarray(output).reshape(shape + (len(layer.observables),))
        return torch.as_tensor(output, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    def jvp(ctx, input_tangent, weight_tangent, _layer_tangent):
        from ._spec import numeric_jvp

        inputs, weights = ctx.saved_tensors
        dx = torch.zeros_like(inputs) if input_tangent is None else input_tangent
        dw = torch.zeros_like(weights) if weight_tangent is None else weight_tangent
        value = numeric_jvp(
            ctx.layer,
            weights.detach().cpu().numpy(),
            inputs.detach().cpu().numpy(),
            dw.detach().cpu().numpy(),
            dx.detach().cpu().numpy(),
        )
        return torch.as_tensor(value, dtype=inputs.dtype, device=inputs.device)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        from ._spec import numeric_vjp

        inputs, weights = ctx.saved_tensors
        layer = ctx.layer
        dw, dx = numeric_vjp(
            layer,
            weights.detach().cpu().numpy(),
            inputs.detach().cpu().numpy(),
            grad_output.detach().cpu().numpy(),
            input_grad=ctx.needs_input_grad[0],
            weight_grad=ctx.needs_input_grad[1],
        )
        return (
            None if dx is None else torch.as_tensor(dx, dtype=inputs.dtype, device=inputs.device),
            None if dw is None else torch.as_tensor(dw, dtype=weights.dtype, device=weights.device),
            None,
        )


def _binding_rows(layer, inputs, weights):
    from ._spec import numeric_rows

    return numeric_rows(layer, weights.detach().cpu().numpy(), inputs.detach().cpu().numpy())


class NativePennyLaneLayer(nn.Module):
    """Keep the native QNode, templates, devices/readouts and its own autodiff graph."""

    def __init__(self, qnode, weight_shapes, *, init_method=None):
        super().__init__()
        from ..adapters._common import dependency

        qml = dependency("pennylane", "pennylane")
        self.native = qml.qnn.TorchLayer(qnode, weight_shapes, init_method=init_method)

    def forward(self, inputs):
        return self.native(inputs)


class QuantumAdapter(nn.Module):
    """Residual feature adapter for MLP/CNN/Transformer tensors (..., embedding_dim)."""

    def __init__(self, features, *, num_qubits=4, depth=2, dtype=None, device=None):
        super().__init__()
        self.encoder = nn.Linear(features, num_qubits, dtype=dtype, device=device)
        self.quantum = QuantumLayer.hardware_efficient(
            num_qubits, depth, dtype=dtype, device=device
        )
        self.projection = nn.Linear(num_qubits, features, dtype=dtype, device=device)
        nn.init.normal_(self.projection.weight, std=0.01)
        nn.init.zeros_(self.projection.bias)

    def forward(self, inputs):
        return inputs + self.projection(self.quantum(torch.tanh(self.encoder(inputs))))


class QuantumConv2d(nn.Module):
    """Trainable image-patch encoder -> shared quantum circuit -> channel projection.

    Accepts the PyTorch (N, C, H, W) convention. patch_chunk_size bounds the
    quantum state batch for large images without evaluating patches one by one.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        *,
        stride=2,
        padding=0,
        num_qubits=4,
        depth=2,
        patch_chunk_size=128,
        dtype=None,
        device=None,
        backend="torch",
        max_memory_mb=256,
    ):
        super().__init__()
        from ..ir import integer

        self.in_channels = integer(in_channels, "in_channels", 1)
        self.out_channels = integer(out_channels, "out_channels", 1)
        self.kernel_size = integer(kernel_size, "kernel_size", 1)
        self.stride = integer(stride, "stride", 1)
        self.padding = integer(padding, "padding")
        self.patch_chunk_size = integer(patch_chunk_size, "patch_chunk_size", 1)
        self.encoder = nn.Linear(
            in_channels * kernel_size**2, num_qubits, dtype=dtype, device=device
        )
        self.quantum = QuantumLayer.hardware_efficient(
            num_qubits,
            depth,
            dtype=dtype,
            device=device,
            backend=backend,
            max_memory_mb=max_memory_mb,
        )
        self.projection = nn.Linear(num_qubits, out_channels, dtype=dtype, device=device)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValidationError(f"Expected image shape (N, {self.in_channels}, H, W).")
        height = (inputs.shape[-2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        width = (inputs.shape[-1] + 2 * self.padding - self.kernel_size) // self.stride + 1
        if min(height, width) < 1:
            raise ValidationError("The convolution kernel is larger than the padded input.")
        patches = nn.functional.unfold(
            inputs, self.kernel_size, padding=self.padding, stride=self.stride
        ).transpose(1, 2)
        flat = patches.reshape(-1, self.in_channels * self.kernel_size**2)
        encoded = torch.tanh(self.encoder(flat))
        chunks = [
            self.quantum(encoded[start : start + self.patch_chunk_size])
            for start in range(0, encoded.shape[0], self.patch_chunk_size)
        ]
        quantum = torch.cat(chunks, dim=0) if chunks else self.quantum(encoded)
        projected = self.projection(quantum).reshape(
            inputs.shape[0], height * width, self.out_channels
        )
        return projected.transpose(1, 2).reshape(inputs.shape[0], self.out_channels, height, width)
