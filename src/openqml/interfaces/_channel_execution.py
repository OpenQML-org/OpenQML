"""Parameter-free native views for explicit density carry execution."""

from ..errors import ValidationError
from ..quantum_channels import QuantumChannelPlan


class ChannelExecution:
    external = False

    def __init__(self, plan, backend, target=None):
        if not isinstance(plan, QuantumChannelPlan) or backend not in {"torch", "jax"}:
            raise ValidationError("Quantum memory requires a native Torch or JAX channel view.")
        self.plan, self.execution_backend, self.target = plan, backend, target
        self.spec = plan.to_spec()

    def __call__(self, weights, controls, **options):
        if self.execution_backend == "torch":
            device, dtype = controls.device, controls.dtype
            target = self.target
            if target is not None:
                weights, controls = weights.to(target), controls.to(target)
                options = {k: v.to(target) if hasattr(v, "to") else v for k, v in options.items()}
            outputs, state = self.plan.scan(
                weights, controls.to(weights.dtype), backend="torch", **options
            )
            return outputs.to(device=device, dtype=dtype), state.to(device=device)
        return self.plan.scan(weights, controls, backend="jax", **options)
