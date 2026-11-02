"""Depth curricula and optimizers whose inactive stage state remains untouched."""

from dataclasses import dataclass

from .errors import ValidationError
from .ir import integer


@dataclass(frozen=True)
class DepthSchedule:
    stages: tuple

    def __post_init__(self):
        rows = tuple(tuple(row) for row in self.stages)
        if not rows or any(len(row) != 2 for row in rows):
            raise ValidationError("DepthSchedule requires (successful_updates, active_depth) rows.")
        for update, depth in rows:
            integer(update, "successful_updates", 0)
            integer(depth, "active_depth", 0)
        if rows[0][0] != 0 or any(
            b[0] <= a[0] or b[1] < a[1] for a, b in zip(rows, rows[1:], strict=False)
        ):
            raise ValidationError("Depth schedules start at update zero and grow monotonically.")
        object.__setattr__(self, "stages", rows)

    def depth_at(self, updates):
        return next(depth for step, depth in reversed(self.stages) if updates >= step)


class StagewiseOptimizer:
    """Optax-compatible grouping with independent activation ages per block.

    Inactive updates are zero and their optimizer states are returned unchanged,
    including Adam moments and bias-correction counters. Shared layers continue
    training. Use a new wrapper at each curriculum boundary with the same state.
    """

    def __init__(self, architecture, optimizer, active_depth):
        if not callable(getattr(optimizer, "init", None)) or not callable(
            getattr(optimizer, "update", None)
        ):
            raise ValidationError("Provide an optimizer implementing init/update.")
        self.architecture, self.optimizer = architecture, optimizer
        self.active_depth = architecture.active(active_depth)
        self.groups = {
            group: tuple(p.name for p in architecture.parameters if p.group == group)
            for group in ("shared", *(f"block_{i}" for i in range(architecture.depth)))
        }

    def init(self, parameters):
        self.architecture.validate_parameters(parameters)
        return {
            group: self.optimizer.init({name: parameters[name] for name in names})
            for group, names in self.groups.items()
        }

    def update(self, gradients, state, parameters=None):
        import jax.numpy as jnp

        if parameters is None:
            raise ValidationError("Stagewise updates require the current parameters.")
        updates, next_state = {}, {}
        for group, names in self.groups.items():
            if group != "shared" and int(group.split("_")[1]) >= self.active_depth:
                updates.update({name: jnp.zeros_like(gradients[name]) for name in names})
                next_state[group] = state[group]
            else:
                local, next_state[group] = self.optimizer.update(
                    {name: gradients[name] for name in names},
                    state[group],
                    {name: parameters[name] for name in names},
                )
                updates.update(local)
        return updates, next_state
