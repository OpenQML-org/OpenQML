"""Whole-network resource choices with shared cost and latency constraints."""

from dataclasses import dataclass, replace

from ..errors import ResourceError, ValidationError
from ..ir.core import integer
from ..runtime.backends import ExecutionOptions, get_backend
from ..runtime.costs import ExecutionCost, TrainingWorkload, check_cost, select_cost
from ..runtime.scheduler import CostAwareBackend


@dataclass(frozen=True)
class NetworkExecutionPlan:
    fingerprint: str
    active_depth: int
    stages: tuple
    forecast: object
    searched_states: int
    stage_indices: tuple = ()

    def to_dict(self):
        value = {
            "architecture": self.fingerprint,
            "active_depth": self.active_depth,
            "forecast": None if self.forecast is None else self.forecast.to_dict(),
            "stages": [s.to_dict() for s in self.stages],
            "searched_states": self.searched_states,
            "scope": "quantum_stage_forecasts_plus_declared_classical_work",
        }
        if self.stage_indices and self.stage_indices != tuple(range(self.active_depth)):
            value["stage_indices"] = list(self.stage_indices)
        return value


def reserve_native_cost(plan, scheduler, *, native_devices=("CPU", "GPU")):
    """Reserve one integrated update's native/classical forecast before execution.

    External calls reserve their own refreshed quotes at numeric submission time.
    Keeping them out of this reservation avoids charging the same work twice.
    Failed updates retain reservations until billing reconciliation.
    """
    if scheduler.ledger is None or plan is None or plan.forecast is None:
        return
    external = sum(
        _selected_cost(stage).upper_cost
        for stage in plan.stages
        if stage.device not in native_devices
    )
    scheduler.ledger.reserve(max(0, plan.forecast.upper_cost - external))


def _selected_cost(plan):
    value = next(c.estimate for c in plan.candidates if c.backend == plan.backend)
    return None if value is None else ExecutionCost(**value)


def schedule_network(
    architecture,
    *,
    scheduler=None,
    workload=None,
    active_depth=None,
    max_states=64,
    native_devices=("CPU", "GPU"),
    metadata=None,
    microbatch_samples=None,
    native_candidates=(),
):
    scheduler = get_backend("auto") if scheduler is None else scheduler
    if not isinstance(scheduler, CostAwareBackend):
        raise ValidationError("Network scheduling requires CostAwareBackend.")
    workload = scheduler.workload if workload is None else workload
    if not isinstance(workload, TrainingWorkload):
        raise ValidationError("workload must be TrainingWorkload.")
    integer(max_states, "max_states", 1)
    depth = architecture.active(active_depth)
    options = ExecutionOptions(max_memory_mb=architecture.max_memory_mb)
    alternatives, baseline = [], []
    indices = architecture.execution_indices(depth)
    for index in indices:
        cell = architecture.cells[index]
        spec = cell.to_spec()
        shift = TrainingWorkload.parameter_shift(spec.circuit, spec.observables)
        batches = None
        if metadata is not None:
            architecture.operator_specs[index].implementation.validate_metadata(
                metadata, architecture.operator_specs[index].config
            )
            batches = architecture.execution_batches(
                metadata, microbatch_samples or metadata["samples"], index
            )
        elif architecture.operator_specs[index].kind != "pointwise":
            # Constructor bootstrap is only a routing hint. Trainers provide
            # actual input metadata before executing any optimizer update.
            rows = workload.batch_size or 1
            batches = (rows,) * len(
                architecture.operator_specs[index].implementation.roles(
                    architecture.operator_specs[index].config
                )
            )
        work = replace(
            workload,
            batch_size=workload.batch_size if batches is None else sum(batches) or None,
            execution_batches=batches,
            gradient_evaluations=shift.gradient_evaluations,
            native_gradient_evaluations=(1 + int(architecture.rematerialize))
            if workload.native_gradient_evaluations is None
            else workload.native_gradient_evaluations,
            native_devices=tuple(native_devices),
            classical_seconds_per_step=0,
            classical_cost_per_step=0,
        )
        choices = scheduler.alternatives(
            spec.circuit,
            observables=spec.observables,
            options=options,
            workload=work,
            native_candidates=native_candidates,
            allowed_devices=native_devices
            if architecture.operator_specs[index].implementation.native_state
            else None,
        )
        baseline.append(choices[0])
        alternatives.append(tuple((p, _selected_cost(p)) for p in choices))
    if any(cost is None for choices in alternatives for _, cost in choices):
        return NetworkExecutionPlan(
            architecture.fingerprint, depth, tuple(baseline), None, 0, indices
        )
    initial = ExecutionCost(
        workload.steps * workload.classical_seconds_per_step,
        workload.steps * workload.classical_seconds_per_step,
        workload.steps * workload.classical_cost_per_step,
        workload.steps * workload.classical_cost_per_step,
        0,
        0,
        0,
        0,
        0,
        "whole_network_quote",
    )
    states, searched = [((), initial)], 0
    for choices in alternatives:
        expanded = []
        for path, cost in states:
            for plan, other in choices:
                searched += 1
                total = replace(
                    cost,
                    seconds=cost.seconds + other.seconds,
                    upper_seconds=cost.upper_seconds + other.upper_seconds,
                    cost=cost.cost + other.cost,
                    upper_cost=cost.upper_cost + other.upper_cost,
                    estimated_rmse=max(cost.estimated_rmse, other.estimated_rmse),
                    physical_shots=cost.physical_shots + other.physical_shots,
                    circuit_evaluations=cost.circuit_evaluations + other.circuit_evaluations,
                    provider_jobs=cost.provider_jobs + other.provider_jobs,
                    observations=cost.observations + other.observations,
                    pricing_known=cost.pricing_known and other.pricing_known,
                    queue_known=cost.queue_known and other.queue_known,
                    noise_known=cost.noise_known and other.noise_known,
                )
                try:
                    check_cost(total, scheduler.policy)
                    if (
                        scheduler.ledger is not None
                        and total.upper_cost + scheduler.ledger.reserved > scheduler.ledger.limit
                    ):
                        raise ResourceError("Network exceeds the remaining spending budget.")
                except ResourceError:
                    continue
                expanded.append((path + (plan,), total))
        if not expanded:
            raise ResourceError(
                "No whole-network schedule meets the shared cost/time/error constraints."
            )
        _, frontier = select_cost([c for _, c in expanded], scheduler.policy)
        states = [expanded[i] for i in frontier]
        # Bound multi-choice Pareto growth; choose diverse representatives in
        # time/cost order before preference selection at the final stage.
        if len(states) > max_states:
            states.sort(key=lambda row: (row[1].upper_seconds, row[1].upper_cost))
            indices = [
                round(i * (len(states) - 1) / max(1, max_states - 1)) for i in range(max_states)
            ]
            states = [states[i] for i in indices]
    selected, _ = select_cost([c for _, c in states], scheduler.policy)
    path, cost = states[selected]
    check_cost(cost, scheduler.policy)
    return NetworkExecutionPlan(architecture.fingerprint, depth, path, cost, searched, indices)
