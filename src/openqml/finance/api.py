"""Small financial entry points; policies and native results stay explicit."""

from ..compute.kernel import compute_operation
from ..errors import ResourceError, ValidationError
from ..ir.parameters import real
from ._array import Ops
from ._results import PayoffEstimate


def _exact_coherent(payoff, oracle, offset, scale, discount, backend, memory):
    """Use exact simulation only after backend capability/resource admission."""
    from dataclasses import replace

    import numpy as np

    from ..runtime.backends import ExecutionOptions, get_backend
    from ..runtime.scheduler import PriorityBackend, resolve_backend
    from ..sim.dense import expectation, statevector

    source = payoff.preparation if payoff is not None else oracle.preparation
    options = ExecutionOptions(max_memory_mb=memory)
    if isinstance(backend, str) and backend in {"torch", "jax"}:
        device = backend
    else:
        device = get_backend(backend)
        if isinstance(device, PriorityBackend):
            device = replace(device, require_exact=True)
        if payoff is None and source.num_qubits <= memory * 2**20:
            from ..ir.observables import PauliString

            word = "I" * oracle.good_qubit + "Z" + "I" * (source.num_qubits - oracle.good_qubit - 1)
            observable = PauliString(word)
            try:
                chosen = resolve_backend(
                    device, "expectations", [source], (observable,), options=options
                )
                if (
                    chosen.capabilities.native_expectations
                    and chosen.capabilities.exact_expectations
                ):
                    z = float(expectation(source, observable, backend=chosen, max_memory_mb=memory))
                    value = discount * (offset + scale * (1 - z) / 2)
                    return PayoffEstimate(value, ((value, value),), 1.0, None)
            except ResourceError:
                pass
        try:
            device = resolve_backend(device, "statevector", [source], options=options)
        except ResourceError:
            return None
        if (
            not callable(getattr(device, "run", None))
            or not device.capabilities.deterministic_expectations
        ):
            return None
    try:
        state = statevector(source, backend=device, max_memory_mb=memory)
    except ResourceError:
        return None
    state = state.detach().cpu().numpy() if hasattr(state, "detach") else np.asarray(state)
    p = abs(state) ** 2
    value = (
        payoff.classical_value(p)
        if payoff is not None
        else discount * (offset + scale * p.reshape((1 << oracle.good_qubit, 2, -1))[:, 1].sum())
    )
    return PayoffEstimate(value, ((value, value),), 1.0, None)


@compute_operation
def allocate(
    returns=None,
    *,
    risk=None,
    mean=None,
    probabilities=None,
    previous=None,
    caps=1,
    fees=0,
    budget=1,
    shrinkage=0.1,
    variance_floor=1e-8,
    risk_aversion=None,
    turnover=None,
    ridge=None,
    tolerance=None,
    max_memory_mb=None,
    warm_start=None,
    plan=None,
):
    """Allocate directly from [...,samples,assets] arithmetic return observations.

    Arrays and gradients remain native. Supply a FactorRisk instead of returns
    for an existing risk model. Policy options belong either here or in plan.
    """
    from .allocation import AllocationPlan
    from .risk import factor_risk

    policy = {
        k: v
        for k, v in dict(
            risk_aversion=risk_aversion,
            turnover=turnover,
            ridge=ridge,
            tolerance=tolerance,
            max_memory_mb=max_memory_mb,
        ).items()
        if v is not None
    }
    if plan is not None and (policy or not isinstance(plan, AllocationPlan)):
        raise ValidationError("Supply an AllocationPlan or inline policy options.")
    plan = AllocationPlan(**policy) if plan is None else plan
    if (returns is None) == (risk is None):
        raise ValidationError("Supply return observations or a FactorRisk.")
    if risk is None:
        risk = factor_risk(
            returns,
            probabilities,
            shrinkage=shrinkage,
            variance_floor=variance_floor,
            max_memory_mb=plan.max_memory_mb,
        )
    elif probabilities is not None:
        raise ValidationError("FactorRisk already owns its probability masses.")
    return plan.solve(
        risk.mean if mean is None else mean,
        risk,
        previous=previous,
        caps=caps,
        fees=fees,
        budget=budget,
        warm_start=warm_start,
    )


@compute_operation
def price(
    payoffs=None,
    *,
    probabilities=None,
    preparation=None,
    oracle=None,
    bounds=None,
    discount=1,
    method="auto",
    error=0.01,
    confidence=0.95,
    max_queries=100_000,
    max_power=32,
    visibility=1,
    backend="auto",
    seed=None,
    plan=None,
    strict=True,
    control_values=None,
    control_expectation=None,
    control_coefficient=1,
    max_memory_mb=128,
):
    """Discounted bounded expectation, with exact summation preferred when possible.

    Quantum mode requires an actual coherent preparation or AmplitudeOracle.
    Bounds describe the payoff encoded in the marked amplitude. The caller
    supplies the pricing measure; a forecast distribution is not risk-neutral.
    """
    discount, error, confidence = real(discount), real(error), real(confidence)
    if discount <= 0 or error <= 0 or not 0 < confidence < 1 or type(strict) is not bool:
        raise ValidationError("Use positive discount/error and confidence in (0,1).")
    if method not in {"auto", "classical", "quantum"}:
        raise ValidationError("Pricing method must be auto, classical or quantum.")
    if probabilities is not None and method != "quantum":
        if probabilities is None or payoffs is None:
            raise ValidationError("Classical pricing needs payoffs and scenario probabilities.")
        ops = Ops(probabilities, payoffs)
        values = ops.real(payoffs)
        if values.ndim < 1 or not values.shape[-1]:
            raise ValidationError("Payoffs need a nonempty scenario axis.")
        p = ops.probabilities(probabilities, values.shape[-1])
        value = discount * ops.sum(p * values)
        return PayoffEstimate(value, ((value, value),), 1.0, None)
    from ..sim.adaptive_amplitude import AdaptiveAmplitudeEstimationPlan
    from ..sim.amplitude import AmplitudeEstimationPlan, AmplitudeOracle

    if oracle is not None:
        if (
            not isinstance(oracle, AmplitudeOracle)
            or preparation is not None
            or payoffs is not None
        ):
            raise ValidationError("Use an AmplitudeOracle, or a preparation and payoff table.")
        if (
            bounds is None
            or len(bounds) != 2
            or control_values is not None
            or control_expectation is not None
        ):
            raise ValidationError("A direct oracle needs two payoff bounds and no table controls.")
        lower, upper = (real(v) for v in bounds)
        if upper < lower:
            raise ValidationError("Payoff bounds must be ordered.")
        scale, offset = upper - lower, lower
        payoff = None
    else:
        if preparation is None or payoffs is None:
            raise ValidationError("Quantum pricing needs coherent preparation and bounded payoffs.")
        from .pricing import PayoffOraclePlan

        payoff = PayoffOraclePlan(
            preparation,
            payoffs,
            bounds=bounds,
            discount=discount,
            control_values=control_values,
            control_expectation=control_expectation,
            control_coefficient=control_coefficient,
            max_memory_mb=max_memory_mb,
        )
        scale, offset = payoff.scale, payoff.offset
    if scale == 0:
        value = discount * offset
        return PayoffEstimate(value, ((value, value),), 1.0, None)
    if method != "quantum":
        exact = _exact_coherent(payoff, oracle, offset, scale, discount, backend, max_memory_mb)
        if exact is not None:
            return exact
        if method == "classical":
            raise ResourceError("An exact classical pricing method did not fit the backend/budget.")
    if plan is None:
        plan = AdaptiveAmplitudeEstimationPlan(
            tolerance=min(1, error / (discount * scale)),
            confidence=confidence,
            max_power=max_power,
            max_queries=max_queries,
            visibility=visibility,
            max_memory_mb=max_memory_mb,
        )
    elif not isinstance(plan, (AmplitudeEstimationPlan, AdaptiveAmplitudeEstimationPlan)):
        raise ValidationError("Supply a fixed or adaptive amplitude plan.")
    if strict and plan.confidence < confidence:
        raise ValidationError("The estimation plan's confidence is below the requested confidence.")
    if payoff is not None:
        estimate = payoff.estimate(plan, backend=backend, seed=seed)
    else:
        result = plan.run(oracle, backend=backend, seed=seed)
        intervals = tuple(
            (discount * (offset + scale * a), discount * (offset + scale * b))
            for a, b in result.intervals
        )
        estimate = PayoffEstimate(
            discount * (offset + scale * result.probability), intervals, result.confidence, result
        )
    lo, hi = estimate.interval
    achieved = max(abs(estimate.value - lo), abs(hi - estimate.value))
    if strict and (achieved > error or estimate.confidence < confidence):
        raise ResourceError(
            f"Pricing budget did not reach error={error}; achieved bound={achieved}. "
            "Increase the budget, improve calibrated contrast, or inspect with strict=False."
        )
    return estimate
