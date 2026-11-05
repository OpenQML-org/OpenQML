"""Conditional GBM barrier payoffs without a time-stepped path register."""

from dataclasses import dataclass

from ..errors import ValidationError
from ._array import Ops


@dataclass(frozen=True)
class BarrierOption:
    """Single continuous barrier, no rebate, under constant-volatility GBM.

    Evaluate on terminal prices to integrate out the barrier-crossing event.
    Parameters broadcast over batch axes; a non-scalar terminal array has its
    scenario axis last. Outputs are undiscounted conditional expected payoffs.
    The caller supplies a model-consistent terminal law and discount factor.
    """

    spot: object
    strike: object
    barrier: object
    volatility: object
    maturity: object
    direction: str = "up"
    knock: str = "out"
    kind: str = "call"

    def __post_init__(self):
        if self.direction not in {"up", "down"} or self.knock not in {"in", "out"}:
            raise ValidationError("Use direction up/down and knock in/out.")
        if self.kind not in {"call", "put"}:
            raise ValidationError("Option kind must be call or put.")

    def _condition(self, terminal):
        ops = Ops(terminal, self.spot, self.strike, self.barrier, self.volatility, self.maturity)
        terminal = ops.real(terminal)
        terminal = ops.check(terminal, terminal > 0, "Terminal prices must be positive.")
        parameters = []
        for name in ("spot", "strike", "barrier", "volatility", "maturity"):
            value = ops.real(getattr(self, name))
            value = ops.check(
                value,
                value >= 0 if name == "strike" else value > 0,
                "Use positive spot, barrier, volatility and maturity; strike may be zero.",
            )
            parameters.append(value[..., None] if terminal.ndim else value)
        spot, strike, barrier, volatility, maturity = parameters
        variance = volatility * volatility * maturity
        variance = ops.check(
            variance,
            ops.xp.isfinite(variance) & (variance > 0),
            "GBM log-variance must be finite and positive.",
        )
        if self.direction == "up":
            start = ops.xp.log(barrier) - ops.xp.log(spot)
            end = ops.xp.log(barrier) - ops.xp.log(terminal)
        else:
            start = ops.xp.log(spot) - ops.xp.log(barrier)
            end = ops.xp.log(terminal) - ops.xp.log(barrier)
        active = (start > 0) & (end > 0)
        exponent = -2 * ops.maximum(start, 0) * ops.maximum(end, 0) / variance
        return ops, terminal, strike, active, exponent

    def survival_probability(self, terminal):
        """P(no continuous crossing | terminal price), including time zero."""
        ops, _, _, active, exponent = self._condition(terminal)
        return ops.xp.where(active, -ops.xp.expm1(exponent), 0)

    def crossing_probability(self, terminal):
        """P(a continuous crossing | terminal price), including time zero."""
        ops, _, _, active, exponent = self._condition(terminal)
        return ops.xp.where(active, ops.xp.exp(exponent), 1)

    def __call__(self, terminal):
        ops, terminal, strike, active, exponent = self._condition(terminal)
        probability = (
            ops.xp.where(active, -ops.xp.expm1(exponent), 0)
            if self.knock == "out"
            else ops.xp.where(active, ops.xp.exp(exponent), 1)
        )
        difference = terminal - strike if self.kind == "call" else strike - terminal
        return ops.maximum(difference, 0) * probability
