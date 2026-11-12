"""Lagged, self-financing long-only accounting with explicit streaming state."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..compute.kernel import compute_operation, memory_budget
from ..errors import ValidationError
from ..ir.core import integer
from ..ir.parameters import real
from ._array import Ops, admission, rows


class TradingState(NamedTuple):
    positions: object
    cash: object
    pending: object
    ready: object


class TradingResult(NamedTuple):
    wealth: object
    returns: object
    costs: object
    turnover: object
    state: TradingState


@dataclass(frozen=True)
class SelfFinancingPlan:
    fees: object = 0.001
    impact: object = 0
    execution_lag: int = 1
    root_iterations: int = 48
    max_memory_mb: float = 256

    def __post_init__(self):
        for name in ("execution_lag", "root_iterations"):
            object.__setattr__(self, name, integer(getattr(self, name), name, 1))
        object.__setattr__(self, "max_memory_mb", memory_budget(self.max_memory_mb))
        values = []
        for name in ("fees", "impact"):
            x = np.asarray(getattr(self, name), dtype=float)
            if x.ndim > 1 or not x.size or not np.all(np.isfinite(x)) or np.any(x < 0):
                raise ValidationError(
                    "Trading rates must be finite nonnegative scalars or asset vectors."
                )
            values.append(float(np.max(x)))
            object.__setattr__(self, name, float(x) if x.ndim == 0 else tuple(float(v) for v in x))
        if values[0] + 2 * values[1] >= 1 or self.max_memory_mb <= 0:
            raise ValidationError(
                "Long-only financing requires max(fees)+2*max(impact)<1 "
                "and a positive memory budget."
            )

    @compute_operation
    def rollout(
        self,
        returns,
        decisions,
        *,
        cash_returns=0,
        initial_equity=1,
        initial_weights=None,
        state=None,
        mask=None,
    ):
        """Inputs [...,T,N] are arithmetic returns and close-time decisions.

        Decisions execute after execution_lag bars, before that bar's return.
        Targets are fractions of post-cost wealth, including residual cash.
        No trade is sent. Carry contains marked position values, cash, and the
        lag queue; passing it between chunks preserves both funding and AD.
        """
        ops = Ops(
            decisions,
            returns,
            cash_returns,
            initial_equity,
            initial_weights,
            *(state[:3] if isinstance(state, TradingState) else ()),
        )
        xp = ops.xp
        r, w = ops.real(returns, finite=False), ops.real(decisions, finite=False)
        if min(r.ndim, w.ndim) < 2 or r.shape[-2:] != w.shape[-2:] or min(r.shape[-2:]) < 1:
            raise ValidationError(
                "Returns and decisions need matching nonempty time and asset axes."
            )
        prefix = np.broadcast_shapes(r.shape[:-2], w.shape[:-2])
        t, n = r.shape[-2:]
        admission(rows(prefix) * (12 * n + 12 * t + 4 * self.execution_lag * n), self.max_memory_mb)
        r, w = (ops.broadcast(x, prefix + (t, n)) for x in (r, w))
        valid = xp.ones_like(w[..., 0]) > 0 if mask is None else ops.array(mask, boolean=True)
        if valid.shape != prefix + (t,):
            raise ValidationError("Trading mask must have shape [...,T].")
        r, w = xp.where(valid[..., None], r, 0), xp.where(valid[..., None], w, 0)
        r = ops.check(
            r, xp.isfinite(r) & (r >= -1), "Arithmetic returns must be finite and at least -1."
        )
        w = ops.check(w, xp.isfinite(w) & (w >= 0), "Trading targets must be finite and long-only.")
        eps = np.finfo(np.float64 if "64" in str(w.dtype) else np.float32).eps
        w = ops.check(
            w,
            (ops.sum(w) <= 1 + 8 * n * eps)[..., None],
            "Target weights exceed available post-cost capital.",
        )
        w = w / ops.maximum(ops.sum(w, keepdims=True), 1)
        cash_return = ops.broadcast(ops.real(cash_returns, finite=False), prefix + (t,))
        cash_return = xp.where(valid, cash_return, 0)
        cash_return = ops.check(
            cash_return,
            xp.isfinite(cash_return) & (cash_return >= -1),
            "Cash returns must be finite and at least -1.",
        )
        fees, impact = (ops.broadcast(ops.real(v), (n,)) for v in (self.fees, self.impact))
        if state is None:
            equity = real(initial_equity)
            if equity <= 0:
                raise ValidationError("initial_equity must be positive.")
            initial = ops.broadcast(
                ops.real(0 if initial_weights is None else initial_weights), prefix + (n,)
            )
            initial = ops.check(
                initial,
                (initial >= 0) & (ops.sum(initial) <= 1)[..., None],
                "Initial weights must be long-only and cash-financed.",
            )
            positions, cash = initial * equity, (1 - ops.sum(initial)) * equity
            pending = ops.broadcast(initial[..., None, :], prefix + (self.execution_lag, n))
            ready = xp.zeros_like(pending[..., 0]) > 0
            state = TradingState(positions, cash, pending, ready)
        else:
            if not isinstance(state, TradingState):
                raise ValidationError("Streaming trading state must be TradingState.")
            positions, cash, pending = (ops.real(v) for v in state[:3])
            ready = ops.array(state.ready, boolean=True)
            if (
                positions.shape != prefix + (n,)
                or cash.shape != prefix
                or pending.shape != prefix + (self.execution_lag, n)
                or ready.shape != prefix + (self.execution_lag,)
            ):
                raise ValidationError("Trading carry dimensions differ from this stream.")
            positions = ops.check(positions, positions >= 0, "Position values must be nonnegative.")
            cash = ops.check(cash, cash >= 0, "Cash balances must be nonnegative.")
            pending = ops.check(
                pending,
                (pending >= 0) & (ops.sum(pending) <= 1 + 8 * n * eps)[..., None],
                "Pending targets exceed financing constraints.",
            )
            state = TradingState(positions, cash, pending, ready)

        def step(carry, row):
            asset_return, decision, money_return, active = row
            positions, cash, pending, ready = carry
            before = ops.sum(positions) + cash
            denominator = xp.where(before > 0, before, 1)
            held = positions / denominator[..., None]
            target = pending[..., 0, :]
            target = xp.where(ready[..., 0, None], target, held)
            low, high = xp.zeros_like(before), xp.ones_like(before)
            for _ in range(self.root_iterations):
                midpoint = (low + high) * 0.5
                trades = midpoint[..., None] * target - held
                residual = midpoint + ops.sum(fees * xp.abs(trades) + impact * trades * trades) - 1
                low, high = (
                    xp.where(residual < 0, midpoint, low),
                    xp.where(residual < 0, high, midpoint),
                )
            signs = ops.stop(xp.sign(((low + high) * 0.5)[..., None] * target - held))
            a = ops.sum(impact * target * target)
            b = 1 + ops.sum(fees * signs * target - 2 * impact * target * held)
            c = ops.sum(impact * held * held - fees * signs * held) - 1
            ratio = -2 * c / (b + xp.sqrt(b * b - 4 * a * c))
            ratio = ops.clip(ratio, 0, 1)
            notional = before * ratio
            traded_fraction = ratio[..., None] * target - held
            cost = before * ops.sum(
                fees * xp.abs(traded_fraction) + impact * traded_fraction * traded_fraction
            )
            invested = notional[..., None] * target
            remaining_cash = notional * xp.maximum(1 - ops.sum(target), ops.array(0))
            invested = xp.where(ready[..., 0, None], invested, positions)
            remaining_cash = xp.where(ready[..., 0], remaining_cash, cash)
            cost = xp.where(ready[..., 0], cost, 0)
            traded_fraction = xp.where(ready[..., 0, None], traded_fraction, 0)
            ending = invested * (1 + asset_return)
            ending_cash = remaining_cash * (1 + money_return)
            wealth = ops.sum(ending) + ending_cash
            wealth = ops.check(
                wealth, xp.isfinite(wealth), "Trading wealth overflowed its numeric precision."
            )
            growth = xp.where(before > 0, (wealth - before) / denominator, 0)
            queued = ops.cat((pending[..., 1:, :], decision[..., None, :]), axis=-2)
            eligible = ops.cat((ready[..., 1:], xp.ones_like(active)[..., None]), axis=-1)
            next_state = TradingState(
                xp.where(active[..., None], ending, positions),
                xp.where(active, ending_cash, cash),
                xp.where(active[..., None, None], queued, pending),
                xp.where(active[..., None], eligible, ready),
            )
            output = (
                xp.where(active, wealth, before),
                xp.where(active, growth, 0),
                xp.where(active, cost, 0),
                xp.where(active, ops.sum(xp.abs(traded_fraction)), 0),
            )
            return next_state, output

        data = (
            ops.move(r, -2, 0),
            ops.move(w, -2, 0),
            ops.move(cash_return, -1, 0),
            ops.move(valid, -1, 0),
        )
        if ops.name == "jax":
            import jax

            state, output = jax.lax.scan(step, state, data)
        else:
            output = []
            for index in range(t):
                state, row = step(state, tuple(v[index] for v in data))
                output.append(row)
            output = tuple(ops.stack([row[i] for row in output], axis=0) for i in range(4))
        return TradingResult(*(ops.move(v, 0, -1) for v in output), state)
