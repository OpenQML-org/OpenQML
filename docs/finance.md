# Quantitative finance

`openqml.finance` connects the existing quantum models and tensor executors to
financial objectives. It provides native NumPy, PyTorch and JAX calculations;
forecast and scenario architectures also build ordinary Flax models. No market
data, broker connection or extra core dependency is installed.

| Task | Interface | Contract |
| --- | --- | --- |
| Conditional return quantiles | `MarketForecastPlan(F, N)` | Ordered quantiles, pinball loss, recurrent/state-attention/tensor models |
| Joint return or path distributions | `ScenarioPlan(atoms)` | Conditional Born probabilities over fixed user-supplied joint atoms |
| Factor risk | `factor_risk(returns, probabilities)` | Weighted centered factors plus shrinkage diagonal |
| Differentiable allocation | `AllocationPlan.solve(mean, risk, ...)` | Long-only capped allocation with funding and turnover costs |
| Tail risk | `expected_shortfall`, `wasserstein_cvar` | Fractional weighted CVaR and a Wasserstein robustness margin |
| Funded execution | `SelfFinancingPlan.rollout(...)` | Delayed decisions, cash, proportional fees and quadratic impact |
| Chronological validation | `PurgedWalkForward.split(times, label_end)` | Past-only training with overlapping labels removed |
| Discrete asset selection | `PortfolioSelection(...).problem` | Equal-weight fixed-cardinality financial QUBO |

## Train decisions through the scenario model

Install `openqml[torch]`. Supply training-only return atoms `[2**k,N]`, causal
features `[B,T,F]`, and next-bar arithmetic-return labels `[B,T,N]`. Each atom is
an entire cross-asset scenario, so a joint Born distribution retains its
cross-asset dependence. The following factory uses those supplied tensors:

```python
import torch
from openqml import finance as f


def financial_trainer(atoms, input_features, device="cpu"):
    scenarios = f.ScenarioPlan(atoms)
    architecture = scenarios.architecture(input_features, depth=2, hidden_features=24)
    model = architecture.build("torch", device=device)
    allocator = f.AllocationPlan(risk_aversion=2, turnover=0.05)
    trainer = model.trainer(lr=1e-3, quantum_lr=3e-4, microbatch_size=8)

    def loss(network, batch):
        features, next_returns, previous_weights = batch
        probabilities = network(features)
        risk = scenarios.risk(probabilities)
        result = allocator.solve(risk.mean, risk, previous=previous_weights, caps=1, fees=0.001)
        score = scenarios.energy_score(probabilities, next_returns)
        # The observed return determines decision quality during training.
        decision = -(result.weights * next_returns).sum(dim=-1)
        return (score + 0.1 * decision).mean(dim=-1)

    return model, trainer, loss
```

Call `trainer.step((features, next_returns, previous_weights), loss)` with tensors
on the model device. Previous weights must broadcast to `[B,T,N]`; supply the
holdings relevant to each decision. The loss returns one value per example for
gradient accumulation. It trains a one-period decision objective; it does not
simulate funded trades. Evaluate or train a sequence objective with the
self-financing ledger below when execution timing and cumulative wealth matter.

`ScenarioPlan.energy_score` is the multivariate energy score
`E||X-y|| - E||X-X'||/2`. Its finite-support probability gradient continues through
factor risk and the allocation layer into the Born model. The factor solver
retains the probability masses separately, including zero-probability atoms.
It does not differentiate `sqrt(p)` at zero.

For JAX, use `architecture.build("jax")`, `model.init(key)` and
`model(parameters, features)`; use native JAX reductions in the same loss. Flax
uses Linen `init`/`apply`. Existing optimizer groups, microbatching, distributed
training and checkpoints apply. Persist atom values/order, preprocessing and
financial plan settings alongside model checkpoints; the plans do not fit data
or choose train/test boundaries for you.

`MarketForecastPlan(F,N,model="recurrent")` offers an alternative quantile
forecaster. `plan.build(...)` returns a standard architecture model;
`plan.decode(raw)` returns ordered quantiles and a median, and
`plan.loss(raw, targets, mask=...)` computes pinball loss. A median is not a mean
return estimate. Set `model="state-attention"` for causal state attention or
`model="tensor"` for native tensor contraction. `projection_policy="tensor-train"`
factors classical projections. Quantum registers and tensor bonds remain
resource-limited; these options do not remove general simulation complexity.

## Risk and allocation

`factor_risk` constructs `Cov = F.T @ diag(mass) @ F + diag(diagonal)` from a
weighted population covariance, with caller-selected diagonal shrinkage and a
variance floor. Supply a `FactorRisk` directly for an external factor model.
`covariance_matvec` and `portfolio_variance` avoid dense covariance construction;
`dense_covariance` is an explicit, memory-admitted conversion.

`AllocationPlan` minimizes

```text
0.5 * risk_aversion * w'Cov*w - mean'w + 0.5 * ridge * ||w||²
  + 0.5 * turnover * ||w-previous||² + fees'|w-previous|
subject to 0 <= w <= caps, sum(w) = budget.
```

The positive ridge gives a unique solution. A sorted exact cost-aware proximal
projection identifies active bounds and no-trade facets. The final constrained
solve uses preconditioned conjugate gradients with factor matrix products,
without a dense asset KKT factorization or a stored optimizer-iteration tape.
PyTorch implicit adjoints/JVPs and JAX `custom_linear_solve` differentiate the
fixed active face; derivatives at changing faces are piecewise, not globally
smooth. Small-rank spectral work still costs `min(rank,N)**2` workspace.

Check `AllocationResult.residual` and `converged`. Eager nonconvergence raises
`ResourceError`; invalid/nonconverged traced results become NaN so the existing
training finite checks can skip them. `resource_profile(N,rank,batch_size=...)`
admits primal workspace and arithmetic budgets, excluding framework reserves
and autodiff tapes. Computation stays on the selected native tensor device.
This is a single-stage allocation layer with explicit previous holdings, not a
full multi-horizon dynamic portfolio optimizer.

`expected_shortfall(losses, confidence=0.95, probabilities=p)` uses the worst
5% mass and includes fractional boundary atoms. `wasserstein_cvar(w, returns,
confidence=0.95, radius=r)` adds `r * ||w||2 / 0.05`, a conservative W1/L2 robust
CVaR bound, sharp for unrestricted return support. A bounded support can tighten
the bound. The caller must calibrate the radius; this function does not supply
out-of-sample coverage, shift estimation or block-bootstrap validation.

## Execution timing and self-financing accounts

Pass arithmetic bar returns and close-time decisions, both `[...,T,N]`, to
`SelfFinancingPlan(fees=..., impact=..., execution_lag=1).rollout(...)`.
Decision `t` executes before the return at `t+execution_lag`. Initial positions
are simply held until a decision becomes eligible. Labels for a one-bar forecast
are returns at `t+1`; do not pass those shifted labels as bar returns to the
ledger. Causal feature construction remains the caller's responsibility.

For pre-trade positions `h`, wealth `V` and target weights `w`, post-cost wealth
`u` solves `u + sum(fees*abs(u*w-h) + impact*(u*w-h)**2/V) = V`. Targets are
fractions of **post-cost** wealth. Remaining capital is cash; asset and cash
returns then mark positions to market. The plan requires long-only targets
whose sum is at most one and `max(fees)+2*max(impact)<1`. It does not model
leverage, borrowing, shorts, taxes, order-book fills or exchange latency.

The result exposes wealth, period returns, costs, turnover and an explicit
`TradingState`. Pass that state to the next chunk to preserve funding, the lag
queue and gradients. A false padding mask holds state without advancing time;
bankruptcy is absorbing. PyTorch/JAX retain piecewise derivatives through the
analytic financing root. This ledger sends no orders.

`PurgedWalkForward(train_size,test_size,gap=...)` accepts increasing numeric or
NumPy datetime decision timestamps and label end timestamps. Training observations
are strictly earlier than test decisions; labels touching the first test time
are purged. `gap` adds a pre-test gap. Fit scalers, atoms, factors and risk
calibration independently within each training fold.

## Cardinality-preserving quantum selection

`PortfolioSelection(mean,covariance,cardinality=k,previous=...,fees=...)` maps
equal-weight holdings `w=x/k` to the same risk, ridge, turnover and absolute-fee
objective. Its fixed estimates must be explicitly detached from training.
Use `problem.qaoa(depth=...,counterdiabatic="full")` to build a portable
`LayerSpec`, initialize its weights, train its expectation or sample it through
the existing backend scheduler. Costs stay in financial units; cost phases use
the exposed `energy_scale`. `counterdiabatic="local"` reduces circuit cost;
`"none"` selects ordinary XY-mixer QAOA.

The XY mixer and paired first-commutator counterdiabatic generators preserve
Hamming weight under ideal unitaries. Full mode includes Z-conditioned
three-body pairs. Initialization uses a feasible basis state plus XY spreading;
it is not uniform Dicke preparation. The implementation adapts the first
commutator ansatz, not the paper's complete variational gauge-potential solver.
Noisy sample decoding reports invalid shots and the feasible-conditional mean;
it never treats discarded shots as successful constraint preservation.

For hardware-sized subproblems, `problem.condition(indices,fixed_selection)`
retains cross-block linear terms, the global offset and residual cardinality.
`merge` checks the global constraint, `refine` performs feasible improving swaps,
and `exact(max_states=...)` is a bounded classical alternative. Partition and
coordinate selection remain caller-controlled. These are building blocks for
hybrid large-universe optimization, not a polynomial solver for arbitrary QUBOs.

Finite scenario atoms restrict the model's support. Path atoms `[S,H,N]` encode
joint paths; `returns()` compounds each asset without intra-path rebalancing.
`expected_payoff` computes a discounted expectation under the supplied weights;
a risk-neutral measure and no-arbitrage conditions are not inferred from data.
No financial return improvement, quantum advantage or real multi-GPU/QPU
throughput is established by the functional tests.

## Method sources

- [Integrated Prediction and Multi-period Portfolio Optimization](https://arxiv.org/abs/2512.11273): decision-focused training and fixed-point differentiation. OpenQML implements its own single-stage active-face factor solver, not the full multi-period algorithm.
- [Constrained Counterdiabatic Quantum Approximate Optimization Algorithm for Portfolio Optimization](https://arxiv.org/abs/2605.06858): number-preserving commutator generators; the adaptation above omits full gauge-potential optimization.
- [The Lie Algebra of XY-mixer Topologies and Warm Starting QAOA for Constrained Optimization](https://doi.org/10.1038/s41534-026-01192-4), npj Quantum Information 12, 61 (2026): topology, expressivity and trainability tradeoffs of constrained mixers.
- [Large-scale portfolio optimization on a trapped-ion quantum computer](https://arxiv.org/abs/2602.23976): hybrid decomposition motivation; no hardware results are reproduced here.
- [Shift-Aware Gaussian-Supremum Validation for Wasserstein-DRO CVaR Portfolios](https://arxiv.org/abs/2512.16748): robust tail-risk formulation. Radius calibration and statistical validation are separate from the implemented margin.
- [Quantum generative modeling for financial time series with temporal correlations](https://doi.org/10.1088/2632-2153/ae39a2), Machine Learning: Science and Technology 7, 015027 (2026): financial quantum generative modeling; the finite-atom Born model here differs from its QGAN and MPS experiments.
