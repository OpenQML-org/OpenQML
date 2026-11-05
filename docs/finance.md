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
| Decision risk budgets | `risk_budget_projection(weights, scenarios, ...)` | Maximal funded cash-ray scaling under weighted robust CVaR |
| Funded execution | `SelfFinancingPlan.rollout(...)` | Delayed decisions, cash, proportional fees and quadratic impact |
| Chronological validation | `PurgedWalkForward.split(times, label_end)` | Past-only training with overlapping labels removed |
| Rolling model selection | `walk_forward(factory, inputs, targets, ...)` | Fresh hybrid models, nested validation and aligned out-of-sample predictions |
| Discrete asset selection | `PortfolioSelection(...).problem` | Equal-weight fixed-cardinality financial QUBO |
| Coherent financial expectations | `PayoffOraclePlan` and `AmplitudeEstimationPlan` | Signed bounded payoffs, control variates and finite-shot intervals |

## Rolling hybrid model selection

`walk_forward` connects the purged splitter to the high-level hybrid model.
Each fold constructs a fresh model, fits its historical rows, predicts the
disjoint test interval and records the held-out task loss. With
`validation_size`, it reserves a chronological tail of the training fold and
purges training labels that reach the first validation decision time.
The outer purge also keeps every training/validation label strictly before
the test decision interval.

```python
import openqml as oq


def rolling_predictions(features, next_returns, decision_times, label_end):
    return oq.finance.walk_forward(
        lambda: oq.Model(task="forecast", hidden_features=16, depth=2),
        features,
        next_returns,
        times=decision_times,
        label_end=label_end,
        splitter=oq.finance.PurgedWalkForward(train_size=252, test_size=21),
        validation_size=42,
        fit_options={"epochs": 50, "patience": 5, "min_delta": 1e-5},
        predict_options={"batch_size": 128},
    )
```

Supply causal features `[observations,lookback,features]` and matching
next-return labels `[observations,lookback,assets]`. `label_end` is the last
timestamp used by each observation's complete target window. For scalar or
vector regression, a factory returning `Model(task="regression",...)` also
accepts ordinary `[observations,features]` inputs. Set `framework` on the model
factory to select Torch, JAX or Flax.

`WalkForwardResult.predictions` retains the original observation axis.
`mask=True` identifies test predictions and `fold_id` identifies their refit;
uncovered rows contain NaN and fold ID -1. Each `WalkForwardMetrics` entry
records training/validation/test counts, purged labels, test boundaries, test
loss and the selected epoch. Test losses are recorded after fitting.
Use `step_size >= test_size` for disjoint test intervals. Larger steps leave
explicit gaps in the result mask.

Model fitting receives only that fold's admitted training rows and its optional
validation tail. Fit feature normalization, feature selection and scenario
atoms within the training fold as part of the application pipeline. Map
forecasts to allocation decisions separately, then pass decisions and their
original bar returns to `SelfFinancingPlan` for lagged, funded accounting.
`max_memory_mb` admits the NumPy result and indexed-data workspace; configure
native model memory through the chosen architecture and execution backend.

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
solve uses a residual-refined Woodbury solve when the factor rank is at most
64 and at most one quarter of the asset count. It solves a factor-sized system
with factors scaled by the square root of their absolute mass before forming
the Gram matrix. This balancing prevents inactive large factors from overflowing
during the numerical solve. Implicit derivatives use the original operator with
explicit masses, retaining finite derivatives at zero-mass atoms. Other factor
shapes use preconditioned conjugate gradients
with factor matrix products. Both paths avoid an asset-sized KKT factorization
and a stored optimizer-iteration tape.
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

`oq.finance.allocate(returns,...)` is the array-based entry point. Supply the last
solution as `warm_start` to propose an active face for a nearby problem. The
current equations are solved and feasibility/KKT residuals certified before the
iterative primal search can be skipped. A failed certificate falls back to that
search. The seed is detached from differentiation; `previous` still represents
actual economic holdings. JAX uses a compiled conditional, while eager NumPy and
PyTorch branch directly. Compiled PyTorch keeps the ordinary solver path.

`expected_shortfall(losses, confidence=0.95, probabilities=p)` uses the worst
5% mass and includes fractional boundary atoms. `wasserstein_cvar(w, returns,
confidence=0.95, radius=r)` adds `r * ||w||2 / 0.05`, a conservative W1/L2 robust
CVaR bound, sharp for unrestricted return support. A bounded support can tighten
the bound. The caller must calibrate the radius; this function does not supply
out-of-sample coverage, shift estimation or block-bootstrap validation.

When probabilities are omitted, CVaR uses exact partial selection instead of a
full scenario sort: NumPy partition, PyTorch top-k or JAX top-k. Fractional
boundary mass is retained, including when the tail contains less than one atom.
The weighted case sorts losses and accumulates the admitted upper-tail mass
directly, avoiding subtraction against an extreme lower-tail threshold.
At zero confidence it returns the weighted mean. Probability normalization uses
an exact power-of-two scale before summation so large finite weights remain
representable. Fractional unweighted boundaries exclude one selected boundary
atom before summing the interior. Native
gradients remain piecewise; ties can select different valid subgradients across
backends. Selection uses the native array backend's partition or top-k kernel.

## Differentiable decision risk budgets

`risk_budget_projection` takes a long-only proposal whose asset weights sum to
at most one, a joint scenario table, optional Born probabilities, a CVaR budget
and a nonnegative Wasserstein radius. It returns `RiskBudgetResult` with
`weights`, `cash`, `scale`, `cvar`, `robust_cvar` and `budget`.

```python
import openqml as oq

scenarios = [[-0.12, -0.03], [0.04, -0.08], [0.10, 0.02], [-0.05, 0.01]]
decision = oq.finance.risk_budget_projection(
    [0.4, 0.5], scenarios, confidence=0.95, budget=0.02, radius=0.0005
)
print(decision.weights, decision.cash, decision.robust_cvar)
```

For proposal `w`, scenario returns `R` and deterministic cash return `r_c`,
define `D = CVaR(r_c*sum(w) - R@w) + radius*||w||2/(1-confidence)`.
The implementation selects `scale=1` when `D <= budget+r_c`, otherwise
`scale=(budget+r_c)/D`. Positive homogeneity and cash translation make this the
largest feasible scale on the proposal's ray toward cash. The budget must admit
the all-cash position. Asset caps and funding remain satisfied when a feasible
proposal is scaled; pass the resulting holdings to the self-financing ledger
for transaction costs and execution timing.

Scenarios, probabilities, proposals, budgets, cash returns and contextual radii
can be native batched tensors. Gradients reach a scenario model and a learned
radius head through the selected CVaR ordering and projection branch; ties and
active-budget boundaries have piecewise derivatives. Fit/calibrate contextual
radii within the training fold. The reported risk applies to the supplied
scenario horizon and robust margin. Eager invalid inputs raise a validation
error; traced invalid radii, probabilities, scenarios or cash/budget bounds
propagate NaNs into the projected weights. Price spreads, fees, impact and realized
future returns are accounted for by the execution model.

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
`expected_payoff` computes a discounted expectation under the supplied weights.
Financial pricing uses a supplied risk-neutral measure. Evaluate forecasting
and allocation choices with chronological held-out predictions, funded trading
costs and the target execution environment's resource measurements.

## Conditional barrier payoffs

`BarrierOption(spot,strike,barrier,volatility,maturity)` evaluates a continuously
monitored, single-barrier option with zero rebate under constant-volatility GBM.
Choose `direction="up"/"down"`, `knock="in"/"out"` and `kind="call"/"put"`.
The result is the conditional expected terminal payoff, without discounting.
Parameters broadcast over batch axes; the last terminal-price axis is the
scenario axis. PyTorch/JAX parameters retain native derivatives away from kinks.

For an upper barrier, write `b=log(B/S0)`, `x=log(ST/S0)` and
`v=sigma**2*T`. If both endpoints are below the barrier, the Brownian reflection
principle gives `P(hit | ST)=exp(-2*b*(b-x)/v)`; otherwise it is one.
The lower-barrier construction reverses the log distances. Multiplying a
terminal payoff by the conditional survival/crossing probability integrates out
the path extremum. Stable `expm1` evaluation preserves survival probabilities
close to a barrier, and the initial spot is included in crossing checks.

Pass `option(terminal_grid)` to the existing `finance.price`, using either known
terminal probabilities or a coherent terminal preparation. This representation
needs no monitoring-date loop or running-maximum register. It also removes the
conditional Bernoulli crossing variance from a classical estimator. The static
representation adapts the October 2026 source below, using a conditional
one-variable payoff instead of its bivariate maximum register and linearized
rotation. Existing payoff encoding remains exact for the supplied discrete table.

The supplied terminal law must match the GBM parameters under the pricing
measure. Volatility and maturity must be positive. Discrete monitoring,
stochastic/local volatility, multiple barriers, rebates and historical hits
before the supplied initial spot are outside this model. No continuity correction
is silently applied. Select analytic formulas when available. Sampling intervals
cover sampling uncertainty; grid, truncation, market-model error and table
preparation costs require separate controls.

## Quantum Monte Carlo for pricing and tail risk

`oq.finance.price(payoffs,probabilities=p)` computes the native exact weighted
sum with no quantum engine. With coherent inputs, `method="auto"` first attempts
an admitted exact calculation, using the original scenario preparation for a
table. Use `method="quantum"` to select sampled estimation explicitly. `error`
is an absolute discounted payoff error, not a probability tolerance. The default
`strict=True` raises if the budget cannot deliver the requested statistical
bound; `strict=False` exposes the partial result and diagnostics.

Use `import openqml as oq` and `from openqml import finance as f`.
`PayoffOraclePlan(bound_preparation,payoffs,discount=...)` combines a coherent
scenario circuit with a bounded signed payoff. Scenario order is q0-MSB.
`preparation` must be an explicitly bound **unitary** `Circuit`, with no channels,
measurement, classical feedback or unresolved parameters. A static
`BornCircuitPlan.to_spec().circuit.bind(bindings)` is suitable when all its
registers correspond to the table; an entire dissipative recurrent model cannot
be inverted as a coherent oracle.

The plan lazily synthesizes an exact affine arcsin payoff encoding using a
Walsh transform and Gray-code controlled rotations. It preserves the signed
expectation through `offset + scale * probability`, without a small-angle payoff
approximation. Supply `control_values`, their independently known
`control_expectation`, and `control_coefficient` to estimate a narrower residual.
An incorrect control expectation biases the result. A constant residual returns
analytically with zero circuit calls, including when wider bounds were supplied.

Use `classical_value(probabilities)` when probabilities are already available:
this differentiable exact sum compiles no quantum circuit. For a quantum
expectation, construct `schedule = oq.AmplitudeEstimationPlan.geometric(
max_power=32,ratio=1.45,shots=128,confidence=0.95)` and call
`payoff.estimate(schedule,backend=...,seed=...)`. The returned `PayoffEstimate`
contains a value, possibly disconnected `intervals`, their `interval` hull, the
confidence level and the actual forward/inverse oracle query count. Constant
payoffs need no amplitude estimate.

For larger custom arithmetic oracles, use `oq.AmplitudeOracle(A,good_qubit)`
directly and `schedule.run(oracle,...)`. Amplification executes the real A,
its inverse and a zero-state reflection with clean linear workspace; it uses
neither controlled A calls nor quantum Fourier transforms. Independent depth
circuits are available through `schedule.circuits(oracle)` for external parallel
submission; `run` groups equal-shot tasks through the existing batched backend
interface and cost-aware hardware scheduler. External records can be supplied
to `fit_counts`, so inference does not require a particular provider.

The geometric non-power-of-two depth ladder and multimodal likelihood inference
adapt the September 2026 method cited below. Grid resolution accounts for the
largest depth and shot counts; every detected likelihood mode is refined. The
point estimate is numerical. Confidence sets are constructed independently:
per-depth binomial KL-Chernoff intervals receive a simultaneous union-bound error
budget, then **all** inverse-trigonometric branches are intersected. A shallow
single-start fit cannot silently discard an unresolved alias. The intervals use
simultaneous finite-shot coverage, with configurable depth and shot allocation.

`visibility` may be a calibrated value per depth, with the model
`P(1|m) = (1-v)/2 + v*sin((2*m+1)*theta)**2` and target probability
`sin(theta)**2`. Zero visibility provides no information. Incompatible records
raise `BackendError`; the model does not learn unknown device noise. Confidence
requires independent binomial measurements under this specified model and a
fixed shot/depth schedule. It does not cover calibration, discretization,
market-model, payoff-truncation or floating-point errors.

### Adaptive precision and depth selection

The high-level quantum path defaults to `AdaptiveAmplitudeEstimationPlan`.
At each registered depth, a Jeffreys beta-binomial mixture e-process gives a
confidence sequence: its log mixture likelihood minus the candidate binomial
log likelihood is bounded by `log(L/alpha)`, where `L` is the number of registered
depths. Ville's inequality and a union bound cover all rounds, adaptive depth
selection and optional stopping under the fixed calibrated binomial model.
All inverse-Grover branches are retained and successive confidence sets
intersected. The reported point is the interval-hull midpoint; `error_bound` is
its worst-case distance to that hull, rather than a posterior credible width.

Depth selection scores local Fisher information per supplied circuit cost,
subject to a phase-window safeguard, measured visibility and query/work/memory
budgets. `costs` describes a per-shot execution cost and `batch_cost` a fixed
submission overhead; this heuristic is not a globally optimal depth policy.
Confidence coverage does not depend on the heuristic being optimal. A query
counts every forward/inverse oracle call, including the calls inside Grover
amplification. Noisy depths with zero contrast carry no information.

This adapts contrast-aware financial estimation and time-uniform inference from
the sources below. It does not implement their complete Bayesian CVA workflow or
tomography procedure. Visibility must be calibrated independently and remain
valid throughout execution; confidence does not cover an unknown or drifting
noise model. The plan stops at the requested precision or returns an explicit
unconverged result when the query/round budget is exhausted.

Digital loss payoffs `1[loss > threshold]` estimate tail probabilities; positive
excess-loss payoffs `max(loss-threshold,0)` estimate shortfall integrals. Both
reuse the same oracle and estimator, without separate risk-specific circuit
engines. A market price additionally needs the correct risk-neutral measure.

Inspect `schedule.resource_profile(oracle)` before execution: it includes
state preparation, inverse preparation, reflection workspace, shot counts and
portable IR operation counts. Efficient coherent preparation/payoff arithmetic
and useful circuit depth determine the sampled method's resource needs. A
payoff table has O(S) storage/gates and O(S log S) synthesis; classical summation
over known probabilities is O(S). Compare complete workloads including table
loading, transpilation, error correction, queue time and classical variance
reduction or quasi-Monte-Carlo execution.

## Method sources

- [Learning Predictive Ambiguity Sets for Decision-Focused Distributionally
  Robust Optimization](https://arxiv.org/abs/2607.09820) (July 2026):
  motivates contextual scenario masses and learned ambiguity radii. OpenQML's
  cash-ray layer uses an isotropic W1/L2 margin with a closed-form funded risk
  projection; trainable anisotropic metrics require a separate risk model.

- [Constant-Depth Quantum Barrier Option Pricing via a Static Bivariate Oracle Framework](https://arxiv.org/abs/2610.27929) (October 2026): static path compression under GBM. OpenQML uses terminal-price conditioning, exact discrete payoff rotations and explicit state-loading costs for continuous monitoring.

- [Integrated Prediction and Multi-period Portfolio Optimization](https://arxiv.org/abs/2512.11273): decision-focused training and fixed-point differentiation. OpenQML implements its own single-stage active-face factor solver, not the full multi-period algorithm.
- [Constrained Counterdiabatic Quantum Approximate Optimization Algorithm for Portfolio Optimization](https://arxiv.org/abs/2605.06858): number-preserving commutator generators; the adaptation above omits full gauge-potential optimization.
- [The Lie Algebra of XY-mixer Topologies and Warm Starting QAOA for Constrained Optimization](https://doi.org/10.1038/s41534-026-01192-4), npj Quantum Information 12, 61 (2026): topology, expressivity and trainability tradeoffs of constrained mixers.
- [Large-scale portfolio optimization on a trapped-ion quantum computer](https://arxiv.org/abs/2602.23976): hybrid decomposition motivation for hardware-sized asset-selection subproblems.
- [Shift-Aware Gaussian-Supremum Validation for Wasserstein-DRO CVaR Portfolios](https://arxiv.org/abs/2512.16748): robust tail-risk formulation. Radius calibration and statistical validation are separate from the implemented margin.
- [Quantum generative modeling for financial time series with temporal correlations](https://doi.org/10.1088/2632-2153/ae39a2), Machine Learning: Science and Technology 7, 015027 (2026): financial quantum generative modeling; the finite-atom Born model here differs from its QGAN and MPS experiments.
- [Quantum amplitude estimation beyond power-of-two schedules](https://arxiv.org/abs/2609.02715) (September 2026): geometric depth schedules and full multimodal likelihood inference. OpenQML uses its own conservative finite-shot confidence construction and configurable shot allocation.
- [A Noise-Aware Quantum Algorithm for Credit Valuation Adjustments on Real Quantum Hardware](https://arxiv.org/abs/2607.12990) (September 2026 revision): motivates contrast-aware depth and cost selection. OpenQML uses a frequentist confidence sequence rather than the paper's complete inference workflow.
- [Anytime-Valid Quantum State Tomography via Confidence Sequences](https://arxiv.org/abs/2601.20761), IEEE Signal Processing Letters 33 (2026): time-uniform quantum inference; the implementation here concerns binomial amplitude estimation, not tomography.
- [Time-uniform, nonparametric, nonasymptotic confidence sequences](https://arxiv.org/abs/1810.08240), Annals of Statistics 49(2), 1055–1080 (2021): mixture-martingale confidence sequences and optional-stopping coverage.
- [Goldfarb–Idnani Revisited: Invariants, Certificates, and the Limits of Guessing](https://arxiv.org/abs/2608.30933) (2026): motivates certification of a proposed active set. OpenQML retains its specialized factor-risk solver and implicit derivatives.
