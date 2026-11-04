"""Differentiable quantitative finance on OpenQML's existing model/runtime core.

Forecast and scenario plans create ordinary HybridArchitecture programs.
Financial objectives, factor solvers and accounting retain native arrays and
gradients. Cardinality problems compile to portable, resource-admitted circuits.
"""

from .allocation import AllocationPlan, AllocationResult
from .models import MarketForecast, MarketForecastPlan, ScenarioPlan
from .projection import cost_aware_projection
from .quantum import (
    CardinalityProblem,
    CardinalityQAOAPlan,
    PortfolioSamples,
    PortfolioSelection,
    SelectionResult,
)
from .risk import (
    FactorRisk,
    covariance_matvec,
    dense_covariance,
    expected_shortfall,
    factor_risk,
    portfolio_variance,
    quantile_loss,
    wasserstein_cvar,
)
from .trading import SelfFinancingPlan, TradingResult, TradingState
from .validation import FinancialFold, PurgedWalkForward

__all__ = [
    "AllocationPlan",
    "AllocationResult",
    "MarketForecast",
    "MarketForecastPlan",
    "ScenarioPlan",
    "cost_aware_projection",
    "CardinalityProblem",
    "CardinalityQAOAPlan",
    "PortfolioSamples",
    "PortfolioSelection",
    "SelectionResult",
    "FactorRisk",
    "covariance_matvec",
    "dense_covariance",
    "expected_shortfall",
    "factor_risk",
    "portfolio_variance",
    "quantile_loss",
    "wasserstein_cvar",
    "SelfFinancingPlan",
    "TradingResult",
    "TradingState",
    "FinancialFold",
    "PurgedWalkForward",
]
