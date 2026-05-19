"""Core domain types: Asset, MarketSnapshot, ForecastSummary, AnalysisReport, StockInfo, TickerMetadata, ProfitabilityData, OptimizationResult, Portfolio, Allocation, RiskMetrics."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ProfitabilityData:
    trailing_eps: float
    forward_eps: float
    netIncome_to_common: float
    ebitda_margins: float
    operating_margins: float

    def is_profitable(self) -> bool:
        has_positive_earnings = self.trailing_eps > 0 or self.forward_eps > 0
        has_positive_net_income = self.netIncome_to_common > 0
        has_positive_margins = self.ebitda_margins > 0 or self.operating_margins > 0
        return (has_positive_earnings or has_positive_net_income) and has_positive_margins


@dataclass
class TickerMetadata:
    long_name: str
    currency: str
    industry: str
    beta: float
    dividend_yield: float
    total_assets: float
    expense_ratio: float
    average_volume: float
    trailing_eps: float
    forward_eps: float
    net_income_to_common: float
    ebitda_margins: float
    operating_margins: float
    top_holdings: np.ndarray
    sector_weights: dict[str, float]
    description: str


@dataclass
class Asset:
    """Identity and static metadata for a stock or ETF."""

    ticker_symbol: str
    stock_name: str
    currency: str
    industry: str
    description: str
    expense_ratio: float
    assets_under_management: float


@dataclass
class MarketSnapshot:
    """Current market data for a stock or ETF."""

    current_price: float
    beta: float
    standard_deviation: float
    dividend_yield: float
    average_daily_volume: float


@dataclass
class ForecastSummary:
    """Prediction outputs for a stock or ETF."""

    predict_price: float
    prediction_uncertainty: float
    forecast_volatility: float
    two_year_file_name: str
    five_year_file_name: str
    expected_log_return: float = 0.0  # annualised log return; input view for Black-Litterman (§4.3)


@dataclass
class AnalysisReport:
    """Full analysis result composing Asset, MarketSnapshot, and ForecastSummary."""

    asset: Asset
    market: MarketSnapshot
    forecast: ForecastSummary
    top_holdings: np.ndarray
    sector_allocation: dict[str, float]
    is_stock_growing: bool
    profitability_data: ProfitabilityData
    momentum_12_1: float = 0.0  # 12-1 month price return; momentum factor for §4.5 scoring


# Backward-compatible alias so existing imports of StockData keep working.
StockData = AnalysisReport


@dataclass
class StockInfo:
    historic_data: pd.DataFrame
    ticker: TickerMetadata
    point_in_time: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.today().normalize())


class SkipException(Exception):
    pass


@dataclass
class Allocation:
    asset: Asset
    quantity: int
    total_cost: float
    net_profit: float
    capital_gain: float
    dividend_income: float
    expense_fee: float
    expense_ratio: float
    forecast_volatility: float = 0.0


@dataclass
class RiskMetrics:
    volatility: float
    sector_concentration: float
    company_overlap: float


@dataclass
class Portfolio:
    allocations: list[Allocation]
    risk_metrics: RiskMetrics


@dataclass
class OptimizationResult:
    risk_aware: Portfolio
    profit_only: Portfolio
