"""Shared test fixture factories.

Each builder accepts **overrides for any field so callers only specify what
matters for a given test. Default values are minimal but structurally valid.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.domain.data.data import (
    Allocation,
    AnalysisReport,
    Asset,
    ForecastSummary,
    MarketSnapshot,
    OptimizationResult,
    Portfolio,
    ProfitabilityData,
    RiskMetrics,
    StockInfo,
    TickerMetadata,
)
from src.domain.data.forecast import Forecast


def make_profitability_data(**overrides: Any) -> ProfitabilityData:
    defaults: dict[str, Any] = dict(
        trailing_eps=2.0,
        forward_eps=2.5,
        netIncome_to_common=1_000_000.0,
        ebitda_margins=0.3,
        operating_margins=0.25,
    )
    defaults.update(overrides)
    return ProfitabilityData(**defaults)


def make_ticker_metadata(**overrides: Any) -> TickerMetadata:
    defaults: dict[str, Any] = dict(
        long_name="Test ETF",
        currency="USD",
        industry="ETF",
        beta=1.0,
        dividend_yield=0.02,
        total_assets=1_000_000_000.0,
        expense_ratio=0.0003,
        average_volume=1_000_000.0,
        trailing_eps=2.0,
        forward_eps=2.5,
        net_income_to_common=1_000_000.0,
        ebitda_margins=0.3,
        operating_margins=0.25,
        top_holdings=np.array([["Apple Inc", 0.08], ["Microsoft Corp", 0.07]]),
        sector_weights={"Technology": 0.45, "Healthcare": 0.30, "Financials": 0.25},
        description="Test Family || ETF || Test ETF Description",
    )
    defaults.update(overrides)
    return TickerMetadata(**defaults)  # type: ignore[arg-type]


def make_asset(**overrides: Any) -> Asset:
    defaults: dict[str, Any] = dict(
        ticker_symbol="TST",
        stock_name="Test ETF",
        currency="USD",
        industry="ETF",
        description="",
        expense_ratio=0.0003,
        assets_under_management=1_000_000_000.0,
    )
    defaults.update(overrides)
    return Asset(**defaults)


def make_market_snapshot(**overrides: Any) -> MarketSnapshot:
    defaults: dict[str, Any] = dict(
        current_price=100.0,
        beta=1.0,
        standard_deviation=0.12,
        dividend_yield=0.02,
        average_daily_volume=1_000_000.0,
    )
    defaults.update(overrides)
    return MarketSnapshot(**defaults)


def make_forecast_summary(**overrides: Any) -> ForecastSummary:
    defaults: dict[str, Any] = dict(
        predict_price=110.0,
        prediction_uncertainty=0.0,
        forecast_volatility=0.0,
        two_year_file_name="/tmp/tst_2y.png",
        five_year_file_name="/tmp/tst_5y.png",
    )
    defaults.update(overrides)
    return ForecastSummary(**defaults)


def make_stock_data(**overrides: Any) -> AnalysisReport:
    """Build an AnalysisReport accepting flat field overrides for backward compat."""
    # Pull flat-kwarg overrides that belong to sub-objects
    asset_fields = {k: overrides.pop(k) for k in list(overrides) if k in {
        "ticker_symbol", "stock_name", "currency", "industry", "description",
        "expense_ratio", "assets_under_management",
    }}
    market_fields = {k: overrides.pop(k) for k in list(overrides) if k in {
        "current_price", "beta", "standard_deviation", "dividend_yield", "average_daily_volume",
    }}
    forecast_fields = {k: overrides.pop(k) for k in list(overrides) if k in {
        "predict_price", "prediction_uncertainty", "forecast_volatility",
        "two_year_file_name", "five_year_file_name",
    }}

    asset = overrides.pop("asset", make_asset(**asset_fields))
    market = overrides.pop("market", make_market_snapshot(**market_fields))
    forecast = overrides.pop("forecast", make_forecast_summary(**forecast_fields))

    defaults: dict[str, Any] = dict(
        asset=asset,
        market=market,
        forecast=forecast,
        top_holdings=np.array([["Apple Inc", 0.08], ["Microsoft Corp", 0.07]]),
        sector_allocation={"Technology": 0.45, "Healthcare": 0.30, "Financials": 0.25},
        is_stock_growing=True,
        profitability_data=make_profitability_data(),
    )
    defaults.update(overrides)
    return AnalysisReport(**defaults)


def make_historic_df(n: int = 800) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=n)
    dates = pd.date_range(start=start, end=end, freq="D")
    prices = np.abs(100.0 + np.linspace(0, 50, len(dates)) + np.random.default_rng(42).normal(0, 5, len(dates)))
    df = pd.DataFrame({"y": prices}, index=dates)
    df["ds"] = pd.to_datetime(df.index.date)
    return df


def make_stock_info(**overrides: Any) -> StockInfo:
    defaults: dict[str, Any] = dict(
        historic_data=make_historic_df(),
        ticker=make_ticker_metadata(),
    )
    defaults.update(overrides)
    return StockInfo(**defaults)


def make_forecast_df(base: float = 150.0, n: int = 10) -> pd.DataFrame:
    future_dates = pd.date_range(start=datetime.now(), periods=n, freq="D")
    predicted = np.full(n, base)
    return pd.DataFrame(
        {
            "ds": future_dates,
            "y": predicted,
            "yhat": predicted,
            "yhat_lower": predicted - 5,
            "yhat_upper": predicted + 5,
            "uncertainty_range": np.full(n, 10.0),
            "volatility_forecast": np.full(n, 0.15),
        },
        index=future_dates,
    )


def make_forecast(base: float = 150.0, n: int = 10, **overrides: Any) -> Forecast:
    defaults: dict[str, Any] = dict(
        series=make_forecast_df(base, n),
        model=MagicMock(),
    )
    defaults.update(overrides)
    return Forecast(**defaults)


def make_allocation(**overrides: Any) -> Allocation:
    asset = overrides.pop("asset", None) or make_asset()
    defaults: dict[str, Any] = dict(
        asset=asset,
        quantity=2,
        total_cost=200.0,
        net_profit=20.0,
        capital_gain=15.0,
        dividend_income=4.0,
        expense_fee=1.0,
        expense_ratio=0.0003,
        forecast_volatility=0.0,
    )
    defaults.update(overrides)
    return Allocation(**defaults)


def make_portfolio(**overrides: Any) -> Portfolio:
    defaults: dict[str, Any] = dict(
        allocations=[make_allocation()],
        risk_metrics=RiskMetrics(volatility=0.12, sector_concentration=0.4, company_overlap=0.05),
    )
    defaults.update(overrides)
    return Portfolio(**defaults)


def make_optimization_result(**overrides: Any) -> OptimizationResult:
    defaults: dict[str, Any] = dict(
        risk_aware=make_portfolio(),
        profit_only=make_portfolio(),
    )
    defaults.update(overrides)
    return OptimizationResult(**defaults)
