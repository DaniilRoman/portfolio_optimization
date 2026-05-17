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

from src.logic.data.data import (
    ProfitabilityData,
    StockData,
    StockInfo,
    TickerMetadata,
)
from src.logic.data.forecast import Forecast


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


def make_stock_data(**overrides: Any) -> StockData:
    defaults: dict[str, Any] = dict(
        ticker_symbol="TST",
        stock_name="Test ETF",
        currency="USD",
        current_price=100.0,
        predict_price=110.0,
        two_year_file_name="/tmp/tst_2y.png",
        five_year_file_name="/tmp/tst_5y.png",
        is_stock_growing=True,
        industry="ETF",
        profitability_data=make_profitability_data(),
        beta=1.0,
        standard_deviation=0.12,
        dividend_yield=0.02,
        top_holdings=np.array([["Apple Inc", 0.08], ["Microsoft Corp", 0.07]]),
        sector_allocation={"Technology": 0.45, "Healthcare": 0.30, "Financials": 0.25},
        average_daily_volume=1_000_000.0,
        assets_under_management=1_000_000_000.0,
        expense_ratio=0.0003,
        description="",
        forecast_volatility=0.0,
        prediction_uncertainty=0.0,
    )
    defaults.update(overrides)
    return StockData(**defaults)  # type: ignore[arg-type]


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
