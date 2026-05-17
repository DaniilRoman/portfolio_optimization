"""Tests for predicter (GARCH-based)."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.adapter.out.predict import predicter
from src.logic.data.forecast import Forecast


def create_test_data(days: int = 365 * 5, start_date: datetime | None = None) -> pd.DataFrame:
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)

    dates = pd.date_range(start=start_date, periods=days, freq="D")
    np.random.seed(42)

    base_price = 100.0
    trend = np.linspace(0, 50, len(dates))
    weekly_seasonality = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    yearly_seasonality = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 3, len(dates))

    prices = np.abs(base_price + trend + weekly_seasonality + yearly_seasonality + noise)

    return pd.DataFrame({"ds": dates, "y": prices})


def test_predict_basic_functionality():
    test_data = create_test_data(days=120)

    result = predicter.predict(test_data, predict_period=30)

    assert isinstance(result, Forecast)
    assert hasattr(result, "plot")
    assert hasattr(result.model, "plot")
    df = result.series
    assert isinstance(df, pd.DataFrame)
    assert {"ds", "yhat", "yhat_lower", "yhat_upper", "volatility_forecast"}.issubset(df.columns)
    assert len(df) == len(test_data) + 30

    future = df.tail(30)
    assert (future["yhat_upper"] >= future["yhat_lower"]).all()
    assert (future["volatility_forecast"] >= 0).all()


def test_predict_with_custom_parameters_is_compatible():
    test_data = create_test_data(days=200)

    result = predicter.predict(test_data, predict_period=20, interval_width=0.8)
    df = result.series

    assert len(df) == len(test_data) + 20
    assert df["yhat"].notna().all()


def test_predict_error_handling():
    minimal_data = pd.DataFrame({"ds": [datetime.now()], "y": [100.0]})

    with pytest.raises(ValueError, match="Insufficient data for prediction"):
        predicter.predict(minimal_data, predict_period=10)

    data_with_nan = pd.DataFrame(
        {
            "ds": [datetime.now() - timedelta(days=2), datetime.now() - timedelta(days=1), datetime.now()],
            "y": [100.0, np.nan, 102.0],
        }
    )

    result = predicter.predict(data_with_nan, predict_period=5)
    assert len(result.series) == 2 + 5

    data_mixed_dates = pd.DataFrame({"ds": ["2023-01-01", "2023-01-02", "invalid-date"], "y": [100.0, 101.0, 102.0]})

    result = predicter.predict(data_mixed_dates, predict_period=5)
    assert len(result.series) > 0


def test_uncertainty_band_grows_with_horizon():
    test_data = create_test_data(days=300)

    result = predicter.predict(test_data, predict_period=60)
    future = result.series.tail(60).reset_index(drop=True)

    first_band = float(future["yhat_upper"].iloc[0] - future["yhat_lower"].iloc[0])
    last_band = float(future["yhat_upper"].iloc[-1] - future["yhat_lower"].iloc[-1])
    assert last_band > first_band, f"expected widening band, got first={first_band:.4f} last={last_band:.4f}"


def test_interval_width_controls_band_width():
    test_data = create_test_data(days=300)

    narrow = predicter.predict(test_data, predict_period=30, interval_width=0.5)
    wide = predicter.predict(test_data, predict_period=30, interval_width=0.95)

    narrow_band = (narrow.series["yhat_upper"] - narrow.series["yhat_lower"]).tail(30).mean()
    wide_band = (wide.series["yhat_upper"] - wide.series["yhat_lower"]).tail(30).mean()
    assert wide_band > narrow_band


def test_constant_price_input_does_not_raise():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    data = pd.DataFrame({"ds": dates, "y": np.full(len(dates), 100.0)})

    result = predicter.predict(data, predict_period=10)
    df = result.series

    assert len(df) == len(data) + 10
    last_price = float(data["y"].iloc[-1])
    np.testing.assert_allclose(df["yhat"].tail(10).to_numpy(), last_price)


def test_drift_makes_yhat_non_flat_on_trending_series():
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    log_prices = np.linspace(np.log(100.0), np.log(200.0), len(dates))
    data = pd.DataFrame({"ds": dates, "y": np.exp(log_prices)})

    result = predicter.predict(data, predict_period=30)
    future_yhat = result.series["yhat"].tail(30).to_numpy()
    last_price = float(data["y"].iloc[-1])
    assert future_yhat[-1] > last_price, "expected yhat to drift up on an uptrending series"
    assert (np.diff(future_yhat) > 0).all(), "expected monotonically increasing yhat under positive drift"
