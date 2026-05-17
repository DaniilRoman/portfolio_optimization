"""Tests for analyzer.py"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.adapter.out.analyze import analyzer
from src.logic.data.data import ProfitabilityData, StockData, StockInfo
from src.logic.data.forecast import Forecast
from tests.factories import make_forecast, make_stock_info, make_ticker_metadata


def test_last_price():
    """Test __last_price function"""
    print("Testing __last_price function...")
    from src.adapter.out.analyze.analyzer import __last_price

    test_data = pd.DataFrame({"y": [100.0, 101.5, 102.3, 103.7, 104.2]})
    result = __last_price(test_data, "y")
    assert round(result, 2) == round(104.2, 2), f"Expected 104.2, got {result}"
    print("  ✅ __last_price test passed")


def test_slice():
    """Test __slice function"""
    print("Testing __slice function...")
    from src.adapter.out.analyze.analyzer import __slice

    dates = pd.date_range(start="2020-01-01", periods=365 * 5, freq="D")
    historic_data = pd.DataFrame({"y": np.random.randn(len(dates)) * 10 + 100}, index=dates)
    result = __slice(historic_data, 365 * 2, 30)
    assert isinstance(result, pd.DataFrame)
    if len(result) > 0:
        print(f"  Sliced {len(result)} rows")
        print(f"  Date range: {result.index[0]} to {result.index[-1]}")
    print("  ✅ __slice test passed")


def test_is_stock_historicly_growing():
    """Test __is_stock_historicly_growing function"""
    print("Testing __is_stock_historicly_growing function...")
    from src.adapter.out.analyze.analyzer import __is_stock_historicly_growing

    history_slice = pd.DataFrame({"y": [100.0, 110.0, 120.0, 130.0, 140.0]})

    assert __is_stock_historicly_growing(150.0, history_slice), "Should return True when current > all history"
    print("  ✅ Case 1: Current price higher than history - passed")

    assert __is_stock_historicly_growing(125.0, history_slice), "Should return True when current > some history"
    print("  ✅ Case 2: Current price lower than some history - passed")

    assert not __is_stock_historicly_growing(90.0, history_slice), "Should return False when current < all history"
    print("  ✅ Case 3: Current price lower than all history - passed")


def test_is_stock_growing():
    """Test __is_stock_growing function"""
    print("Testing __is_stock_growing function...")
    from src.adapter.out.analyze.analyzer import __is_stock_growing

    dates = pd.date_range(start="2020-01-01", periods=365 * 5, freq="D")
    historic_data = pd.DataFrame({"y": np.linspace(100, 200, len(dates))}, index=dates)

    result = __is_stock_growing(180.0, 200.0, 220.0, historic_data)
    print(f"  Growing stock test result: {result}")

    result_non_growing = __is_stock_growing(180.0, 150.0, 120.0, historic_data)
    assert not result_non_growing, "Should return False when predicted prices are lower"
    print("  ✅ Non-growing stock test passed")


def _run_analyses(ticker_symbol: str, stock_info: StockInfo, two_year_forecast: Forecast, five_year_forecast: Forecast) -> StockData:
    with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.figure"):
        return analyzer.analyses(
            ticker_symbol=ticker_symbol,
            stock_info=stock_info,
            two_year_forecast=two_year_forecast,
            five_year_forecast=five_year_forecast,
        )


def test_analyses_function():
    """Test the main analyses function"""
    print("Testing analyses function...")
    ticker_symbol = "VOO"
    stock_info = make_stock_info(ticker=make_ticker_metadata(trailing_eps=5.0, forward_eps=5.5))
    two_year_forecast = make_forecast(base=200.0)
    five_year_forecast = make_forecast(base=210.0)

    result = _run_analyses(ticker_symbol, stock_info, two_year_forecast, five_year_forecast)

    assert isinstance(result, StockData), "Result should be a StockData object"
    print("  ✅ Returned StockData object")

    assert result.ticker_symbol == ticker_symbol
    assert result.stock_name == "Test ETF"
    assert result.currency == "USD"
    assert result.industry == "ETF"
    assert result.beta == 1.0
    assert isinstance(result.standard_deviation, float)
    assert result.dividend_yield == 0.02
    assert result.assets_under_management == 1_000_000_000.0
    assert result.expense_ratio == 0.0003

    assert isinstance(result.profitability_data, ProfitabilityData)
    assert result.profitability_data.trailing_eps == 5.0
    assert result.two_year_file_name == f"two_year_{ticker_symbol}.png"
    assert result.five_year_file_name == f"five_year_{ticker_symbol}.png"
    print("  ✅ All StockData fields correctly populated")


def test_analyses_with_missing_fund_data():
    """Test analyses function with zeroed-out optional fields"""
    print("Testing analyses with missing fund data...")
    ticker_symbol = "TEST"
    stock_info = make_stock_info(ticker=make_ticker_metadata(total_assets=0, expense_ratio=0, dividend_yield=0))
    two_year_forecast = make_forecast()
    five_year_forecast = make_forecast()

    result = _run_analyses(ticker_symbol, stock_info, two_year_forecast, five_year_forecast)

    print(f"  assets_under_management: {result.assets_under_management}")
    print(f"  expense_ratio: {result.expense_ratio}")
    print(f"  dividend_yield: {result.dividend_yield}")
    assert result.assets_under_management == 0, f"Expected 0, got {result.assets_under_management}"
    assert result.expense_ratio == 0, f"Expected 0, got {result.expense_ratio}"
    assert result.dividend_yield == 0, f"Expected 0, got {result.dividend_yield}"
    print("  ✅ Missing fund data handled correctly")


def test_analyses_with_empty_description():
    """Test analyses function when description is empty"""
    print("Testing analyses with empty description...")
    ticker_symbol = "TEST"
    stock_info = make_stock_info(ticker=make_ticker_metadata(description=""))
    two_year_forecast = make_forecast()
    five_year_forecast = make_forecast()

    result = _run_analyses(ticker_symbol, stock_info, two_year_forecast, five_year_forecast)

    assert result.description == ""
    print("  ✅ Empty description handled correctly")


def test_pessimistic_predict_price_calculation():
    """Test the pessimistic predict_price uses min of yhat_lower values"""
    print("Testing pessimistic predict_price calculation...")
    ticker_symbol = "TEST"
    stock_info = make_stock_info()

    future_dates = pd.date_range(start=datetime.now(), periods=10, freq="D")
    mock_model = MagicMock()

    two_year_forecast = Forecast(
        series=pd.DataFrame({"yhat": [200.0] * 10, "yhat_lower": [195.0] * 10, "yhat_upper": [205.0] * 10}, index=future_dates),
        model=mock_model,
    )
    five_year_forecast = Forecast(
        series=pd.DataFrame({"yhat": [210.0] * 10, "yhat_lower": [205.0] * 10, "yhat_upper": [215.0] * 10}, index=future_dates),
        model=mock_model,
    )

    result = _run_analyses(ticker_symbol, stock_info, two_year_forecast, five_year_forecast)

    expected_predict_price = 195.0
    print(f"  Expected predict_price: {expected_predict_price}")
    print(f"  Actual predict_price: {result.predict_price}")
    assert round(result.predict_price, 2) == round(expected_predict_price, 2)
    print("  ✅ Pessimistic predict_price calculation test passed")


def test_pessimistic_predict_price_with_different_values():
    """Test pessimistic calculation when five-year minimum is lower"""
    print("Testing pessimistic predict_price with different values...")
    ticker_symbol = "TEST"
    stock_info = make_stock_info()

    future_dates = pd.date_range(start=datetime.now(), periods=10, freq="D")
    mock_model = MagicMock()

    two_year_forecast = Forecast(
        series=pd.DataFrame({"yhat": [205.0] * 10, "yhat_lower": [200.0] * 10, "yhat_upper": [210.0] * 10}, index=future_dates),
        model=mock_model,
    )
    five_year_forecast = Forecast(
        series=pd.DataFrame({"yhat": [195.0] * 10, "yhat_lower": [190.0] * 10, "yhat_upper": [200.0] * 10}, index=future_dates),
        model=mock_model,
    )

    result = _run_analyses(ticker_symbol, stock_info, two_year_forecast, five_year_forecast)

    expected_predict_price = 190.0
    print(f"  Expected predict_price: {expected_predict_price}")
    print(f"  Actual predict_price: {result.predict_price}")
    assert round(result.predict_price, 2) == round(expected_predict_price, 2)
    print("  ✅ Pessimistic predict_price with different values test passed")
