"""Unit tests for the downloader adapter."""

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

from src.adapter.out.download.downloader import _extract_ticker_metadata, download_stock_data
from src.logic.data.data import SkipException, StockInfo, TickerMetadata


def _make_hist_df(n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = np.linspace(100, 110, n)
    df = pd.DataFrame({"Close": prices}, index=dates)
    df.index.name = "Date"
    return df


def _make_mock_ticker(hist: pd.DataFrame | None = None) -> MagicMock:
    mock = MagicMock()
    mock.history.return_value = hist if hist is not None else _make_hist_df()

    mock.info = {
        "longName": "Test ETF",
        "currency": "EUR",
        "industry": "ETF",
        "averageVolume": 500_000.0,
        "beta": 0.95,
        "totalAssets": 2_000_000_000.0,
        "netExpenseRatio": 0.07,
        "yield": 0.015,
        "trailingEps": 3.0,
        "forwardEps": 3.5,
        "netIncomeToCommon": 500_000.0,
        "ebitdaMargins": 0.25,
        "operatingMargins": 0.20,
    }

    mock_funds = MagicMock()
    mock_funds.top_holdings.values = np.array([["AAPL", 0.05], ["MSFT", 0.04]])
    mock_funds.sector_weightings = {"Technology": 0.6, "Healthcare": 0.4}
    mock_funds.fund_overview = {"family": "iShares", "legalType": "ETF"}
    mock_funds.description = "A diversified ETF"
    mock.funds_data = mock_funds

    return mock


class TestExtractTickerMetadata:
    def test_happy_path_populates_all_fields(self):
        mock_ticker = _make_mock_ticker()
        meta = _extract_ticker_metadata(mock_ticker)

        assert isinstance(meta, TickerMetadata)
        assert meta.long_name == "Test ETF"
        assert meta.currency == "EUR"
        assert meta.industry == "ETF"
        assert meta.beta == 0.95
        assert meta.total_assets == 2_000_000_000.0
        assert abs(meta.expense_ratio - 0.0007) < 1e-9  # 0.07% / 100
        assert meta.dividend_yield == 0.015
        assert meta.trailing_eps == 3.0
        assert meta.forward_eps == 3.5
        assert meta.average_volume == 500_000.0
        assert len(meta.top_holdings) == 2
        assert "Technology" in meta.sector_weights

    def test_missing_info_fields_default_to_zero(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_ticker.funds_data = None

        meta = _extract_ticker_metadata(mock_ticker)

        assert meta.long_name == "Unknown"
        assert meta.beta == 0.0
        assert meta.dividend_yield == 0.0
        assert meta.expense_ratio == 0.0
        assert meta.top_holdings.size == 0
        assert meta.sector_weights == {}

    def test_funds_data_exception_returns_empty_defaults(self):
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        type(mock_ticker).funds_data = PropertyMock(side_effect=Exception("network error"))

        meta = _extract_ticker_metadata(mock_ticker)

        assert meta.top_holdings.size == 0
        assert meta.description == ""

    def test_expense_ratio_divided_by_100(self):
        mock_ticker = _make_mock_ticker()
        mock_ticker.info["netExpenseRatio"] = 0.20  # 0.20%
        meta = _extract_ticker_metadata(mock_ticker)
        assert abs(meta.expense_ratio - 0.002) < 1e-9


class TestDownloadStockData:
    def test_happy_path_returns_stock_info_with_correct_schema(self):
        mock_ticker = _make_mock_ticker()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = download_stock_data("VOO", start_date="2024-01-01", end_date="2024-01-15")

        assert isinstance(result, StockInfo)
        assert isinstance(result.ticker, TickerMetadata)
        assert "y" in result.historic_data.columns
        assert "ds" in result.historic_data.columns
        assert not result.historic_data.empty

    def test_empty_history_raises_skip_exception(self):
        mock_ticker = _make_mock_ticker(hist=pd.DataFrame())
        with patch("yfinance.Ticker", return_value=mock_ticker):
            with pytest.raises(SkipException):
                download_stock_data("INVALID", start_date="2024-01-01", end_date="2024-01-15")

    def test_historic_data_has_no_duplicates(self):
        hist = _make_hist_df(n=20)
        mock_ticker = _make_mock_ticker(hist=hist)
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = download_stock_data("VOO", start_date="2024-01-01", end_date="2024-02-01")

        assert result.historic_data["ds"].is_unique

    def test_default_dates_used_when_not_provided(self):
        mock_ticker = _make_mock_ticker()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            with patch("src.adapter.out.download.downloader.utils.current_date_str", return_value="2024-01-01"):
                with patch("src.adapter.out.download.downloader.utils.next_day", return_value="2024-01-02"):
                    result = download_stock_data("VOO")
        assert isinstance(result, StockInfo)
