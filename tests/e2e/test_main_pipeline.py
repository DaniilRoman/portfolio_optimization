"""E2E happy-path test for the full analysis pipeline.

Mocks all network I/O (yfinance, telepot, counter URL) so the test runs
offline and deterministically.  The fixture CSV in tests/fixtures/TST_history.csv
provides a small but realistic price history.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.logic.data.data import StockData

FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"


def _load_fixture_hist() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "TST_history.csv")
    df.index = pd.to_datetime(df["Date"])
    df.index.name = "Date"
    return df[["Close"]]


def _make_mock_ticker(hist: pd.DataFrame) -> MagicMock:
    mock = MagicMock()
    mock.history.return_value = hist
    mock.info = {
        "longName": "Test ETF",
        "currency": "USD",
        "industry": "ETF",
        "averageVolume": 1_000_000.0,
        "beta": 1.0,
        "totalAssets": 5e9,
        "netExpenseRatio": 0.03,
        "yield": 0.012,
        "trailingEps": 2.0,
        "forwardEps": 2.4,
        "netIncomeToCommon": 1e6,
        "ebitdaMargins": 0.20,
        "operatingMargins": 0.15,
    }
    mock_funds = MagicMock()
    mock_funds.top_holdings.values = np.array([["AAPL", 0.07]])
    mock_funds.sector_weightings = {"technology": 0.60}
    mock_funds.fund_overview = {"family": "Vanguard", "legalType": "ETF"}
    mock_funds.description = "Tracks a broad index"
    mock.funds_data = mock_funds
    return mock


def _run_pipeline(stock_name: str = "TST") -> StockData | None:
    """Run the full pipeline with all network I/O mocked."""
    hist = _load_fixture_hist()
    mock_ticker = _make_mock_ticker(hist)

    with (
        patch("yfinance.Ticker", return_value=mock_ticker),
        patch("src.logic.stock_finder.notifier.notify") as _m_notify,
        patch("src.logic.stock_finder.stats_calculator.calculate"),
        patch("os.remove"),
        patch("matplotlib.pyplot.savefig"),
        patch("matplotlib.pyplot.close"),
    ):
        from src.logic import stock_finder

        return stock_finder.run(stock_name)


class TestMainPipelineHappyPath:
    def test_pipeline_produces_recommendation(self) -> None:
        result = _run_pipeline()
        assert result is not None, "Pipeline should return a StockData, not None"

    def test_pipeline_result_has_valid_prices(self) -> None:
        result = _run_pipeline()
        assert result is not None
        assert result.current_price > 0
        assert result.predict_price > 0
        assert result.currency == "USD"

    def test_pipeline_calls_notifier(self) -> None:
        hist = _load_fixture_hist()
        mock_ticker = _make_mock_ticker(hist)
        mock_notify = MagicMock()

        with (
            patch("yfinance.Ticker", return_value=mock_ticker),
            patch("src.logic.stock_finder.notifier.notify", mock_notify),
            patch("src.logic.stock_finder.stats_calculator.calculate"),
            patch("os.remove"),
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
        ):
            from src.logic import stock_finder

            stock_finder.run("TST")

        mock_notify.assert_called_once()

    def test_optimizer_produces_recommendation_string(self) -> None:
        from src.logic.data.data import ProfitabilityData, StockData

        stocks = [
            StockData(
                ticker_symbol="TST",
                stock_name="Test ETF",
                currency="USD",
                current_price=350.0,
                predict_price=380.0,
                two_year_file_name="/tmp/x.png",
                five_year_file_name="/tmp/y.png",
                is_stock_growing=True,
                industry="ETF",
                profitability_data=ProfitabilityData(
                    trailing_eps=2.0,
                    forward_eps=2.5,
                    netIncome_to_common=1e6,
                    ebitda_margins=0.20,
                    operating_margins=0.15,
                ),
                beta=1.0,
                standard_deviation=0.12,
                dividend_yield=0.01,
                top_holdings=np.array([["AAPL", 0.07]]),
                sector_allocation={"technology": 0.60},
                average_daily_volume=1_000_000.0,
                assets_under_management=5e9,
                expense_ratio=0.0003,
                description="Fixture ETF",
            )
        ]

        mock_response = MagicMock()
        mock_response.content = b"0"

        with patch("requests.get", return_value=mock_response):
            from src.adapter.out.optimization import optimizer_dispatcher

            recommendation = optimizer_dispatcher.optimize(stocks, budget=10_000)

        assert isinstance(recommendation, str)
        assert len(recommendation) > 0
