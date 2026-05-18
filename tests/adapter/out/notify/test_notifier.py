"""Unit tests for the notifier adapter."""

from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from src.adapter.out.notify import notifier
from src.logic.data.data import AnalysisReport, OptimizationResult, Portfolio, RiskMetrics
from tests.factories import make_allocation, make_stock_data


def _make_stock_data(**overrides: object) -> AnalysisReport:
    kwargs: dict = {
        "ticker_symbol": "VOO",
        "stock_name": "Vanguard S&P 500 ETF",
        "currency": "USD",
        "current_price": 400.0,
        "predict_price": 450.0,
        "two_year_file_name": "/tmp/voo_2y.png",
        "five_year_file_name": "/tmp/voo_5y.png",
        "is_stock_growing": True,
        "dividend_yield": 0.013,
        "top_holdings": np.array([["AAPL", 0.07], ["MSFT", 0.06]]),
        "sector_allocation": {"technology": 0.30, "healthcare": 0.20},
        "average_daily_volume": 5_000_000.0,
        "assets_under_management": 800_000_000_000.0,
        "expense_ratio": 0.0003,
        "description": "Tracks the S&P 500 index",
        "forecast_volatility": 0.12,
        "prediction_uncertainty": 15.0,
    }
    kwargs.update(overrides)
    return make_stock_data(**kwargs)


def _make_optimization_result() -> OptimizationResult:
    stock = _make_stock_data()
    alloc = make_allocation(
        asset=stock.asset,
        quantity=2,
        total_cost=800.0,
        net_profit=95.0,
        capital_gain=100.0,
        dividend_income=10.4,
        expense_fee=0.24,
        expense_ratio=0.0003,
        forecast_volatility=stock.forecast.forecast_volatility,
    )
    portfolio = Portfolio(
        allocations=[alloc],
        risk_metrics=RiskMetrics(volatility=0.12, sector_concentration=0.3, company_overlap=0.1),
    )
    return OptimizationResult(risk_aware=portfolio, profit_only=portfolio)


class TestSendTextMessage:
    def test_calls_bot_with_correct_args(self) -> None:
        mock_bot = MagicMock()
        with patch("telepot.Bot", return_value=mock_bot):
            notifier.send_text_message("hello world")

        mock_bot.getMe.assert_called_once()
        mock_bot.sendMessage.assert_called_once_with(chat_id=mock_bot.sendMessage.call_args[1]["chat_id"], text="hello world")

    def test_passes_correct_text(self) -> None:
        mock_bot = MagicMock()
        with patch("telepot.Bot", return_value=mock_bot):
            notifier.send_text_message("test message")

        _, kwargs = mock_bot.sendMessage.call_args
        assert kwargs["text"] == "test message"


class TestNotify:
    def test_sends_media_group_with_two_photos(self) -> None:
        result = _make_stock_data()
        mock_bot = MagicMock()
        with (
            patch("telepot.Bot", return_value=mock_bot),
            patch("builtins.open", mock_open(read_data=b"img")),
        ):
            notifier.notify(result)

        mock_bot.sendMediaGroup.assert_called_once()
        call_kwargs = mock_bot.sendMediaGroup.call_args[1]
        assert len(call_kwargs["media"]) == 2

    def test_media_group_has_caption_on_first_photo(self) -> None:
        result = _make_stock_data()
        mock_bot = MagicMock()
        with (
            patch("telepot.Bot", return_value=mock_bot),
            patch("builtins.open", mock_open(read_data=b"img")),
        ):
            notifier.notify(result)

        media = mock_bot.sendMediaGroup.call_args[1]["media"]
        assert "caption" in media[0]
        assert "caption" not in media[1]

    def test_notify_always_sends_regardless_of_is_stock_growing(self) -> None:
        # __is_notifyable currently always returns True — verify current behaviour.
        result = _make_stock_data(is_stock_growing=False)
        mock_bot = MagicMock()
        with (
            patch("telepot.Bot", return_value=mock_bot),
            patch("builtins.open", mock_open(read_data=b"img")),
        ):
            notifier.notify(result)

        mock_bot.sendMediaGroup.assert_called_once()


class TestMarkdownEscaping:
    def _get_caption(self, result: AnalysisReport) -> str:
        mock_bot = MagicMock()
        with (
            patch("telepot.Bot", return_value=mock_bot),
            patch("builtins.open", mock_open(read_data=b"img")),
        ):
            notifier.notify(result)
        media = mock_bot.sendMediaGroup.call_args[1]["media"]
        return str(media[0]["caption"])

    def test_asterisk_in_sector_is_escaped(self) -> None:
        result = _make_stock_data(sector_allocation={"tech*sector": 0.5})
        caption = self._get_caption(result)
        assert r"\*" in caption

    def test_underscore_in_sector_is_escaped(self) -> None:
        result = _make_stock_data(sector_allocation={"tech_sector": 0.5})
        caption = self._get_caption(result)
        assert r"\_" in caption

    def test_backtick_in_holdings_is_escaped(self) -> None:
        result = _make_stock_data(top_holdings=np.array([["TICK`ER", 0.05]]))
        caption = self._get_caption(result)
        assert r"\`" in caption


class TestSendOptimizationResult:
    def test_sends_text_message(self) -> None:
        result = _make_optimization_result()
        mock_bot = MagicMock()
        with patch("telepot.Bot", return_value=mock_bot):
            notifier.send_optimization_result(result)
        mock_bot.sendMessage.assert_called_once()

    def test_message_contains_portfolio_sections(self) -> None:
        result = _make_optimization_result()
        mock_bot = MagicMock()
        with patch("telepot.Bot", return_value=mock_bot):
            notifier.send_optimization_result(result)
        _, kwargs = mock_bot.sendMessage.call_args
        text = kwargs["text"]
        assert "Risk-Aware Optimization" in text
        assert "Profit-Only Optimization" in text
        assert "VOO" in text

    def test_empty_portfolio_shows_no_etfs_message(self) -> None:
        empty_portfolio = Portfolio(
            allocations=[],
            risk_metrics=RiskMetrics(volatility=0.0, sector_concentration=0.0, company_overlap=0.0),
        )
        result = OptimizationResult(risk_aware=empty_portfolio, profit_only=empty_portfolio)
        mock_bot = MagicMock()
        with patch("telepot.Bot", return_value=mock_bot):
            notifier.send_optimization_result(result)
        _, kwargs = mock_bot.sendMessage.call_args
        assert "No ETFs" in kwargs["text"]
