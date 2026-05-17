"""Unit tests for stock_finder orchestration."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd

from src.logic.data.data import ProfitabilityData, StockData, StockInfo, TickerMetadata
from src.logic.data.forecast import Forecast


def _make_ticker_meta() -> TickerMetadata:
    return TickerMetadata(
        long_name="Test ETF",
        currency="USD",
        industry="ETF",
        beta=1.0,
        dividend_yield=0.01,
        total_assets=1e9,
        expense_ratio=0.0003,
        average_volume=1_000_000.0,
        trailing_eps=2.0,
        forward_eps=2.5,
        net_income_to_common=0.0,
        ebitda_margins=0.0,
        operating_margins=0.0,
        top_holdings=np.array([]),
        sector_weights={},
        description="",
    )


def _make_historic_df(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    prices = np.linspace(100.0, 130.0, n)
    df = pd.DataFrame({"y": prices}, index=dates)
    df["ds"] = pd.to_datetime(df.index.date)
    return df


def _make_stock_data() -> StockData:
    return StockData(
        ticker_symbol="TST",
        stock_name="Test ETF",
        currency="USD",
        current_price=130.0,
        predict_price=145.0,
        two_year_file_name="/tmp/tst_2y.png",
        five_year_file_name="/tmp/tst_5y.png",
        is_stock_growing=True,
        industry="ETF",
        profitability_data=ProfitabilityData(
            trailing_eps=2.0,
            forward_eps=2.5,
            netIncome_to_common=0.0,
            ebitda_margins=0.0,
            operating_margins=0.0,
        ),
        beta=1.0,
        standard_deviation=0.1,
        dividend_yield=0.01,
        top_holdings=np.array([]),
        sector_allocation={},
        average_daily_volume=1_000_000.0,
        assets_under_management=1e9,
        expense_ratio=0.0003,
        description="",
    )


def _make_forecast() -> Forecast:
    n = 10
    df = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=n),
            "y": np.linspace(100.0, 110.0, n),
            "yhat": np.linspace(100.0, 112.0, n),
            "yhat_lower": np.linspace(95.0, 106.0, n),
            "yhat_upper": np.linspace(105.0, 118.0, n),
            "uncertainty_range": np.full(n, 10.0),
            "volatility_forecast": np.full(n, 0.15),
        }
    )
    return Forecast(series=df, model=MagicMock())


class TestRunCallSequence:
    def _run_with_patches(
        self,
        stock_name: str = "TST",
        *,
        stock_info: StockInfo | None = None,
        forecast: Forecast | None = None,
        analyses_result: StockData | None = None,
    ) -> tuple[StockData | None, dict[str, MagicMock]]:
        if stock_info is None:
            stock_info = StockInfo(historic_data=_make_historic_df(800), ticker=_make_ticker_meta())
        if forecast is None:
            forecast = _make_forecast()
        if analyses_result is None:
            analyses_result = _make_stock_data()

        mocks: dict[str, MagicMock] = {}
        with (
            patch("src.logic.stock_finder.downloader.download_stock_data", return_value=stock_info) as m_dl,
            patch("src.logic.stock_finder.predicter.predict", return_value=forecast) as m_pred,
            patch("src.logic.stock_finder.analyzer.analyses", return_value=analyses_result) as m_an,
            patch("src.logic.stock_finder.notifier.notify") as m_notify,
            patch("src.logic.stock_finder.stats_calculator.calculate") as m_stats,
            patch("os.remove") as m_rm,
        ):
            mocks["download"] = m_dl
            mocks["predict"] = m_pred
            mocks["analyze"] = m_an
            mocks["notify"] = m_notify
            mocks["stats"] = m_stats
            mocks["remove"] = m_rm
            from src.logic import stock_finder

            result = stock_finder.run(stock_name)

        return result, mocks

    def test_returns_stock_data(self) -> None:
        result, _ = self._run_with_patches()
        assert isinstance(result, StockData)

    def test_downloader_called_with_stock_name(self) -> None:
        _, mocks = self._run_with_patches("VOO")
        mocks["download"].assert_called_once()
        args = mocks["download"].call_args
        assert args[0][0] == "VOO" or args[1].get("stock_name") == "VOO"

    def test_predicter_called_twice(self) -> None:
        _, mocks = self._run_with_patches()
        assert mocks["predict"].call_count == 2

    def test_analyzer_called_once(self) -> None:
        _, mocks = self._run_with_patches()
        mocks["analyze"].assert_called_once()

    def test_notifier_called_once(self) -> None:
        _, mocks = self._run_with_patches()
        mocks["notify"].assert_called_once()

    def test_stats_calculator_called_once(self) -> None:
        _, mocks = self._run_with_patches()
        mocks["stats"].assert_called_once()

    def test_clean_artifacts_removes_both_files(self) -> None:
        result, mocks = self._run_with_patches()
        assert result is not None
        mocks["remove"].assert_has_calls(
            [call(result.two_year_file_name), call(result.five_year_file_name)],
            any_order=True,
        )

    def test_picks_stock_name_when_none_given(self) -> None:
        with (
            patch("src.logic.stock_finder.stock_picker.pick", return_value="SPY") as m_pick,
            patch("src.logic.stock_finder.downloader.download_stock_data", return_value=StockInfo(historic_data=_make_historic_df(800), ticker=_make_ticker_meta())),
            patch("src.logic.stock_finder.predicter.predict", return_value=_make_forecast()),
            patch("src.logic.stock_finder.analyzer.analyses", return_value=_make_stock_data()),
            patch("src.logic.stock_finder.notifier.notify"),
            patch("src.logic.stock_finder.stats_calculator.calculate"),
            patch("os.remove"),
        ):
            from src.logic import stock_finder

            stock_finder.run(None)
            m_pick.assert_called_once()


class TestSliceWindow:
    def _call_slice(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        from src.logic import stock_finder

        return vars(stock_finder)["__slice"](df, days)  # type: ignore[no-any-return]

    def test_slice_returns_two_year_window(self) -> None:
        df = _make_historic_df(n=800)
        sliced = self._call_slice(df, 365 * 2)
        last_date = df.iloc[-1].name
        earliest = sliced.iloc[0].name
        assert (last_date - earliest).days <= 365 * 2 + 7

    def test_full_df_returned_when_window_exceeds_data(self) -> None:
        df = _make_historic_df(n=10)
        sliced = self._call_slice(df, 365 * 10)
        assert len(sliced) == len(df)
