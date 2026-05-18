"""Unit tests for stock_finder orchestration."""

from unittest.mock import MagicMock, call, patch

from src.domain.data.data import AnalysisReport, StockInfo
from src.domain.data.forecast import Forecast
from tests.factories import make_forecast, make_historic_df, make_stock_data, make_stock_info


class TestRunCallSequence:
    def _run_with_patches(
        self,
        stock_name: str = "TST",
        *,
        stock_info: StockInfo | None = None,
        forecast: Forecast | None = None,
        analyses_result: AnalysisReport | None = None,
    ) -> tuple[AnalysisReport | None, dict[str, MagicMock]]:
        if stock_info is None:
            stock_info = make_stock_info(historic_data=make_historic_df(800))
        if forecast is None:
            forecast = make_forecast()
        if analyses_result is None:
            analyses_result = make_stock_data()

        mocks: dict[str, MagicMock] = {}
        with (
            patch("src.application.stock_finder.downloader.download_stock_data", return_value=stock_info) as m_dl,
            patch("src.application.stock_finder.predicter.predict", return_value=forecast) as m_pred,
            patch("src.application.stock_finder.analyzer.analyses", return_value=analyses_result) as m_an,
            patch("src.application.stock_finder.notifier.notify") as m_notify,
            patch("src.application.stock_finder.stats_calculator.calculate") as m_stats,
            patch("os.remove") as m_rm,
        ):
            mocks["download"] = m_dl
            mocks["predict"] = m_pred
            mocks["analyze"] = m_an
            mocks["notify"] = m_notify
            mocks["stats"] = m_stats
            mocks["remove"] = m_rm
            from src.application import stock_finder

            result = stock_finder.run(stock_name)

        return result, mocks

    def test_returns_analysis_report(self) -> None:
        result, _ = self._run_with_patches()
        assert isinstance(result, AnalysisReport)

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
            [call(result.forecast.two_year_file_name), call(result.forecast.five_year_file_name)],
            any_order=True,
        )

    def test_picks_stock_name_when_none_given(self) -> None:
        with (
            patch("src.application.stock_finder.stock_picker.pick", return_value="SPY") as m_pick,
            patch(
                "src.application.stock_finder.downloader.download_stock_data",
                return_value=make_stock_info(historic_data=make_historic_df(800)),
            ),
            patch("src.application.stock_finder.predicter.predict", return_value=make_forecast()),
            patch("src.application.stock_finder.analyzer.analyses", return_value=make_stock_data()),
            patch("src.application.stock_finder.notifier.notify"),
            patch("src.application.stock_finder.stats_calculator.calculate"),
            patch("os.remove"),
        ):
            from src.application import stock_finder

            stock_finder.run(None)
            m_pick.assert_called_once()


class TestSliceWindow:
    def _call_slice(self, df, days: int):  # type: ignore[no-untyped-def]
        from src.application import stock_finder

        return vars(stock_finder)["__slice"](df, days)

    def test_slice_returns_two_year_window(self) -> None:
        df = make_historic_df(n=800)
        sliced = self._call_slice(df, 365 * 2)
        last_date = df.iloc[-1].name
        earliest = sliced.iloc[0].name
        assert (last_date - earliest).days <= 365 * 2 + 7

    def test_full_df_returned_when_window_exceeds_data(self) -> None:
        df = make_historic_df(n=10)
        sliced = self._call_slice(df, 365 * 10)
        assert len(sliced) == len(df)
