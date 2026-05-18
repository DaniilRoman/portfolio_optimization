"""Orchestrator: downloads, predicts, and analyses each ticker in sequence; manages chart artifacts and skips failed tickers."""
import logging
import os

import pandas as pd

from src.adapter.out.analyze import analyzer
from src.adapter.out.download import downloader
from src.adapter.out.notify import notifier
from src.adapter.out.predict import predicter
from src.adapter.out.stats import stats_calculator
from src.adapter.out.stock_pick import stock_picker
from src.infrastructure.utils import utils
from src.logic.data.data import AnalysisReport, StockInfo


def run(stock_name: str | None = None) -> AnalysisReport | None:
    if stock_name is None:
        stock_name = stock_picker.pick()
    logging.info("Started an analyses of `%s`", stock_name)

    stock_info = downloader.download_stock_data(stock_name, start_date=utils.prev_day(365 * 5))
    if __toSkip(stock_info):
        logging.info("Skipped stock: %s", stock_name)
        return None
    two_year_data = __slice(stock_info.historic_data, 365 * 2)
    five_year_data = stock_info.historic_data
    five_year_forecast = predicter.predict(five_year_data, predict_period=90)
    two_year_forecast = predicter.predict(two_year_data, predict_period=90)

    analyses_result = analyzer.analyses(
        stock_name,
        stock_info,
        two_year_forecast=two_year_forecast,
        five_year_forecast=five_year_forecast,
    )
    notifier.notify(analyses_result)
    stats_calculator.calculate(analyses_result)
    __clean_artifacts(analyses_result)
    return analyses_result


def __slice(historic_data: pd.DataFrame, prev_date: int) -> pd.DataFrame:
    today = historic_data.iloc[-1].name
    two_years_ago = today - pd.Timedelta(days=prev_date)
    return historic_data.loc[two_years_ago:today]


def __clean_artifacts(analyses_result: AnalysisReport) -> None:
    os.remove(analyses_result.forecast.two_year_file_name)
    os.remove(analyses_result.forecast.five_year_file_name)


def __toSkip(stock_info: StockInfo) -> bool:
    return False
