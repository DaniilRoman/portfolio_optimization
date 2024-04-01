import logging
import os

from src.adapter.out.stock_pick import stock_picker
from src.adapter.out.download import downloader
from src.adapter.out.predict import predicter
from src.adapter.out.notify import notifier
from src.adapter.out.analyze import analyzer
from src.infrastructure.utils import utils
from src.logic.data.data import StockData


def __clean_artifacts(analyses_result: StockData):
    os.remove(analyses_result.file_name)

def run(stock_name = None):
    if stock_name is None:
        stock_name = stock_picker.pick()
    logging.info(f"Started an analyses of `{stock_name}`")

    stock_info = downloader.download_stock_data(stock_name, start_date=utils.prev_day(365*2))
    prophet, predicted_prices = predicter.predict(stock_info.historic_data, predict_period=90)

    analyses_result = analyzer.analyses(stock_name, prophet, stock_info, predicted_prices)
    notifier.notify(analyses_result)
    __clean_artifacts(analyses_result)
