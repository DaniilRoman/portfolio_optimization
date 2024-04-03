import logging
import os
import pandas as pd

from src.adapter.out.stock_pick import stock_picker
from src.adapter.out.download import downloader
from src.adapter.out.predict import predicter
from src.adapter.out.notify import notifier
from src.adapter.out.analyze import analyzer
from src.infrastructure.utils import utils
from src.logic.data.data import StockData


def __clean_artifacts(analyses_result: StockData):
    os.remove(analyses_result.two_year_file_name)
    os.remove(analyses_result.five_year_file_name)

def __slice(historic_data, prev_date) -> any:
    today = historic_data.iloc[-1].name  # Get the latest date in the data
    two_years_ago = today - pd.Timedelta(days=prev_date)  # Calculate date 2 years ago
    data_2y = historic_data.loc[two_years_ago:today]
    return data_2y

def run(stock_name = None):
    if stock_name is None:
        stock_name = stock_picker.pick()
    logging.info(f"Started an analyses of `{stock_name}`")

    stock_info = downloader.download_stock_data(stock_name, start_date=utils.prev_day(365*5))
    two_year_data = __slice(stock_info.historic_data, 365 * 2)
    five_year_data = stock_info.historic_data
    five_year_prophet, five_year_predicted_prices = predicter.predict(five_year_data, predict_period=90)
    two_year_prophet, two_year_predicted_prices = predicter.predict(two_year_data, predict_period=90)

    analyses_result = analyzer.analyses(stock_name, 
                                        stock_info, 
                                        two_year_prophet=two_year_prophet, 
                                        two_year_predicted_prices=two_year_predicted_prices, 
                                        five_year_prophet=five_year_prophet, 
                                        five_year_predicted_prices=five_year_predicted_prices)
    notifier.notify(analyses_result)
    __clean_artifacts(analyses_result)
