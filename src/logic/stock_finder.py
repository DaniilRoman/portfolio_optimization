import logging
import pandas as pd
import os

from src.adapter.out.stock_pick import stock_picker
from src.adapter.out.download import downloader
from src.adapter.out.predict import predicter
from src.adapter.out.notify import notifier
from src.infrastructure.utils import utils
from src.logic.data.data import StockData

from prophet import Prophet
import matplotlib.pyplot as plt

def __analyses(stock_name: str, prophet: Prophet, historic_prices: pd.DataFrame, predicted_prices: pd.DataFrame):
    current_price = __last_price(historic_prices, "y")
    last_predicted_price = __last_price(predicted_prices, "yhat")
    # if last_predicted_price <= current_price:
    prophet.plot(predicted_prices)
    file_name = f'{stock_name}.png'
    plt.savefig(file_name)
    return StockData(
        stock_name=stock_name, 
        current_price=current_price, 
        predict_price=last_predicted_price,
        file_name=file_name)
    

def __last_price(one_stock_data, column: str) -> float:
    return round(one_stock_data.tail(1)[column].iloc[0], 2) # TODO make rounding based on value

def __clean_artifacts(analyses_result: StockData):
    os.remove(analyses_result.file_name)

def run():
    stock_name = stock_picker.pick()
    logging.info(f"Started an analyses of `{stock_name}`")

    historic_prices = downloader.download_stock_data(stock_name, start_date=utils.prev_day(365*2))
    prophet, predicted_prices = predicter.predict(historic_prices, predict_period=90)

    analyses_result = __analyses(stock_name, prophet, historic_prices, predicted_prices)
    print(analyses_result)
    if analyses_result != None:
        notifier.notify(analyses_result)
        __clean_artifacts(analyses_result)
