import yfinance as yf
from pandas import DataFrame
import pandas as pd

from stock_predict import get_future_value


def get_stock_data(stock_name: str, history_period="1y") -> DataFrame:
    stock = yf.Ticker(stock_name)

    hist = stock.history(period=history_period)
    data = hist["Close"].to_frame("y")
    data["ds"] = data.index.date
    return data


def get_prices(stock_data, predict_period):
    current_price = stock_data.tail(1)["y"].iloc[0]
    future_price = get_future_value(stock_data, predict_period)
    return current_price, future_price


def construct_data_for_optimization(stock_names, max_stock_count_list, predict_period):
    list_for_df = []
    for stock_name, max_stock_count in zip(stock_names, max_stock_count_list):
        stock_data = get_stock_data(stock_name)
        current_price, future_price = get_prices(stock_data, predict_period)

        if max_stock_count == 0:
            max_stock_count = 100
        list_for_df.append([stock_name, current_price, future_price, max_stock_count])

    result = pd.DataFrame.from_records(list_for_df)
    result.columns = ['Name', 'Value', 'Price', 'MaxCount']
    return result
