import yfinance as yf
from pandas import DataFrame
from prophet import Prophet

from stock_predict import get_future_value


def get_stock_data(stock_name: str) -> DataFrame:
    stock = yf.Ticker(stock_name)

    hist = stock.history(period="1y")
    data = hist["Close"].to_frame("y")
    data["ds"] = data.index.date
    return data


prophet = Prophet()
data = get_stock_data("MSFT")

current_price = data.tail(1)["y"].iloc[0]
future_price = get_future_value(data, prophet)
print(future_price)
print(current_price)
