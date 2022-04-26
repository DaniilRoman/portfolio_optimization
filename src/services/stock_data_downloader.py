from typing import List

import yfinance as yf
from pandas import DataFrame
import json
import urllib

from data import PriceData
from optimization_job_repo import OptimizationRepository
from src.services.stock_predict import get_predict_value
from utils import get_current_date_str


def download_stock_data(stock_name: str, history_period="1y") -> DataFrame:
    stock = yf.Ticker(stock_name)

    hist = stock.history(period=history_period)
    data = hist["Close"].to_frame("y")
    data["ds"] = data.index.date
    return data


def get_current_price(one_stock_data):
    return round(one_stock_data.tail(1)["y"].iloc[0], 2)


def download_current_price(stock_name: str, repo: OptimizationRepository, date: str = get_current_date_str()):
    exist_price = repo.get_saved_stock_price(date, stock_name)
    if exist_price is not None:
        return exist_price

    data = download_stock_data(stock_name, "1d")
    res = get_current_price(data)
    repo.save_stock_price(date, stock_name, res)
    print(f"Downloaded: {stock_name}")
    return res


def download_current_prices(stock_names: List[str], repo: OptimizationRepository):
    return [download_current_price(stock_name, repo) for stock_name in stock_names]


def search_stocks(stock_name_query):
    response = urllib.request.urlopen(f'https://query2.finance.yahoo.com/v1/finance/search?q={stock_name_query}')
    content = response.read()
    return [i['symbol'] for i in json.loads(content.decode('utf8'))['quotes']]


def get_current_and_predict_prices(stock_name: str, predict_period: int, repo: OptimizationRepository,
                                   is_backtest: bool = False):
    stock_data = download_stock_data(stock_name)

    if is_backtest:
        real_future_price = stock_data.tail(1)["y"].iloc[0]
        stock_data = stock_data[:-predict_period]
    else:
        real_future_price = 0

    current_price = get_current_price(stock_data)
    predict_price = get_predict_value(stock_name, stock_data, repo, predict_period)
    return PriceData(current_price, predict_price, real_future_price)
