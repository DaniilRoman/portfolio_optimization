import yfinance as yf
from pandas import DataFrame
import json
import urllib

from data import PriceData
from optimization_job_repo import OptimizationRepository
from services.stock_predict import predict_value
from utils import get_current_date_str, get_prev_day, get_next_day, get_current_date, round_precise


def download_stock_data(stock_name: str, start_date: str = get_current_date_str(), end_date=get_next_day(1)) -> DataFrame:
    stock = yf.Ticker(stock_name)

    hist = stock.history(start=start_date, end=end_date)
    data = hist["Close"].to_frame("y")
    data["ds"] = data.index.date
    return data


def get_current_price(one_stock_data):
    return round(one_stock_data.tail(1)["y"].iloc[0], round_precise)


def download_current_price(stock_name: str, repo: OptimizationRepository, date: str = get_current_date_str()):
    exist_price = repo.get_saved_stock_price(date, stock_name)
    if exist_price is not None:
        return exist_price

    data = download_stock_data(stock_name, start_date=date, end_date=get_next_day(30, date))[:1]
    res = get_current_price(data)

    repo.save_stock_price(date, stock_name, res)
    print(f"Downloaded: {stock_name}")
    return res


def search_stocks(stock_name_query):
    response = urllib.request.urlopen(f'https://query2.finance.yahoo.com/v1/finance/search?q={stock_name_query}')
    content = response.read()
    return [i['symbol'] for i in json.loads(content.decode('utf8'))['quotes']]


def get_predict_value(stock_name: str, repo: OptimizationRepository,
                      predict_period: int = 30, current_date: str = get_current_date()) -> float:
    exist_price = repo.get_saved_predicted_stock_price(current_date, stock_name)
    if exist_price is not None:
        return exist_price

    data = download_stock_data(stock_name, start_date=get_prev_day(365*3, current_date), end_date=current_date)
    res = predict_value(data, predict_period)

    repo.save_stock_predicted_price(current_date, stock_name, res)
    print(f"Predicted: {stock_name}")
    return res


def construct_price_data(stock_name: str, predict_period: int, repo: OptimizationRepository,
                         is_backtest: bool = False, current_date: str = get_current_date_str()) -> PriceData:
    if is_backtest:
        future_date = current_date
        current_date = get_prev_day(predict_period)

        real_future_price = download_current_price(stock_name, repo, future_date)
        current_price = download_current_price(stock_name, repo, current_date)
        predicted_price = get_predict_value(stock_name, repo, predict_period, current_date)
    else:
        current_price = download_current_price(stock_name, repo, current_date)
        predicted_price = get_predict_value(stock_name, repo, predict_period, current_date)
        real_future_price = 0

    return PriceData(stock_name, current_price, predicted_price, real_future_price)
