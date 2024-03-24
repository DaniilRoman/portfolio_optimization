import yfinance as yf
from pandas import DataFrame
import json
import urllib

from adapter.out.predict import predicter
from utils.utils import get_current_date_str, get_next_day


def __search_stocks(stock_name_query):
    response = urllib.request.urlopen(f'https://query2.finance.yahoo.com/v1/finance/search?q={stock_name_query}')
    content = response.read()
    return [i['symbol'] for i in json.loads(content.decode('utf8'))['quotes']]

def download_stock_data(stock_name: str, start_date: str = get_current_date_str(), end_date=get_next_day(1)) -> DataFrame:
    stock = yf.Ticker(stock_name)

    hist = stock.history(start=start_date, end=end_date)
    data = hist["Close"].to_frame("y")
    data["ds"] = data.index.date
    return data
