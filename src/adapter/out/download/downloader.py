import yfinance as yf
from pandas import DataFrame
import json
import urllib

from src.infrastructure.utils import utils
from src.logic.data.data import StockInfo


def __search_stocks(stock_name_query):
    response = urllib.request.urlopen(f'https://query2.finance.yahoo.com/v1/finance/search?q={stock_name_query}')
    content = response.read()
    return [i['symbol'] for i in json.loads(content.decode('utf8'))['quotes']]

def download_stock_data(
        stock_name: str, 
        start_date: str = utils.current_date_str(), 
        end_date=utils.next_day(1)
        ) -> StockInfo:
    stock = yf.Ticker(stock_name)

    hist = stock.history(start=start_date, end=end_date)
    data = hist["Close"].to_frame("y")
    data["ds"] = data.index.date
    return StockInfo(stock, data)
