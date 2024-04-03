import yfinance as yf
import json
import urllib

from src.infrastructure.utils import utils
from src.logic.data.data import StockInfo
from src.logic.data import data


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
    if hist.empty:
        raise data.SkipException(f'History data is empty for a stock: {stock_name}')
    
    historic_data = hist["Close"].to_frame("y")
    historic_data["ds"] = historic_data.index.date
    return StockInfo(
        historic_data=historic_data, 
        ticker=stock)
