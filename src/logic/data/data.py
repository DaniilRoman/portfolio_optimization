import pandas as pd
from dataclasses import dataclass

@dataclass
class StockData:
    ticker_symbol: str
    stock_name: str
    currency: str
    current_price: float
    predict_price: float
    file_name: str

@dataclass
class StockInfo:
    historic_data: pd.DataFrame
    ticker: any
