from dataclasses import dataclass

@dataclass
class StockData:
    stock_name: str
    current_price: float
    predict_price: float
    file_name: str
