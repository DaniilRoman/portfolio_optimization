import pandas as pd
from dataclasses import dataclass

@dataclass
class ProfitabilityData:
    trailing_eps: float
    forward_eps: float
    netIncome_to_common: float
    ebitda_margins: float
    operating_margins: float

    def is_profitable(self) -> bool:
        # TODO tmp fix (remove a zero check from condition)
        return self.trailing_eps >= 0 and self.forward_eps >= 0 and self.netIncome_to_common >= 0 and self.ebitda_margins >= 0 and self.operating_margins >= 0

@dataclass
class StockData:
    ticker_symbol: str
    stock_name: str
    currency: str
    current_price: float
    predict_price: float
    two_year_file_name: str
    five_year_file_name: str
    is_stock_growing: bool
    industry: str
    profitability_data: ProfitabilityData


@dataclass
class StockInfo:
    historic_data: pd.DataFrame
    ticker: any

class SkipException(Exception):
    pass