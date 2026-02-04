import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class ProfitabilityData:
    trailing_eps: float
    forward_eps: float
    netIncome_to_common: float
    ebitda_margins: float
    operating_margins: float

    def is_profitable(self) -> bool:
        # Check if the company is profitable based on key metrics
        # A company is considered profitable if it has positive earnings and margins
        # We check trailing EPS and forward EPS for earnings, and margins for profitability
        has_positive_earnings = self.trailing_eps > 0 or self.forward_eps > 0
        has_positive_net_income = self.netIncome_to_common > 0
        has_positive_margins = self.ebitda_margins > 0 or self.operating_margins > 0
        
        # Company is profitable if it has positive earnings OR positive net income AND at least one positive margin
        return (has_positive_earnings or has_positive_net_income) and has_positive_margins

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
    beta: float
    standard_deviation: float
    dividend_yield: float
    top_holdings: np.ndarray
    sector_allocation: dict
    average_daily_volume: float
    assets_under_management: float
    expense_ratio: float
    description: str
    prediction_uncertainty: float = 0.0  # New: uncertainty range from Prophet forecast


@dataclass
class StockInfo:
    historic_data: pd.DataFrame
    ticker: any

class SkipException(Exception):
    pass
