from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ProfitabilityData:
    trailing_eps: float
    forward_eps: float
    netIncome_to_common: float
    ebitda_margins: float
    operating_margins: float

    def is_profitable(self) -> bool:
        has_positive_earnings = self.trailing_eps > 0 or self.forward_eps > 0
        has_positive_net_income = self.netIncome_to_common > 0
        has_positive_margins = self.ebitda_margins > 0 or self.operating_margins > 0
        return (has_positive_earnings or has_positive_net_income) and has_positive_margins


@dataclass
class TickerMetadata:
    long_name: str
    currency: str
    industry: str
    beta: float
    dividend_yield: float
    total_assets: float
    expense_ratio: float
    average_volume: float
    trailing_eps: float
    forward_eps: float
    net_income_to_common: float
    ebitda_margins: float
    operating_margins: float
    top_holdings: np.ndarray
    sector_weights: dict[str, float]
    description: str


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
    sector_allocation: dict[str, float]
    average_daily_volume: float
    assets_under_management: float
    expense_ratio: float
    description: str
    forecast_volatility: float = 0.0
    prediction_uncertainty: float = 0.0


@dataclass
class StockInfo:
    historic_data: pd.DataFrame
    ticker: TickerMetadata


class SkipException(Exception):
    pass
