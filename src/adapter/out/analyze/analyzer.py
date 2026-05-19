"""Builds an AnalysisReport aggregate from StockInfo + Forecast; generates two/five-year price-history chart images."""

import math

import matplotlib.pyplot as plt
import pandas as pd

from src.domain.data.data import AnalysisReport, Asset, ForecastSummary, MarketSnapshot, ProfitabilityData, StockInfo
from src.domain.data.forecast import Forecast

_HORIZON_YEARS = 3.5  # geometric midpoint between 2-year and 5-year forecast horizons


def analyses(
    ticker_symbol: str,
    stock_info: StockInfo,
    two_year_forecast: Forecast,
    five_year_forecast: Forecast,
) -> AnalysisReport:
    meta = stock_info.ticker
    current_price = __last_price(stock_info.historic_data, "y")

    two_year_point = two_year_forecast.final_price()
    five_year_point = five_year_forecast.final_price()
    mean_forecast = (two_year_point + five_year_point) / 2
    # Use the unbiased mean point estimate instead of the lower uncertainty band
    predict_price = mean_forecast

    forecast_uncertainty = two_year_forecast.uncertainty_band()
    forecast_volatility = two_year_forecast.volatility()

    # Annualised log-return view for Black-Litterman (§4.3)
    if current_price > 0 and mean_forecast > 0:
        expected_log_return = math.log(mean_forecast / current_price) / _HORIZON_YEARS
    else:
        expected_log_return = 0.0

    is_stock_growing = __is_stock_growing(current_price, two_year_point, five_year_point, stock_info.historic_data)

    two_year_forecast.plot(two_year_forecast.series)
    two_year_file_name = f"two_year_{ticker_symbol}.png"
    plt.savefig(two_year_file_name)

    five_year_forecast.plot(five_year_forecast.series)
    five_year_file_name = f"five_year_{ticker_symbol}.png"
    plt.savefig(five_year_file_name)

    standard_deviation = __calc_std(stock_info.historic_data)
    momentum_12_1 = __calc_momentum_12_1(stock_info.historic_data)

    profitability_data = ProfitabilityData(
        ebitda_margins=meta.ebitda_margins,
        forward_eps=meta.forward_eps,
        netIncome_to_common=meta.net_income_to_common,
        operating_margins=meta.operating_margins,
        trailing_eps=meta.trailing_eps,
    )

    return AnalysisReport(
        asset=Asset(
            ticker_symbol=ticker_symbol,
            stock_name=meta.long_name,
            currency=meta.currency,
            industry=meta.industry,
            description=meta.description,
            expense_ratio=meta.expense_ratio,
            assets_under_management=meta.total_assets,
        ),
        market=MarketSnapshot(
            current_price=current_price,
            beta=meta.beta,
            standard_deviation=standard_deviation,
            dividend_yield=meta.dividend_yield,
            average_daily_volume=meta.average_volume,
        ),
        forecast=ForecastSummary(
            predict_price=predict_price,
            prediction_uncertainty=forecast_uncertainty,
            forecast_volatility=forecast_volatility,
            two_year_file_name=two_year_file_name,
            five_year_file_name=five_year_file_name,
            expected_log_return=expected_log_return,
        ),
        top_holdings=meta.top_holdings,
        sector_allocation=meta.sector_weights,
        is_stock_growing=is_stock_growing,
        profitability_data=profitability_data,
        momentum_12_1=momentum_12_1,
    )


def __calc_momentum_12_1(historic_data: pd.DataFrame) -> float:
    """12-1 month price return: return from 12 months ago to 1 month ago (skip recent month to avoid reversal)."""
    prices = historic_data["y"]
    if len(prices) < 253:
        return 0.0
    price_12m_ago = float(prices.iloc[-253])  # ~252 trading days ≈ 12 months
    price_1m_ago = float(prices.iloc[-22])  # skip last 21 trading days ≈ 1 month
    return (price_1m_ago - price_12m_ago) / price_12m_ago if price_12m_ago > 0 else 0.0


def __calc_std(historic_data: pd.DataFrame) -> float:
    if historic_data.empty:
        return 0.0
    returns = historic_data["y"].pct_change().dropna()
    if returns.empty:
        return 0.0
    return float(returns.std() * (252**0.5))


def __is_stock_growing(
    current_price: float,
    two_year_last_predicted_price: float,
    five_year_last_predicted_price: float,
    historic_data: pd.DataFrame,
) -> bool:
    month_2_years_ago = __slice(historic_data, 365 * 2, 30)
    month_5_years_ago = __slice(historic_data, 365 * 5, 30)
    percent_change = ((two_year_last_predicted_price - current_price) / current_price) * 100
    growth_in_5_percent_archived = percent_change > 9.9
    return (
        current_price <= two_year_last_predicted_price
        and current_price <= five_year_last_predicted_price
        and __is_stock_historicly_growing(current_price, month_2_years_ago)
        and __is_stock_historicly_growing(current_price, month_5_years_ago)
        and growth_in_5_percent_archived
    )


def __is_stock_historicly_growing(current_price: float, history_slice: pd.DataFrame) -> bool:
    return any(val < current_price for val in history_slice["y"])


def __last_price(one_stock_data: pd.DataFrame, column: str) -> float:
    price = float(one_stock_data.tail(1)[column].iloc[0])
    if price < 1:
        return round(price, 4)
    elif price < 10:
        return round(price, 3)
    else:
        return round(price, 2)


def __slice(historic_data: pd.DataFrame, prev_date: int, window: int) -> pd.DataFrame:
    today = historic_data.iloc[-1].name
    old_data_start = today - pd.Timedelta(days=prev_date)
    old_data_end = today - pd.Timedelta(days=prev_date - window)
    return historic_data.loc[old_data_start:old_data_end]
