"""Builds an AnalysisReport aggregate from StockInfo + Forecast; generates two/five-year price-history chart images."""
import matplotlib.pyplot as plt
import pandas as pd

from src.logic.data.data import AnalysisReport, Asset, ForecastSummary, MarketSnapshot, ProfitabilityData, StockInfo
from src.logic.data.forecast import Forecast


def analyses(
    ticker_symbol: str,
    stock_info: StockInfo,
    two_year_forecast: Forecast,
    five_year_forecast: Forecast,
) -> AnalysisReport:
    meta = stock_info.ticker
    current_price = __last_price(stock_info.historic_data, "y")

    predict_price = min(two_year_forecast.lower_price(), five_year_forecast.lower_price())
    two_year_last_predicted = two_year_forecast.final_price()
    five_year_last_predicted = five_year_forecast.final_price()

    forecast_uncertainty = two_year_forecast.uncertainty_band()
    forecast_volatility = two_year_forecast.volatility()

    is_stock_growing = __is_stock_growing(
        current_price, two_year_last_predicted, five_year_last_predicted, stock_info.historic_data
    )

    two_year_forecast.plot(two_year_forecast.series)
    two_year_file_name = f"two_year_{ticker_symbol}.png"
    plt.savefig(two_year_file_name)

    five_year_forecast.plot(five_year_forecast.series)
    five_year_file_name = f"five_year_{ticker_symbol}.png"
    plt.savefig(five_year_file_name)

    standard_deviation = __calc_std(stock_info.historic_data)

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
        ),
        top_holdings=meta.top_holdings,
        sector_allocation=meta.sector_weights,
        is_stock_growing=is_stock_growing,
        profitability_data=profitability_data,
    )


def __calc_std(historic_data: pd.DataFrame) -> float:
    if historic_data.empty:
        return 0.0
    returns = historic_data["y"].pct_change().dropna()
    if returns.empty:
        return 0.0
    return float(returns.std() * (252**0.5))


def __predicted_price(two_year_predicted_prices: pd.DataFrame, five_year_predicted_prices: pd.DataFrame) -> float:
    two_year_min = __last_price(two_year_predicted_prices, "yhat_lower")
    five_year_min = __last_price(five_year_predicted_prices, "yhat_lower")
    return min(two_year_min, five_year_min)


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
