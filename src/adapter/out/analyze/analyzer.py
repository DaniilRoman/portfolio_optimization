import pandas as pd

from src.logic.data.data import StockData, StockInfo, ProfitabilityData

from prophet import Prophet
import matplotlib.pyplot as plt

def analyses(ticker_symbol: str, stock_info: StockInfo, two_year_prophet: Prophet, two_year_predicted_prices: pd.DataFrame, five_year_prophet: Prophet, five_year_predicted_prices: pd.DataFrame) -> StockData:
    current_price = __last_price(stock_info.historic_data, "y")
    two_year_last_predicted_price = __last_price(two_year_predicted_prices, "yhat")
    five_year_last_predicted_price = __last_price(five_year_predicted_prices, "yhat")
    is_stock_growing = __is_stock_growing(current_price, two_year_last_predicted_price, five_year_last_predicted_price, stock_info.historic_data)
    
    two_year_prophet.plot(two_year_predicted_prices)
    two_year_file_name = f'two_year_{ticker_symbol}.png'
    plt.savefig(two_year_file_name)

    five_year_prophet.plot(five_year_predicted_prices)
    five_year_file_name = f'five_year_{ticker_symbol}.png'
    plt.savefig(five_year_file_name)

    # Extract data from FundsData
    funds_data = stock_info.ticker.funds_data
    total_assets = getattr(funds_data, 'total_assets', 0)
    expense_ratio = getattr(funds_data, 'expense_ratio', 0)
    yield_ = getattr(funds_data, 'yield_', 0)

    profitability_data = ProfitabilityData(
        ebitda_margins=stock_info.ticker.info.get('ebitdaMargins', 0),
        forward_eps=stock_info.ticker.info.get('forwardEps', 0),
        netIncome_to_common=stock_info.ticker.info.get('netIncomeToCommon', 0),
        operating_margins=stock_info.ticker.info.get('operatingMargins', 0),
        trailing_eps=stock_info.ticker.info.get('trailingEps', 0)
    )

    description = f"{funds_data.fund_overview.get('family', '')} || {funds_data.fund_overview.get('legalType', '')} || {funds_data.description}"
    return StockData(
        ticker_symbol=ticker_symbol, 
        stock_name=stock_info.ticker.info.get('longName', 'Unknown'),
        currency=stock_info.ticker.info.get('currency', 'Unknown'),
        current_price=current_price, 
        predict_price=two_year_last_predicted_price,
        two_year_file_name=two_year_file_name,
        five_year_file_name=five_year_file_name,
        is_stock_growing=is_stock_growing,
        industry=stock_info.ticker.info.get('industry', ""),
        profitability_data=profitability_data,
        average_daily_volume=stock_info.ticker.info.get('averageVolume', 0),
        description=description,
        top_holdings=funds_data.top_holdings.values,
        sector_allocation=funds_data.sector_weightings,
        # New unverified fields
        beta=stock_info.ticker.info.get('beta', 0),
        standard_deviation=stock_info.ticker.info.get('standardDeviation', 0),
        dividend_yield=yield_,
        assets_under_management=total_assets,
        expense_ratio=expense_ratio
    )


def __is_stock_growing(current_price: float, two_year_last_predicted_price: float, five_year_last_predicted_price: float, historic_data: pd.DataFrame) -> bool:
    month_2_years_ago = __slice(historic_data, 365 * 2, 30)
    month_5_years_ago = __slice(historic_data, 365 * 5, 30)
    return current_price <= two_year_last_predicted_price \
        and current_price <= five_year_last_predicted_price \
        and __is_stock_historicly_growing(current_price, month_2_years_ago) \
        and __is_stock_historicly_growing(current_price, month_5_years_ago)

def __is_stock_historicly_growing(current_price: float, history_slice: pd.DataFrame) -> bool:
    return any(val < current_price for val in history_slice["y"])

def __last_price(one_stock_data, column: str) -> float:
    return round(one_stock_data.tail(1)[column].iloc[0], 2) # TODO make rounding based on value

def __slice(historic_data: pd.DataFrame, prev_date: int, slice: int) -> pd.DataFrame:
    today = historic_data.iloc[-1].name  # Get the latest date in the data
    old_data_start = today - pd.Timedelta(days=prev_date)  # Calculate date 2 years ago
    old_data_end = today - pd.Timedelta(days=prev_date-slice)  # Calculate date 2 years ago
    slice_data = historic_data.loc[old_data_start:old_data_end]
    return slice_data
