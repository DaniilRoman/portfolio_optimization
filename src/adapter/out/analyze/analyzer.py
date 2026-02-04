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

    # Extract data from FundsData with proper None handling
    funds_data = stock_info.ticker.funds_data
    info = stock_info.ticker.info
    
    # Get total_assets from info dict (not funds_data)
    total_assets = info.get('totalAssets') or info.get('netAssets') or 0
    # Get expense_ratio from info dict (not funds_data)
    expense_ratio = info.get('netExpenseRatio') or info.get('expenseRatio') or 0
    # Get yield from info dict (not funds_data)
    yield_ = info.get('yield') or info.get('dividendYield') or info.get('trailingAnnualDividendYield') or 0
    
    # Get beta - check both 'beta' and 'beta3Year'
    beta = info.get('beta') or info.get('beta3Year') or 0
    
    # Standard deviation doesn't exist in yfinance info, calculate from historical data
    # Calculate standard deviation of daily returns from historic data
    if not stock_info.historic_data.empty:
        # Calculate daily returns
        historic_data = stock_info.historic_data
        returns = historic_data['y'].pct_change().dropna()
        if not returns.empty:
            # Annualize the standard deviation (assuming 252 trading days)
            standard_deviation = returns.std() * (252 ** 0.5)
        else:
            standard_deviation = 0
    else:
        standard_deviation = 0

    profitability_data = ProfitabilityData(
        ebitda_margins=info.get('ebitdaMargins') or 0,
        forward_eps=info.get('forwardEps') or 0,
        netIncome_to_common=info.get('netIncomeToCommon') or 0,
        operating_margins=info.get('operatingMargins') or 0,
        trailing_eps=info.get('trailingEps') or 0
    )

    try:
        description = f"{funds_data.fund_overview.get('family', '')} || {funds_data.fund_overview.get('legalType', '')} || {funds_data.description}"
    except:
        description = ''
    return StockData(
        ticker_symbol=ticker_symbol, 
        stock_name=info.get('longName') or 'Unknown',
        currency=info.get('currency') or 'Unknown',
        current_price=current_price, 
        predict_price=two_year_last_predicted_price,
        two_year_file_name=two_year_file_name,
        five_year_file_name=five_year_file_name,
        is_stock_growing=is_stock_growing,
        industry=info.get('industry') or "",
        profitability_data=profitability_data,
        average_daily_volume=info.get('averageVolume') or 0,
        description=description,
        top_holdings=funds_data.top_holdings.values,
        sector_allocation=funds_data.sector_weightings,
        # New unverified fields
        beta=beta,
        standard_deviation=standard_deviation,
        dividend_yield=yield_,
        assets_under_management=total_assets,
        expense_ratio=expense_ratio
    )


def __is_stock_growing(current_price: float, two_year_last_predicted_price: float, five_year_last_predicted_price: float, historic_data: pd.DataFrame) -> bool:
    month_2_years_ago = __slice(historic_data, 365 * 2, 30)
    month_5_years_ago = __slice(historic_data, 365 * 5, 30)
    percent_change = ((two_year_last_predicted_price - current_price) / (current_price)) * 100
    growth_in_5_percent_archived = percent_change > 9.9
    return current_price <= two_year_last_predicted_price \
        and current_price <= five_year_last_predicted_price \
        and __is_stock_historicly_growing(current_price, month_2_years_ago) \
        and __is_stock_historicly_growing(current_price, month_5_years_ago) \
        and growth_in_5_percent_archived

def __is_stock_historicly_growing(current_price: float, history_slice: pd.DataFrame) -> bool:
    return any(val < current_price for val in history_slice["y"])

def __last_price(one_stock_data, column: str) -> float:
    price = one_stock_data.tail(1)[column].iloc[0]
    # Round based on price value: more decimal places for lower prices
    if price < 1:
        return round(price, 4)
    elif price < 10:
        return round(price, 3)
    elif price < 100:
        return round(price, 2)
    else:
        return round(price, 2)

def __slice(historic_data: pd.DataFrame, prev_date: int, slice: int) -> pd.DataFrame:
    today = historic_data.iloc[-1].name  # Get the latest date in the data
    old_data_start = today - pd.Timedelta(days=prev_date)  # Calculate date 2 years ago
    old_data_end = today - pd.Timedelta(days=prev_date-slice)  # Calculate date 2 years ago
    slice_data = historic_data.loc[old_data_start:old_data_end]
    return slice_data
