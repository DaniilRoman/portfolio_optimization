import pandas as pd

from src.logic.data.data import StockData, StockInfo, ProfitabilityData

from prophet import Prophet
import matplotlib.pyplot as plt

def analyses(ticker_symbol: str, stock_info: StockInfo, two_year_prophet: Prophet, two_year_predicted_prices: pd.DataFrame, five_year_prophet: Prophet, five_year_predicted_prices: pd.DataFrame) -> StockData:
    current_price = __last_price(stock_info.historic_data, "y")
    last_predicted_price = __last_price(two_year_predicted_prices, "yhat")
    is_stock_growing = __is_stock_growing(current_price, last_predicted_price)
    
    two_year_prophet.plot(two_year_predicted_prices)
    two_year_file_name = f'two_year_{ticker_symbol}.png'
    plt.savefig(two_year_file_name)

    five_year_prophet.plot(five_year_predicted_prices)
    five_year_file_name = f'five_year_{ticker_symbol}.png'
    plt.savefig(five_year_file_name)

    profitability_data = ProfitabilityData(
        ebitda_margins=stock_info.ticker.info['ebitdaMargins'],
        forward_eps=stock_info.ticker.info['forwardEps'],
        netIncome_to_common=stock_info.ticker.info['netIncomeToCommon'],
        operating_margins=stock_info.ticker.info['operatingMargins'],
        trailing_eps=stock_info.ticker.info['trailingEps']
    )

    return StockData(
        ticker_symbol=ticker_symbol, 
        stock_name=stock_info.ticker.info['shortName'],
        currency=stock_info.ticker.basic_info['currency'],
        current_price=current_price, 
        predict_price=last_predicted_price,
        two_year_file_name=two_year_file_name,
        five_year_file_name=five_year_file_name,
        is_stock_growing=is_stock_growing,
        industry=stock_info.ticker.info['industry'],
        profitability_data=profitability_data
    )
    
def __is_stock_growing(current_price, last_predicted_price) -> bool:
    return current_price <= last_predicted_price

def __last_price(one_stock_data, column: str) -> float:
    return round(one_stock_data.tail(1)[column].iloc[0], 2) # TODO make rounding based on value