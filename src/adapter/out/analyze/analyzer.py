import pandas as pd

from src.logic.data.data import StockData, StockInfo

from prophet import Prophet
import matplotlib.pyplot as plt

def analyses(ticker_symbol: str, prophet: Prophet, stock_info: StockInfo, predicted_prices: pd.DataFrame) -> StockData:
    current_price = __last_price(stock_info.historic_data, "y")
    last_predicted_price = __last_price(predicted_prices, "yhat")
    is_stock_growing = __is_stock_growing(current_price, last_predicted_price)
    prophet.plot(predicted_prices)
    file_name = f'{ticker_symbol}.png'
    plt.savefig(file_name)
    return StockData(
        ticker_symbol=ticker_symbol, 
        stock_name=stock_info.ticker.info['shortName'],
        currency=stock_info.ticker.basic_info['currency'],
        current_price=current_price, 
        predict_price=last_predicted_price,
        file_name=file_name,
        is_stock_growing=is_stock_growing)
    
def __is_stock_growing(current_price, last_predicted_price) -> bool:
    return current_price <= last_predicted_price

def __last_price(one_stock_data, column: str) -> float:
    return round(one_stock_data.tail(1)[column].iloc[0], 2) # TODO make rounding based on value