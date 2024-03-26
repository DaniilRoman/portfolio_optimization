import requests
import configuration
import stock_names

def pick():
    current_counter = int(requests.get(configuration.GET_AND_INCREMENT_COUNTER_URL).content)
    stock_index = len(stock_names.sp500) % current_counter
    return stock_names.sp500[stock_index]