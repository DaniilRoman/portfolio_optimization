import requests
import config.configuration as configuration
import config.stock_names as stock_names

def __calculate_index(current_counter: int) -> int:
    tmp_counter = current_counter
    len_of_stocks = len(stock_names.etf_list)
    while tmp_counter >= len_of_stocks:
        tmp_counter = tmp_counter - len_of_stocks
    return tmp_counter

def pick():
    url = configuration.GET_AND_INCREMENT_COUNTER_URL + "?window=1"
    current_counter = int(requests.get(configuration.GET_AND_INCREMENT_COUNTER_URL).content)
    stock_index = __calculate_index(current_counter)
    return stock_names.etf_list[stock_index]