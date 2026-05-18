"""Filters the configured stock universe to a tradable subset by removing delisted or unreachable tickers."""

import requests

import config.stock_names as stock_names
from config.configuration import settings


def __calculate_index(current_counter: int) -> int:
    tmp_counter = current_counter
    len_of_stocks = len(stock_names.etf_list)
    while tmp_counter >= len_of_stocks:
        tmp_counter = tmp_counter - len_of_stocks
    return tmp_counter


def pick() -> str:
    current_counter = int(requests.get(settings.GET_AND_INCREMENT_COUNTER_URL).content)
    stock_index = __calculate_index(current_counter)
    return stock_names.etf_list[stock_index]
