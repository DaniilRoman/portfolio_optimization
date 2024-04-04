import requests
from src.logic.data.data import StockData
import configuration

def calculate(analyses_result: StockData):
    if analyses_result.is_stock_growing and analyses_result.profitability_data.is_profitable():
        increment_counter_endpoint = f'https://script.google.com/macros/s/{configuration.APP_SCRIPT_ID}/exec?value="{analyses_result.industry}"&sheet="industry"'
    else:
        increment_counter_endpoint = f'https://script.google.com/macros/s/{configuration.APP_SCRIPT_ID}/exec?value="{analyses_result.industry}"&sheet="negative_industry"'
    requests.get(increment_counter_endpoint)