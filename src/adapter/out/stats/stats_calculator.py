import requests
from src.logic.data.data import StockData
import config.configuration as configuration

def calculate(analyses_result: StockData):
    if analyses_result.is_stock_growing and analyses_result.profitability_data.is_profitable():
        increment_counter_endpoint = f'https://script.google.com/macros/s/{configuration.APP_SCRIPT_ID}/exec?value="{analyses_result.industry}"&sheet="etf_industry_positive"'
        increment_counter_endpoint_company = f'https://script.google.com/macros/s/{configuration.APP_SCRIPT_ID}/exec?value="{analyses_result.stock_name}"&sheet="etf_positive"'
    else:
        increment_counter_endpoint = f'https://script.google.com/macros/s/{configuration.APP_SCRIPT_ID}/exec?value="{analyses_result.industry}"&sheet="etf_industry_negative"'
        increment_counter_endpoint_company = f'https://script.google.com/macros/s/{configuration.APP_SCRIPT_ID}/exec?value="{analyses_result.stock_name}"&sheet="etf_negative"'
    requests.get(increment_counter_endpoint)
    requests.get(increment_counter_endpoint_company)
