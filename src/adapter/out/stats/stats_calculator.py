"""Fetches and increments the portfolio-run counter via the external counter URL; outputs a count used by the notifier."""
import requests

from config.configuration import settings
from src.domain.data.data import AnalysisReport


def calculate(analyses_result: AnalysisReport) -> None:
    if analyses_result.is_stock_growing and analyses_result.profitability_data.is_profitable():
        increment_counter_endpoint = f'https://script.google.com/macros/s/{settings.APP_SCRIPT_ID}/exec?value="{analyses_result.asset.industry}"&sheet="etf_industry_positive"'
        increment_counter_endpoint_company = f'https://script.google.com/macros/s/{settings.APP_SCRIPT_ID}/exec?value="{analyses_result.asset.stock_name}"&sheet="etf_positive"'
    else:
        increment_counter_endpoint = f'https://script.google.com/macros/s/{settings.APP_SCRIPT_ID}/exec?value="{analyses_result.asset.industry}"&sheet="etf_industry_negative"'
        increment_counter_endpoint_company = f'https://script.google.com/macros/s/{settings.APP_SCRIPT_ID}/exec?value="{analyses_result.asset.stock_name}"&sheet="etf_negative"'
    requests.get(increment_counter_endpoint)
    requests.get(increment_counter_endpoint_company)
