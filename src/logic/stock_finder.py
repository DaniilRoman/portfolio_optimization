from src.adapter.out.download import downloader
from src.adapter.out.predict import predicter
from src.adapter.out.notify import notifier
from src.infrastructure.utils import utils

def __analyses(historic_prices, predicted_prices):
    return

def __last_price(one_stock_data):
    return round(one_stock_data.tail(1)["y"].iloc[0], utils.round_precise)

def run(stock_name: str):
    print(f"Job `{stock_name}` created")

    historic_prices = downloader.download_stock_data(stock_name, start_date=utils.prev_day(365))
    predicted_prices = predicter.predict(historic_prices, predict_period=30)
    analyses_result = __analyses(historic_prices, predicted_prices)
    if analyses_result != None:
        notifier.notify(analyses_result)
