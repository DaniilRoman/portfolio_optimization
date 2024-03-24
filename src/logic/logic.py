from adapter.out.download import downloader
from adapter.out.predict import predicter
from infrastructure.utils import utils

def __optimal(current_price, predicted_price):
    return

def __current_price(one_stock_data):
    return round(one_stock_data.tail(1)["y"].iloc[0], round_precise)

def run(stock_name: str):
    print(f"Job `{stock_name}` created")

    prices = downloader.download_stock_data(stock_name, start_date=utils.prev_day(365*3))
    predicted_price = predicter.predict_value(prices, predict_period, current_date)
    optimal = __optimal(current_price, predicted_price)
    notifier.notify(optimal)
