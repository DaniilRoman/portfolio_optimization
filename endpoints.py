from src.stock_data_downloader import get_price, search_stocks


def get_price_endpoint(stock_name):
    return get_price(stock_name, "1d")



if __name__ == '__main__':
    print(get_price_endpoint("AMD"))
    print(search_stocks("AM"))

