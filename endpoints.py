from services.stock_data_downloader import get_current_price, search_stocks


def get_price_endpoint(stock_name):
    return get_current_price(stock_name)


def search_stocks_endpoint(stock_name_query):
    return search_stocks(stock_name_query)


if __name__ == '__main__':
    print(get_price_endpoint("AMD"))
    print(search_stocks("AM"))
