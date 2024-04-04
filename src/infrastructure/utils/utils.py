import pandas as pd
from datetime import datetime, timedelta

round_precise = 5


def current_date_str():
    return str(current_date())


def current_date():
    return datetime.date(datetime.now())


def __start_date(start_day: str):
    if start_day is None:
        return current_date()
    else:
        return datetime.strptime(start_day, "%Y-%m-%d").date()


def next_day(days, start_day: str = None):
    return str(__start_date(start_day) + timedelta(days=days))


def prev_day(days, start_day: str = None):
    return str(__start_date(start_day) - timedelta(days=days))


def sp500_stocks():
    # There are 2 tables on the Wikipedia page
    # we want the first table

    payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    first_table = payload[0]
    second_table = payload[1]

    df = first_table

    symbols = df['Symbol'].values.tolist()
    excluded_stocks = ["BRK.B", "BF.B", "MMM", "AES", "AFL", "A", "ABT", "ADBE"]
    return list(filter(lambda s: s not in excluded_stocks, symbols))
