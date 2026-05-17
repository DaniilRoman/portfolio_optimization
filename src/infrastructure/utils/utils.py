"""Shared date-arithmetic and rounding utilities used across adapters."""
import datetime as _dt
from datetime import datetime, timedelta

import pandas as pd

round_precise = 5


def current_date_str() -> str:
    return str(current_date())


def current_date() -> _dt.date:
    return datetime.date(datetime.now())


def __start_date(start_day: str | None) -> _dt.date:
    if start_day is None:
        return current_date()
    else:
        return datetime.strptime(start_day, "%Y-%m-%d").date()


def next_day(days: int, start_day: str | None = None) -> str:
    return str(__start_date(start_day) + timedelta(days=days))


def prev_day(days: int, start_day: str | None = None) -> str:
    return str(__start_date(start_day) - timedelta(days=days))


def sp500_stocks() -> list[str]:
    payload = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = payload[0]
    symbols = df["Symbol"].values.tolist()
    excluded_stocks = ["BRK.B", "BF.B", "MMM", "AES", "AFL", "A", "ABT", "ADBE"]
    return list(filter(lambda s: s not in excluded_stocks, symbols))
