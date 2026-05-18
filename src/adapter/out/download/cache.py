"""Parquet-backed price-history cache keyed by (ticker, vintage_date).

Cache path: data/cache/prices/{ticker}/1d/{YYYY-MM-DD}.parquet
The file stores the raw yfinance history DataFrame (Close, Volume, Dividends, Stock Splits).
vintage_date is the calendar date the data was downloaded, providing reproducibility.
"""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

_CACHE_ROOT = Path("data/cache/prices")


def _path(ticker: str, vintage_date: date) -> Path:
    return _CACHE_ROOT / ticker / "1d" / f"{vintage_date}.parquet"


def load(ticker: str, vintage_date: date) -> pd.DataFrame | None:
    p = _path(ticker, vintage_date)
    if p.exists():
        logging.debug("Cache hit: %s @ %s", ticker, vintage_date)
        return pd.read_parquet(p)
    return None


def save(ticker: str, df: pd.DataFrame, vintage_date: date) -> None:
    p = _path(ticker, vintage_date)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    logging.debug("Cache write: %s @ %s (%d rows)", ticker, vintage_date, len(df))
