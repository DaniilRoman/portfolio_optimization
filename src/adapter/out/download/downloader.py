"""Downloads OHLCV price history and ticker metadata from yfinance; outputs StockInfo with TickerMetadata."""
import json
import logging
import urllib
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf

from src.adapter.out.download import cache
from src.domain.data import data
from src.domain.data.data import StockInfo, TickerMetadata
from src.domain.data.schemas import validate_history_frame, validate_history_quality
from src.infrastructure.utils import utils


def __search_stocks(stock_name_query: str) -> list[str]:
    response = urllib.request.urlopen(f"https://query2.finance.yahoo.com/v1/finance/search?q={stock_name_query}")
    content = response.read()
    return [i["symbol"] for i in json.loads(content.decode("utf8"))["quotes"]]


def _extract_ticker_metadata(stock: yf.Ticker) -> TickerMetadata:
    info = stock.info or {}

    expense_ratio = info.get("netExpenseRatio") or info.get("expenseRatio") or 0
    if expense_ratio != 0:
        expense_ratio = expense_ratio / 100.0

    try:
        funds = stock.funds_data
        top_holdings: np.ndarray = funds.top_holdings.values if funds is not None else np.array([])
        sector_weights: dict[str, float] = funds.sector_weightings if funds is not None else {}
        fund_description = (
            f"{funds.fund_overview.get('family', '')} || {funds.fund_overview.get('legalType', '')} || {funds.description}"
            if funds is not None
            else ""
        )
    except Exception:
        top_holdings = np.array([])
        sector_weights = {}
        fund_description = ""

    return TickerMetadata(
        long_name=info.get("longName") or "Unknown",
        currency=info.get("currency") or "Unknown",
        industry=info.get("industry") or "",
        beta=float(info.get("beta") or info.get("beta3Year") or 0),
        dividend_yield=float(info.get("yield") or info.get("dividendYield") or info.get("trailingAnnualDividendYield") or 0),
        total_assets=float(info.get("totalAssets") or info.get("netAssets") or 0),
        expense_ratio=float(expense_ratio),
        average_volume=float(info.get("averageVolume") or 0),
        trailing_eps=float(info.get("trailingEps") or 0),
        forward_eps=float(info.get("forwardEps") or 0),
        net_income_to_common=float(info.get("netIncomeToCommon") or 0),
        ebitda_margins=float(info.get("ebitdaMargins") or 0),
        operating_margins=float(info.get("operatingMargins") or 0),
        top_holdings=top_holdings,
        sector_weights=sector_weights,
        description=fund_description,
    )


def download_stock_data(
    stock_name: str,
    start_date: str | None = None,
    end_date: str | None = None,
    point_in_time: pd.Timestamp | None = None,
) -> StockInfo:
    if start_date is None:
        start_date = utils.current_date_str()
    if end_date is None:
        end_date = utils.next_day(1)
    pit = (point_in_time if point_in_time is not None else pd.Timestamp.today()).normalize()
    vintage_date = pit.date()

    hist = cache.load(stock_name, vintage_date)
    if hist is None:
        stock = yf.Ticker(stock_name)
        hist = stock.history(start=start_date, end=end_date, auto_adjust=True, actions=True)
        if hist.empty:
            raise data.SkipException(f"History data is empty for a stock: {stock_name}")
        hist = hist[~hist.index.duplicated(keep="first")]
        cache.save(stock_name, hist, vintage_date)
    elif hist.empty:
        raise data.SkipException(f"History data is empty for a stock: {stock_name}")

    stock = yf.Ticker(stock_name)

    historic_data = hist["Close"].to_frame("y")
    historic_data["ds"] = pd.to_datetime(historic_data.index.date)
    historic_data = historic_data.drop_duplicates(subset=["ds"], keep="last")
    historic_data = historic_data.sort_index()

    ticker_meta = _extract_ticker_metadata(stock)
    validate_history_frame(historic_data)
    validate_history_quality(historic_data)
    logging.debug("Loaded %s: %d rows, last price %.2f", stock_name, len(historic_data), historic_data["y"].iloc[-1])

    return StockInfo(historic_data=historic_data, ticker=ticker_meta, point_in_time=pit)
