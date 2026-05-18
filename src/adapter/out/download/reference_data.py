"""Reference data fetchers: benchmark prices, risk-free rate, Fama-French factor returns.

All results are cached under data/cache/ with vintage-date keys, matching the
convention established in cache.py. All external I/O goes through three narrow
entry points (_yf_history, _urlopen_bytes) so tests can mock at the boundary.
"""

import io
import logging
import urllib.request
import zipfile
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import yfinance as yf

_BENCH_CACHE = Path("data/cache/benchmarks")
_RF_CACHE = Path("data/cache/risk_free")
_FF_CACHE = Path("data/cache/ff_factors")

_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

_FF5_URLS: dict[str, str] = {
    "daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "monthly": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
}
_MOM_URLS: dict[str, str] = {
    "daily": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
    "monthly": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
}

FF_COLUMNS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


# ---------------------------------------------------------------------------
# Internal I/O helpers — thin wrappers so tests can monkeypatch cleanly
# ---------------------------------------------------------------------------


def _yf_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True, actions=False)


def _urlopen_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        return bytes(resp.read())


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _load_ref_cache(root: Path, key: str, vintage_date: date) -> pd.DataFrame | None:
    p = root / f"{key}_{vintage_date}.parquet"
    if p.exists():
        logging.debug("Ref cache hit: %s @ %s", key, vintage_date)
        return pd.read_parquet(p)
    return None


def _save_ref_cache(root: Path, key: str, df: pd.DataFrame, vintage_date: date) -> None:
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{key}_{vintage_date}.parquet"
    df.to_parquet(p)
    logging.debug("Ref cache write: %s @ %s (%d rows)", key, vintage_date, len(df))


def _today_pit(point_in_time: pd.Timestamp | None) -> tuple[pd.Timestamp, date]:
    pit = (point_in_time if point_in_time is not None else pd.Timestamp.today()).normalize()
    return pit, pit.date()


def _strip_tz(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert to UTC then drop timezone info so cached parquets are always tz-naive."""
    idx = pd.to_datetime(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_benchmark(
    ticker: str,
    start: str,
    end: str,
    point_in_time: pd.Timestamp | None = None,
) -> pd.Series:
    """Return total-return adjusted Close prices for *ticker* as a pd.Series."""
    pit, vintage_date = _today_pit(point_in_time)

    cached = _load_ref_cache(_BENCH_CACHE, ticker, vintage_date)
    if cached is not None:
        return cached["Close"].loc[pd.Timestamp(start) : pd.Timestamp(end)]

    hist = _yf_history(ticker, start, end)
    if hist.empty:
        raise ValueError(f"No benchmark data returned for {ticker!r}")
    hist = hist[~hist.index.duplicated(keep="first")]
    to_cache = hist[["Close"]].copy()
    to_cache.index = _strip_tz(to_cache.index)
    _save_ref_cache(_BENCH_CACHE, ticker, to_cache, vintage_date)
    series = to_cache["Close"]
    return series


def fetch_risk_free_rate(
    start: str,
    end: str,
    series: str = "DGS3MO",
    point_in_time: pd.Timestamp | None = None,
) -> pd.Series:
    """Return the annualised risk-free rate as a decimal pd.Series (e.g. 0.0525 = 5.25 %).

    Source: FRED public CSV endpoint. Missing values (weekends, holidays) are dropped.
    """
    pit, vintage_date = _today_pit(point_in_time)

    cached = _load_ref_cache(_RF_CACHE, series, vintage_date)
    if cached is not None:
        rate = cached["Rate"]
        rate.index = pd.to_datetime(rate.index)
        return rate.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    url = _FRED_URL.format(series=series)
    raw = _urlopen_bytes(url).decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True, na_values=["."])
    df = df.dropna()
    df.columns = ["Rate"]
    df["Rate"] = df["Rate"] / 100.0
    df.index.name = "Date"
    _save_ref_cache(_RF_CACHE, series, df, vintage_date)
    rate = df["Rate"]
    return rate.loc[pd.Timestamp(start) : pd.Timestamp(end)]


def fetch_ff_factors(
    model: Literal["ff5"] = "ff5",
    start: str = "",
    end: str = "",
    freq: Literal["daily", "monthly"] = "daily",
    point_in_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return Fama-French factor returns as a pd.DataFrame with columns FF_COLUMNS.

    Values are in decimal form (0.0005 = 0.05%). The RF column from the FF5 file
    is dropped; use fetch_risk_free_rate for the risk-free series.
    """
    pit, vintage_date = _today_pit(point_in_time)
    cache_key = f"{model}_{freq}"

    cached = _load_ref_cache(_FF_CACHE, cache_key, vintage_date)
    if cached is not None:
        cached.index = pd.to_datetime(cached.index)
        result = cached
        if start:
            result = result.loc[pd.Timestamp(start) :]
        if end:
            result = result.loc[: pd.Timestamp(end)]
        return result

    ff5 = _fetch_and_parse_ff_zip(_FF5_URLS[freq], freq)
    mom = _fetch_and_parse_ff_zip(_MOM_URLS[freq], freq)

    # FF5 columns: Mkt-RF, SMB, HML, RMW, CMA, RF — drop RF
    ff5 = ff5.drop(columns=["RF"], errors="ignore")

    # Normalise the single MOM column to "Mom" regardless of source name
    mom = mom.rename(columns={mom.columns[0]: "Mom"})

    combined = ff5.join(mom, how="inner")
    combined = combined / 100.0
    combined.index.name = "Date"

    _save_ref_cache(_FF_CACHE, cache_key, combined, vintage_date)

    result = combined
    if start:
        result = result.loc[pd.Timestamp(start) :]
    if end:
        result = result.loc[: pd.Timestamp(end)]
    return result


# ---------------------------------------------------------------------------
# FF CSV parsing
# ---------------------------------------------------------------------------


def _fetch_and_parse_ff_zip(url: str, freq: Literal["daily", "monthly"]) -> pd.DataFrame:
    """Download a Kenneth French zip and return the parsed factor DataFrame (raw %)."""
    raw_bytes = _urlopen_bytes(url)
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        csv_name = next(n for n in zf.namelist() if n.upper().endswith(".CSV"))
        content = zf.read(csv_name).decode("latin-1")

    return _parse_ff_csv(content, freq)


def _parse_ff_csv(content: str, freq: Literal["daily", "monthly"]) -> pd.DataFrame:
    """Parse the text of a Kenneth French factor CSV file.

    The files use whitespace as delimiter and have a preamble + optional footer.
    We locate the header row, collect contiguous data rows, strip leading/trailing
    whitespace from each line, then parse with sep=r'\\s+'.
    """
    lines = content.splitlines()

    # Find the header line: FF5 files contain "Mkt-RF"; MOM files contain "Mom"
    header_idx = next(
        i for i, line in enumerate(lines) if "Mkt-RF" in line or ("Mom" in line and "Date" not in line and i > 0)
    )

    # Collect header + data rows; stop at copyright/annual-summary footer
    data_lines = [lines[header_idx].strip()]
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if "Copyright" in line or "Annual" in line:
            break
        # Monthly FF files have a second "Annual Factors" table — stop there
        first_token = stripped.split()[0]
        if freq == "monthly" and len(first_token) == 4 and first_token.isdigit() and int(first_token) > 2050:
            break
        data_lines.append(stripped)

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        index_col=0,
        sep=r"\s+",
        na_values=["-99.99", "-999"],
        engine="python",
    )
    df = df.dropna(how="all")
    df.columns = df.columns.str.strip()
    df.index = df.index.astype(str).str.strip()

    if freq == "daily":
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
    else:
        # Monthly: index is YYYYMM → last day of that month
        df.index = pd.to_datetime(df.index, format="%Y%m") + pd.offsets.MonthEnd(0)

    df.index.name = "Date"
    return df
