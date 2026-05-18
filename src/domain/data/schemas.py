"""DataFrame column contracts for the two internal frame shapes.

`validate_history_frame` is called at the downloader boundary;
`validate_forecast_frame` is called inside `Forecast.__post_init__`.
Both raise `ValueError` with a descriptive message on mismatch so the
failure surfaces at the source rather than inside the analyzer.
"""

import logging

import numpy as np
import pandas as pd

# Required columns — sufficient to detect a renamed or missing column early.
HISTORY_COLUMNS: frozenset[str] = frozenset({"ds", "y"})
FORECAST_COLUMNS: frozenset[str] = frozenset({"ds", "yhat", "yhat_lower", "yhat_upper"})


def validate_history_frame(df: pd.DataFrame) -> None:
    """Raise ValueError if *df* is missing any required history columns."""
    missing = HISTORY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"HistoryFrame is missing required columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}"
        )


def validate_forecast_frame(df: pd.DataFrame) -> None:
    """Raise ValueError if *df* is missing any required forecast columns."""
    missing = FORECAST_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"ForecastFrame is missing required columns: {sorted(missing)}. "
            f"Got: {sorted(df.columns)}"
        )


_MAX_GAP_CALENDAR_DAYS = 7
_OUTLIER_SIGMA = 5.0


def validate_history_quality(df: pd.DataFrame) -> None:
    """Warn or raise on data-quality issues in a validated history frame.

    Raises ValueError on non-positive prices.
    Logs a warning for calendar gaps > 7 days or single-day log-return outliers > 5σ.
    """
    if not (df["y"] > 0).all():
        bad = df.loc[df["y"] <= 0, "y"]
        raise ValueError(f"Non-positive prices in history frame: {bad.head().to_dict()}")

    dates = pd.to_datetime(df["ds"]).sort_values()
    max_gap = int(dates.diff().dt.days.dropna().max())
    if max_gap > _MAX_GAP_CALENDAR_DAYS:
        logging.warning("Price series has a %d-day gap (threshold=%d)", max_gap, _MAX_GAP_CALENDAR_DAYS)

    prices = df["y"].to_numpy(dtype=float)
    log_returns = np.diff(np.log(prices))
    sigma = float(log_returns.std())
    if sigma > 0:
        n_outliers = int((np.abs(log_returns) > _OUTLIER_SIGMA * sigma).sum())
        if n_outliers:
            logging.warning("%d single-day log-return outlier(s) > %.0fσ detected", n_outliers, _OUTLIER_SIGMA)
