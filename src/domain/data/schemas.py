"""DataFrame column contracts for the two internal frame shapes.

`validate_history_frame` is called at the downloader boundary;
`validate_forecast_frame` is called inside `Forecast.__post_init__`.
Both raise `ValueError` with a descriptive message on mismatch so the
failure surfaces at the source rather than inside the analyzer.
"""

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
