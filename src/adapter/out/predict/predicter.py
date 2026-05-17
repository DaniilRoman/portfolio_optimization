"""Predicter dispatcher.

Routes ``predict`` calls to the GARCH or Prophet backend based on
``PREDICTER`` setting (env var ``PREDICTER``: "garch" | "prophet").
Both backends return a :class:`~src.logic.data.forecast.Forecast` carrying
``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``, ``uncertainty_range``,
``volatility_forecast`` and a ``model`` that exposes ``.plot(df)``.
"""

import pandas as pd

from config.configuration import settings
from src.logic.data.forecast import Forecast


def predict(data: pd.DataFrame, predict_period: int = 30, **kwargs: object) -> Forecast:
    backend = settings.PREDICTER.lower()
    if backend == "prophet":
        from .predicter_prophet import predict as _predict  # type: ignore[assignment]
    elif backend == "garch":
        from .predicter_garch import predict as _predict  # type: ignore[assignment]
    else:
        raise ValueError(f"Unknown PREDICTER backend: {backend!r}. Expected 'garch' or 'prophet'.")
    return _predict(data, predict_period=predict_period, **kwargs)  # type: ignore[arg-type]
