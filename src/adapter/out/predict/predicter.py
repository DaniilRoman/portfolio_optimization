"""Predicter dispatcher.

Routes ``predict`` calls to the GARCH or Prophet backend based on
``PREDICTER`` setting (env var ``PREDICTER``: "garch" | "prophet").
Both backends return a :class:`~src.logic.data.forecast.Forecast` carrying
``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``, ``uncertainty_range``,
``volatility_forecast`` and a ``model`` that exposes ``.plot(df)``.
"""

import importlib

import pandas as pd

from config.configuration import settings
from src.logic.data.forecast import Forecast

_BACKENDS: dict[str, str] = {
    "garch": ".predicter_garch",
    "prophet": ".predicter_prophet",
}


def predict(data: pd.DataFrame, predict_period: int = 30, **kwargs: object) -> Forecast:
    backend = settings.PREDICTER.lower()
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown PREDICTER backend: {backend!r}. Expected one of: {sorted(_BACKENDS)}")
    module = importlib.import_module(_BACKENDS[backend], package=__package__)
    return module.predict(data, predict_period=predict_period, **kwargs)  # type: ignore[no-any-return]
