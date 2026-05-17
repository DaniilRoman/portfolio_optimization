"""Predicter dispatcher.

Routes ``predict`` calls to the GARCH or Prophet backend based on
``config.configuration.PREDICTER`` (env var ``PREDICTER``: "garch" | "prophet").
Both backends share the same return contract: ``(model, forecast_df)`` where
the model exposes ``.plot(forecast_df)`` and the DataFrame carries
``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``, ``uncertainty_range``,
``volatility_forecast``.
"""

from typing import Any, Tuple

from pandas import DataFrame

import config.configuration as configuration


def predict(data: DataFrame, predict_period: int = 30, **kwargs) -> Tuple[Any, DataFrame]:
    backend = (configuration.PREDICTER or "garch").lower()
    if backend == "prophet":
        from .predicter_prophet import predict as _predict
    elif backend == "garch":
        from .predicter_garch import predict as _predict
    else:
        raise ValueError(
            f"Unknown PREDICTER backend: {backend!r}. Expected 'garch' or 'prophet'."
        )
    return _predict(data, predict_period=predict_period, **kwargs)
