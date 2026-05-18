"""Forecast value object wrapping a prediction DataFrame with typed accessors for final price, uncertainty band, and volatility."""
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.logic.data.schemas import validate_forecast_frame


@dataclass
class Forecast:
    """Typed wrapper around a predictor's output DataFrame.

    Columns expected: ds, yhat, yhat_lower, yhat_upper,
    uncertainty_range (optional), volatility_forecast (optional).
    The ``model`` attribute is the raw backend object (GARCH ARCHModel or
    Prophet instance) needed only for plotting.
    """

    series: pd.DataFrame
    model: Any = field(default=None)

    def __post_init__(self) -> None:
        validate_forecast_frame(self.series)

    def final_price(self) -> float:
        return float(self.series["yhat"].iloc[-1])

    def lower_price(self) -> float:
        return float(self.series["yhat_lower"].iloc[-1])

    def uncertainty_band(self) -> float:
        if "uncertainty_range" in self.series.columns:
            return float(self.series["uncertainty_range"].mean())
        return 0.0

    def volatility(self) -> float:
        if "volatility_forecast" in self.series.columns:
            return float(self.series["volatility_forecast"].mean())
        return 0.0

    def plot(self, df: pd.DataFrame) -> None:
        """Delegate to the backend model's plot method for chart generation."""
        if self.model is not None:
            self.model.plot(df)
