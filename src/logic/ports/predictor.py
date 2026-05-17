"""Predictor protocol: defines the predict(history, period) -> Forecast contract all predictor backends must satisfy."""
from typing import Protocol

import pandas as pd

from src.logic.data.forecast import Forecast


class Predictor(Protocol):
    def predict(self, history: pd.DataFrame, period: int) -> Forecast: ...
