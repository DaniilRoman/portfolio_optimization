from typing import Protocol

import pandas as pd

from src.logic.data.forecast import Forecast


class Predictor(Protocol):
    def predict(self, history: pd.DataFrame, period: int) -> Forecast: ...
