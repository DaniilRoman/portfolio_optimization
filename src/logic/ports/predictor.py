from typing import Any, Protocol

import pandas as pd


class Predictor(Protocol):
    def predict(self, history: pd.DataFrame, period: int) -> tuple[Any, pd.DataFrame]: ...
