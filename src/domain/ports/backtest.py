"""WeightProvider port — any callable that maps price history to portfolio weights."""
from typing import Protocol

import pandas as pd


class WeightProvider(Protocol):
    def compute_weights(self, prices: pd.DataFrame) -> dict[str, float]:
        """Return {ticker: weight} summing to ~1, given price history (dates × tickers)."""
        ...
