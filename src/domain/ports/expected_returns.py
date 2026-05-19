"""ExpectedReturnsModel port — maps price history + views to a posterior expected-return vector."""

from typing import Protocol

import pandas as pd


class ExpectedReturnsModel(Protocol):
    def mu(
        self,
        prices: pd.DataFrame,
        market_caps: dict[str, float],
        views: dict[str, float],
        uncertainties: dict[str, float],
    ) -> pd.Series:
        """Return annualised expected simple returns indexed by ticker.

        Args:
            prices: Historical price DataFrame (dates × tickers).
            market_caps: Total assets / market cap per ticker.
            views: Per-ticker annualised log-return views (from GARCH/Prophet).
            uncertainties: Per-ticker forecast uncertainty (positive scalar).
        """
        ...
