"""FactorModel port — maps a list of AnalysisReports to cross-sectional factor scores."""

from typing import Protocol

import pandas as pd

from src.domain.data.data import AnalysisReport


class FactorModel(Protocol):
    def score(self, reports: list[AnalysisReport]) -> pd.DataFrame:
        """Return DataFrame indexed by ticker with columns [value, quality, momentum, low_vol, size, composite].

        All factor columns are cross-sectionally z-scored.  'composite' is their equal-weight mean.
        """
        ...
