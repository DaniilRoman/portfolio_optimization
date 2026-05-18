"""RiskModel protocol: computes a covariance matrix from price history."""

from typing import Protocol

import numpy as np
import pandas as pd


class RiskModel(Protocol):
    def cov(self, prices: pd.DataFrame) -> np.ndarray: ...
