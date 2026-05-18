"""Ledoit-Wolf shrinkage covariance estimator for portfolio optimization."""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

_TRADING_DAYS = 252.0


def compute_covariance(prices: pd.DataFrame) -> np.ndarray:
    """Compute Ledoit-Wolf shrunk annualized covariance matrix.

    Args:
        prices: DataFrame with asset tickers as columns and dates as index.
                Must have at least 2 rows and 2 columns.
    Returns:
        Annualized covariance matrix, shape (n_assets, n_assets), PSD by construction.
    Raises:
        ValueError: if prices has fewer than 2 rows or fewer than 2 assets.
    """
    if prices.shape[0] < 2:
        raise ValueError(f"Need at least 2 price observations; got {prices.shape[0]}")
    if prices.shape[1] < 2:
        raise ValueError(f"Need at least 2 assets; got {prices.shape[1]}")

    log_returns = np.log(prices / prices.shift(1)).dropna()
    lw = LedoitWolf()
    lw.fit(log_returns.to_numpy(dtype=float))
    return np.asarray(lw.covariance_ * _TRADING_DAYS)


class LedoitWolfCovarianceModel:
    """RiskModel adapter using Ledoit-Wolf linear shrinkage."""

    def cov(self, prices: pd.DataFrame) -> np.ndarray:
        return compute_covariance(prices)
