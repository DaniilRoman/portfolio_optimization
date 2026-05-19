"""Routes optimize() calls to the GA, CVXPY, or HRP backend based on settings.OPTIMIZER."""

import importlib
import logging

import numpy as np
import pandas as pd

from config.configuration import settings
from src.domain.data.data import OptimizationResult, StockData

_OPTIMIZERS: dict[str, str] = {
    "ga": ".optimizer",
    "cvxpy": ".optimizer_cvxpy",
    "hrp": ".optimizer_hrp",
}

_SIGMA_BACKENDS = {"cvxpy", "hrp"}


def optimize(
    stocks: list[StockData],
    budget: int = 100,
    price_history: pd.DataFrame | None = None,
) -> OptimizationResult:
    name = settings.OPTIMIZER
    if name not in _OPTIMIZERS:
        raise ValueError(f"Unknown OPTIMIZER: {name!r}. Expected one of: {sorted(_OPTIMIZERS)}")
    module = importlib.import_module(_OPTIMIZERS[name], package=__package__)

    if name in _SIGMA_BACKENDS:
        sigma: np.ndarray | None = None
        if price_history is not None and price_history.shape[1] >= 2:
            from src.adapter.out.risk.covariance import compute_covariance

            sigma = compute_covariance(price_history)

        if name == "cvxpy":
            mu: np.ndarray | None = None
            if settings.EXPECTED_RETURNS == "bl" and sigma is not None:
                mu = _compute_bl_mu(stocks, sigma)
                if mu is not None:
                    logging.info("Using Black-Litterman posterior expected returns for CVXPY optimizer.")
            return module.optimize(stocks, budget, sigma=sigma, mu=mu)  # type: ignore[no-any-return]

        # HRP and other sigma-based backends: no expected-return override
        return module.optimize(stocks, budget, sigma=sigma)  # type: ignore[no-any-return]

    return module.optimize(stocks, budget)  # type: ignore[no-any-return]


def _compute_bl_mu(stocks: list[StockData], sigma: np.ndarray) -> np.ndarray | None:
    """Compute Black-Litterman posterior expected returns and return as numpy array aligned with stocks."""
    try:
        from src.adapter.out.expected_returns.black_litterman import compute_bl_mu

        tickers = [s.asset.ticker_symbol for s in stocks]
        views = {s.asset.ticker_symbol: s.forecast.expected_log_return for s in stocks}
        uncertainties = {s.asset.ticker_symbol: s.forecast.prediction_uncertainty for s in stocks}
        market_caps = {s.asset.ticker_symbol: max(s.asset.assets_under_management, 1.0) for s in stocks}

        bl_series = compute_bl_mu(tickers, sigma, market_caps, views, uncertainties)
        return np.array([float(bl_series.get(t, 0.0)) for t in tickers], dtype=float)
    except Exception as exc:
        logging.warning("Black-Litterman failed (%s); falling back to mean expected returns.", exc)
        return None
