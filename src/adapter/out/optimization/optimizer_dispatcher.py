"""Routes optimize() calls to the GA or CVXPY backend based on settings.OPTIMIZER."""
import importlib

import numpy as np
import pandas as pd

from config.configuration import settings
from src.domain.data.data import OptimizationResult, StockData


_OPTIMIZERS: dict[str, str] = {
    "ga": ".optimizer",
    "cvxpy": ".optimizer_cvxpy",
}


def optimize(
    stocks: list[StockData],
    budget: int = 10000,
    price_history: pd.DataFrame | None = None,
) -> OptimizationResult:
    name = settings.OPTIMIZER
    if name not in _OPTIMIZERS:
        raise ValueError(f"Unknown OPTIMIZER: {name!r}. Expected one of: {sorted(_OPTIMIZERS)}")
    module = importlib.import_module(_OPTIMIZERS[name], package=__package__)

    if name == "cvxpy":
        sigma: np.ndarray | None = None
        if price_history is not None and price_history.shape[1] >= 2:
            from src.adapter.out.risk.covariance import compute_covariance
            sigma = compute_covariance(price_history)
        return module.optimize(stocks, budget, sigma=sigma)  # type: ignore[no-any-return]

    return module.optimize(stocks, budget)  # type: ignore[no-any-return]
