"""Routes optimize() calls to the GA or CVXPY backend based on settings.OPTIMIZER."""
from config.configuration import settings
from src.logic.data.data import OptimizationResult, StockData

_OPTIMIZERS: dict[str, str] = {
    "ga": ".optimizer",
    "cvxpy": ".optimizer_cvxpy",
}


def optimize(stocks: list[StockData], budget: int = 10000) -> OptimizationResult:
    name = settings.OPTIMIZER
    if name not in _OPTIMIZERS:
        raise ValueError(f"Unknown OPTIMIZER: {name!r}. Expected one of: {sorted(_OPTIMIZERS)}")
    if name == "cvxpy":
        from .optimizer_cvxpy import optimize as _optimize
    else:
        from .optimizer import optimize as _optimize
    return _optimize(stocks, budget)
