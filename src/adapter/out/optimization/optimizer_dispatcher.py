from config.configuration import settings
from src.logic.data.data import StockData


def optimize(stocks: list[StockData], budget: int = 10000) -> str:
    if settings.OPTIMIZER == "cvxpy":
        from .optimizer_cvxpy import optimize as _optimize
    else:  # "ga"
        from .optimizer import optimize as _optimize
    return _optimize(stocks, budget)
