"""Routes optimize() calls to the GA or CVXPY backend based on settings.OPTIMIZER."""
import importlib

from config.configuration import settings
from src.domain.data.data import OptimizationResult, StockData

_OPTIMIZERS: dict[str, str] = {
    "ga": ".optimizer",
    "cvxpy": ".optimizer_cvxpy",
}


def optimize(stocks: list[StockData], budget: int = 10000) -> OptimizationResult:
    name = settings.OPTIMIZER
    if name not in _OPTIMIZERS:
        raise ValueError(f"Unknown OPTIMIZER: {name!r}. Expected one of: {sorted(_OPTIMIZERS)}")
    module = importlib.import_module(_OPTIMIZERS[name], package=__package__)
    return module.optimize(stocks, budget)  # type: ignore[no-any-return]
