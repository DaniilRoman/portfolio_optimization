from typing import Protocol

from src.logic.data.data import OptimizationResult, StockData


class Optimizer(Protocol):
    def optimize(self, stocks: list[StockData], budget: int) -> OptimizationResult: ...
