from typing import Protocol

from src.logic.data.data import StockData


class Optimizer(Protocol):
    def optimize(self, stocks: list[StockData], budget: int) -> str: ...
