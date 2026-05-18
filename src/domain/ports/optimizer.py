"""Optimizer protocol: defines the optimize(stocks, budget) -> OptimizationResult contract all optimizer backends must satisfy."""
from typing import Protocol

from src.domain.data.data import AnalysisReport, OptimizationResult


class Optimizer(Protocol):
    def optimize(self, stocks: list[AnalysisReport], budget: int) -> OptimizationResult: ...
