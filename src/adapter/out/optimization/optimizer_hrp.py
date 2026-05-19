"""Hierarchical Risk Parity optimizer using pypfopt.HRPOpt; implements both Optimizer and WeightProvider protocols."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pypfopt import HRPOpt

from config.configuration import settings
from src.adapter.out.risk.covariance import compute_covariance
from src.domain.data.data import AnalysisReport, OptimizationResult

from .optimizer import (
    __get_etf_map,
    _build_portfolio,
    _calculate_max_shares,
    _create_ownership_weights,
    _prepare_stock_data,
)
from .optimizer_cvxpy import _repair_solution

StockData = AnalysisReport


def compute_weights(prices: pd.DataFrame) -> dict[str, float]:
    """WeightProvider protocol: maps price history to HRP fractional weights for the backtest harness."""
    sigma = compute_covariance(prices)
    tickers = list(prices.columns)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    cov_df = pd.DataFrame(sigma, index=tickers, columns=tickers)
    hrp = HRPOpt(returns=log_returns, cov_matrix=cov_df)
    weights = hrp.optimize(linkage_method=settings.hrp.linkage_method)
    return {str(k): float(v) for k, v in weights.items()}


def _hrp_weights_to_shares(
    weights: dict[str, float],
    tickers: list[str],
    current_prices: np.ndarray,
    max_shares: np.ndarray,
    budget: float,
    scores: np.ndarray,
) -> list[int]:
    target_values = np.array([weights.get(t, 0.0) * budget for t in tickers])
    target_shares = np.where(
        current_prices > 0,
        target_values / np.maximum(current_prices, 1e-9),
        0.0,
    )
    return _repair_solution(target_shares, current_prices, max_shares, budget, scores)


def _greedy_profit_shares(
    mu: np.ndarray,
    current_prices: np.ndarray,
    max_shares: np.ndarray,
    budget: float,
    scores: np.ndarray,
) -> list[int]:
    """Profit-only: allocate budget proportionally to positive expected profit per share."""
    weights = np.maximum(mu, 0.0)
    total = float(weights.sum())
    if total <= 0:
        weights = np.ones(len(mu), dtype=float)
        total = float(len(mu))
    target_values = weights / total * budget
    target_shares = np.where(
        current_prices > 0,
        target_values / np.maximum(current_prices, 1e-9),
        0.0,
    )
    return _repair_solution(target_shares, current_prices, max_shares, budget, scores)


def optimize(
    stocks: list[StockData],
    budget: float = 50.0,
    max_per_etf_budget: float = 50.0,
    sigma: np.ndarray | None = None,
) -> OptimizationResult:
    if sigma is None:
        raise ValueError(
            "HRP optimizer requires a covariance matrix (sigma). "
            "Ensure price_history is passed to optimizer_dispatcher.optimize()."
        )
    if max_per_etf_budget is None:
        max_per_etf_budget = min(50.0, budget / 2)
    elif max_per_etf_budget > budget:
        max_per_etf_budget = budget

    etf_map = __get_etf_map()
    tickers, current_prices, predicted_prices, dividend_yields, expense_ratios = _prepare_stock_data(stocks)

    current_prices_arr = np.asarray(current_prices, dtype=float)
    max_shares_arr = np.asarray(_calculate_max_shares(current_prices, max_per_etf_budget), dtype=int)
    ownership_weights = _create_ownership_weights(tickers, etf_map)

    mu = np.array(
        [
            (
                (predicted_prices[i] - current_prices[i])
                + current_prices[i] * dividend_yields[i]
                - current_prices[i] * expense_ratios[i]
            )
            * ownership_weights[i]
            for i in range(len(stocks))
        ],
        dtype=float,
    )
    scores = mu / np.maximum(current_prices_arr, 1e-9)

    cov_df = pd.DataFrame(sigma, index=tickers, columns=tickers)
    hrp = HRPOpt(cov_matrix=cov_df)
    weights = hrp.optimize(linkage_method=settings.hrp.linkage_method)

    risk_individual = _hrp_weights_to_shares(weights, tickers, current_prices_arr, max_shares_arr, budget, scores)
    profit_individual = _greedy_profit_shares(mu, current_prices_arr, max_shares_arr, budget, scores)

    return OptimizationResult(
        risk_aware=_build_portfolio(
            risk_individual, stocks, current_prices, predicted_prices, dividend_yields, expense_ratios, tickers
        ),
        profit_only=_build_portfolio(
            profit_individual, stocks, current_prices, predicted_prices, dividend_yields, expense_ratios, tickers
        ),
    )
