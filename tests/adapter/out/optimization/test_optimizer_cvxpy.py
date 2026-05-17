"""Tests for the CVXPY optimizer."""

from unittest.mock import patch

import numpy as np

from src.adapter.out.optimization import optimizer_cvxpy
from src.logic.data.data import OptimizationResult, Portfolio
from tests.factories import make_stock_data


def _make_stocks():
    sector_allocations = [
        {"Technology": 0.90, "Healthcare": 0.10},
        {"Technology": 0.25, "Healthcare": 0.20, "Financials": 0.20, "Consumer": 0.20, "Industrials": 0.15},
        {"Technology": 0.15, "Healthcare": 0.15, "Financials": 0.15, "Consumer": 0.15, "Industrials": 0.15,
         "Utilities": 0.10, "Energy": 0.10, "Materials": 0.05},
        {"Technology": 0.95, "Healthcare": 0.05},
        {"Technology": 0.60, "Healthcare": 0.40},
    ]
    top_holdings_list = [
        np.array([["Apple Inc", 0.25], ["Microsoft Corp", 0.20], ["NVIDIA Corp", 0.15]]),
        np.array([["Apple Inc", 0.08], ["Microsoft Corp", 0.07], ["Amazon.com Inc", 0.06],
                  ["Alphabet Inc", 0.05], ["Tesla Inc", 0.04], ["Meta Platforms", 0.03]]),
        np.array([["Apple Inc", 0.04], ["Microsoft Corp", 0.04], ["Amazon.com Inc", 0.03],
                  ["Alphabet Inc", 0.03], ["Johnson & Johnson", 0.03], ["JPMorgan Chase", 0.03],
                  ["Procter & Gamble", 0.03], ["Exxon Mobil", 0.03], ["Walmart", 0.03]]),
        np.array([["Tesla Inc", 0.50], ["Rivian", 0.20], ["Lucid", 0.15]]),
        np.array([["Apple Inc", 0.15], ["Microsoft Corp", 0.12], ["Amazon.com Inc", 0.10]]),
    ]
    params = [
        ("TECH", "Technology Concentrated ETF", 15.0, 18.0, 0.25, 0.01, 0.005),
        ("BLEND", "Blended Market ETF", 25.0, 26.5, 0.15, 0.02, 0.01),
        ("DIV", "Diversified Low Vol ETF", 20.0, 20.8, 0.10, 0.03, 0.02),
        ("CONC", "Highly Concentrated ETF", 12.0, 15.0, 0.30, 0.005, 0.003),
        ("LOSS", "Losing ETF", 18.0, 17.5, 0.20, 0.015, 0.015),
    ]
    stocks = []
    for idx, (ticker, name, cur, pred, std, div, exp) in enumerate(params):
        stocks.append(make_stock_data(
            ticker_symbol=ticker,
            stock_name=name,
            current_price=cur,
            predict_price=pred,
            standard_deviation=std,
            beta=1.0 + (std - 0.15) * 2,
            dividend_yield=div,
            expense_ratio=exp,
            sector_allocation=sector_allocations[idx],
            top_holdings=top_holdings_list[idx],
        ))
    return stocks


_EMPTY_MAP = {"TECH": 0, "BLEND": 0, "DIV": 0, "CONC": 0, "LOSS": 0}


def test_cvxpy_returns_optimization_result():
    stocks = _make_stocks()
    with patch.object(optimizer_cvxpy, "__get_etf_map", return_value=_EMPTY_MAP):
        result = optimizer_cvxpy.optimize(stocks, budget=50.0)
    assert isinstance(result, OptimizationResult)
    assert isinstance(result.risk_aware, Portfolio)
    assert isinstance(result.profit_only, Portfolio)


def test_budget_not_exceeded():
    stocks = _make_stocks()
    budget = 50.0
    with patch.object(optimizer_cvxpy, "__get_etf_map", return_value=_EMPTY_MAP):
        result = optimizer_cvxpy.optimize(stocks, budget=budget)
    for portfolio in (result.risk_aware, result.profit_only):
        total = sum(a.total_cost for a in portfolio.allocations)
        assert total <= budget + 1e-6, f"Total cost {total:.2f} exceeds budget {budget}"


def test_all_quantities_non_negative():
    stocks = _make_stocks()
    with patch.object(optimizer_cvxpy, "__get_etf_map", return_value=_EMPTY_MAP):
        result = optimizer_cvxpy.optimize(stocks, budget=50.0)
    for portfolio in (result.risk_aware, result.profit_only):
        for alloc in portfolio.allocations:
            assert alloc.quantity >= 0


def test_risk_metrics_present():
    stocks = _make_stocks()
    with patch.object(optimizer_cvxpy, "__get_etf_map", return_value=_EMPTY_MAP):
        result = optimizer_cvxpy.optimize(stocks, budget=50.0)
    for portfolio in (result.risk_aware, result.profit_only):
        rm = portfolio.risk_metrics
        assert rm.volatility >= 0
        assert rm.sector_concentration >= 0
        assert rm.company_overlap >= 0
