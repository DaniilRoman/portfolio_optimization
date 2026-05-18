"""Convex portfolio optimizer using CVXPY; solves a risk-adjusted allocation problem subject to budget and concentration constraints."""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from config.configuration import settings
from src.domain.data.data import AnalysisReport, OptimizationResult

from .optimizer import (
    __get_etf_map,
    _build_portfolio,
    _calculate_max_shares,
    _create_ownership_weights,
    _prepare_stock_data,
)

StockData = AnalysisReport  # local alias for annotations in this module


def _build_expected_net_profit_per_share(
    stocks: list[StockData],
    current_prices: list[float],
    predicted_prices: list[float],
    dividend_yields: list[float],
    expense_ratios: list[float],
    ownership_weights: list[float],
) -> np.ndarray:
    mu = np.zeros(len(stocks), dtype=float)

    for i, _stock in enumerate(stocks):
        current_price = float(current_prices[i])
        predicted_price = float(predicted_prices[i])
        dividend_yield = float(dividend_yields[i])
        expense_ratio = float(expense_ratios[i])
        ownership_weight = float(ownership_weights[i])
        expected_gain_per_share = predicted_price - current_price
        dividend_income_per_share = current_price * dividend_yield
        expense_fee_per_share = current_price * expense_ratio
        net_profit_per_share = expected_gain_per_share + dividend_income_per_share - expense_fee_per_share
        mu[i] = net_profit_per_share * ownership_weight

    return mu


def _build_sector_matrix(stocks: list[StockData]) -> tuple[np.ndarray, list[str]]:
    sectors = sorted({sector for stock in stocks for sector in stock.sector_allocation.keys()})
    sector_index = {sector: idx for idx, sector in enumerate(sectors)}
    matrix = np.zeros((len(sectors), len(stocks)), dtype=float)

    for i, _stock in enumerate(stocks):
        for sector, allocation in _stock.sector_allocation.items():
            matrix[sector_index[sector], i] = float(allocation)

    return matrix, sectors


def _build_company_matrix(stocks: list[StockData]) -> tuple[np.ndarray, list[str]]:
    company_set: set[str] = set()
    for _stock in stocks:
        for holding in _stock.top_holdings:
            try:
                company_set.add(str(holding[0]))
            except Exception:
                continue

    companies: list[str] = sorted(company_set)
    company_index = {company: idx for idx, company in enumerate(companies)}
    matrix = np.zeros((len(companies), len(stocks)), dtype=float)

    for i, _stock in enumerate(stocks):
        for holding in _stock.top_holdings:
            try:
                company = str(holding[0])
                weight = float(holding[1])
            except Exception:
                continue
            matrix[company_index[company], i] += weight

    return matrix, companies


def _repair_solution(
    values: np.ndarray,
    current_prices: np.ndarray,
    max_shares: np.ndarray,
    budget: float,
    scores: np.ndarray,
) -> list[int]:
    rounded = np.rint(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)).astype(int)
    rounded = np.clip(rounded, 0, max_shares.astype(int))

    cost = float(current_prices @ rounded)
    if cost <= budget:
        return rounded.tolist()

    removable = [i for i in range(len(rounded)) if rounded[i] > 0 and current_prices[i] > 0]
    removable.sort(key=lambda i: (scores[i], current_prices[i]))

    while cost > budget and removable:
        i = removable[0]
        rounded[i] -= 1
        cost -= float(current_prices[i])
        if rounded[i] <= 0:
            removable.pop(0)

    return rounded.tolist()


def _run_cvxpy_optimization_with_map(
    stocks: list[StockData],
    budget: float,
    max_per_etf_budget: float,
    etf_map: dict[str, int],
    include_risk: bool = True,
) -> tuple[list[int], list[StockData], list[float], list[float], list[float], list[float], list[str]]:
    tickers, current_prices, predicted_prices, dividend_yields, expense_ratios = _prepare_stock_data(stocks)
    max_shares_per_stock = np.asarray(_calculate_max_shares(current_prices, max_per_etf_budget), dtype=int)
    ownership_weights = _create_ownership_weights(tickers, etf_map)

    current_prices_arr = np.asarray(current_prices, dtype=float)
    predicted_prices_arr = np.asarray(predicted_prices, dtype=float)
    dividend_yields_arr = np.asarray(dividend_yields, dtype=float)
    expense_ratios_arr = np.asarray(expense_ratios, dtype=float)

    mu = _build_expected_net_profit_per_share(
        stocks,
        current_prices_arr.tolist(),
        predicted_prices_arr.tolist(),
        dividend_yields_arr.tolist(),
        expense_ratios_arr.tolist(),
        ownership_weights,
    )

    A_sector, _sectors = _build_sector_matrix(stocks)
    A_company, _companies = _build_company_matrix(stocks)

    def _stock_volatility(stock: StockData, current_price: float) -> float:
        forecast_vol = stock.forecast.forecast_volatility
        if forecast_vol <= 0:
            pred_unc = stock.forecast.prediction_uncertainty
            if current_price > 0 and pred_unc > 0:
                forecast_vol = pred_unc / current_price
        return max(
            stock.market.standard_deviation,
            forecast_vol,
            stock.market.beta * 0.15,
        )

    volatility_proxy = np.asarray([
        _stock_volatility(_stock, float(current_prices_arr[i]))
        for i, _stock in enumerate(stocks)
    ], dtype=float)

    x = cp.Variable(len(stocks), integer=True)
    value_vector = cp.multiply(current_prices_arr, x)
    cost = cp.sum(value_vector)
    profit = mu @ x

    sector_exposure = A_sector @ value_vector if A_sector.size else None
    company_exposure = A_company @ value_vector if A_company.size else None

    constraints: list[cp.Constraint] = [x >= 0, x <= max_shares_per_stock, cost <= budget + 1e-6]
    if sector_exposure is not None:
        constraints.extend([sector_exposure <= settings.ga.max_sector_concentration * budget])
    if company_exposure is not None:
        constraints.extend([company_exposure <= settings.cvxpy.company_max_exposure * budget])

    risk_penalty = 0
    if include_risk:
        risk_penalty = (
            0.35 * cp.sum_squares(value_vector / max(budget, 1e-9))
            + 0.25 * cp.sum_squares(cp.multiply(volatility_proxy, value_vector) / max(budget, 1e-9))
        )
        if sector_exposure is not None and sector_exposure.size > 0:
            risk_penalty += 0.20 * cp.sum_squares(sector_exposure / max(budget, 1e-9))
        if company_exposure is not None and company_exposure.size > 0:
            risk_penalty += 0.20 * cp.sum_squares(company_exposure / max(budget, 1e-9))

    objective = profit - (settings.cvxpy.risk_gamma * risk_penalty if include_risk else 0)

    try:
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve(solver=cp.ECOS_BB, verbose=False)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise cp.error.SolverError(problem.status)
    except Exception:
        relaxed_x = cp.Variable(len(stocks))
        relaxed_value_vector = cp.multiply(current_prices_arr, relaxed_x)
        relaxed_cost = cp.sum(relaxed_value_vector)
        relaxed_profit = mu @ relaxed_x
        relaxed_constraints: list[cp.Constraint] = [relaxed_x >= 0, relaxed_x <= max_shares_per_stock, relaxed_cost <= budget + 1e-6]
        if A_sector.size:
            relaxed_constraints.append(A_sector @ relaxed_value_vector <= settings.ga.max_sector_concentration * budget)
        if A_company.size:
            relaxed_constraints.append(A_company @ relaxed_value_vector <= settings.cvxpy.company_max_exposure * budget)

        relaxed_risk_penalty = 0
        if include_risk:
            relaxed_sector = A_sector @ relaxed_value_vector if A_sector.size else None
            relaxed_company = A_company @ relaxed_value_vector if A_company.size else None
            relaxed_risk_penalty = (
                0.35 * cp.sum_squares(relaxed_value_vector / max(budget, 1e-9))
                + 0.25 * cp.sum_squares(cp.multiply(volatility_proxy, relaxed_value_vector) / max(budget, 1e-9))
            )
            if relaxed_sector is not None and relaxed_sector.size > 0:
                relaxed_risk_penalty += 0.20 * cp.sum_squares(relaxed_sector / max(budget, 1e-9))
            if relaxed_company is not None and relaxed_company.size > 0:
                relaxed_risk_penalty += 0.20 * cp.sum_squares(relaxed_company / max(budget, 1e-9))

        relaxed_objective = relaxed_profit - (settings.cvxpy.risk_gamma * relaxed_risk_penalty if include_risk else 0)
        relaxed_problem = cp.Problem(cp.Maximize(relaxed_objective), relaxed_constraints)
        relaxed_problem.solve(solver=cp.OSQP, verbose=False)
        values = np.asarray(relaxed_x.value if relaxed_x.value is not None else np.zeros(len(stocks)))
    else:
        values = np.asarray(x.value if x.value is not None else np.zeros(len(stocks)))

    scores = np.asarray(mu / np.maximum(current_prices_arr, 1e-9), dtype=float)
    best_individual = _repair_solution(values, current_prices_arr, max_shares_per_stock, budget, scores)

    return best_individual, stocks, current_prices, predicted_prices, dividend_yields, expense_ratios, tickers


def optimize(stocks: list[StockData], budget: float = 50.0, max_per_etf_budget: float = 50.0) -> OptimizationResult:
    if max_per_etf_budget is None:
        max_per_etf_budget = min(50.0, budget / 2)
    elif max_per_etf_budget > budget:
        max_per_etf_budget = budget

    etf_map = __get_etf_map()

    risk_individual, r_stocks, r_prices, r_predicted, r_div, r_exp, r_tickers = _run_cvxpy_optimization_with_map(
        stocks, budget, max_per_etf_budget, etf_map, include_risk=True
    )
    profit_individual, p_stocks, p_prices, p_predicted, p_div, p_exp, p_tickers = _run_cvxpy_optimization_with_map(
        stocks, budget, max_per_etf_budget, etf_map, include_risk=False
    )

    return OptimizationResult(
        risk_aware=_build_portfolio(risk_individual, r_stocks, r_prices, r_predicted, r_div, r_exp, r_tickers),
        profit_only=_build_portfolio(profit_individual, p_stocks, p_prices, p_predicted, p_div, p_exp, p_tickers),
    )
