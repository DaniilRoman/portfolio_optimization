import re
from unittest.mock import patch

import numpy as np

from src.adapter.out.optimization import optimizer_cvxpy
from src.logic.data.data import ProfitabilityData, StockData


def create_test_stock_data():
    stocks = []

    test_data = [
        ("TECH", "Technology Concentrated ETF", 15.0, 18.0, 0.25, 0.01, 0.005),
        ("BLEND", "Blended Market ETF", 25.0, 26.5, 0.15, 0.02, 0.01),
        ("DIV", "Diversified Low Vol ETF", 20.0, 20.8, 0.10, 0.03, 0.02),
        ("CONC", "Highly Concentrated ETF", 12.0, 15.0, 0.30, 0.005, 0.003),
        ("LOSS", "Losing ETF", 18.0, 17.5, 0.20, 0.015, 0.015),
    ]

    sector_allocations = [
        {"Technology": 0.90, "Healthcare": 0.10},
        {"Technology": 0.25, "Healthcare": 0.20, "Financials": 0.20, "Consumer": 0.20, "Industrials": 0.15},
        {"Technology": 0.15, "Healthcare": 0.15, "Financials": 0.15, "Consumer": 0.15, "Industrials": 0.15, "Utilities": 0.10, "Energy": 0.10, "Materials": 0.05},
        {"Technology": 0.95, "Healthcare": 0.05},
        {"Technology": 0.60, "Healthcare": 0.40},
    ]

    top_holdings_list = [
        np.array([["Apple Inc", 0.25], ["Microsoft Corp", 0.20], ["NVIDIA Corp", 0.15]]),
        np.array([["Apple Inc", 0.08], ["Microsoft Corp", 0.07], ["Amazon.com Inc", 0.06], ["Alphabet Inc", 0.05], ["Tesla Inc", 0.04], ["Meta Platforms", 0.03]]),
        np.array([["Apple Inc", 0.04], ["Microsoft Corp", 0.04], ["Amazon.com Inc", 0.03], ["Alphabet Inc", 0.03], ["Johnson & Johnson", 0.03], ["JPMorgan Chase", 0.03], ["Procter & Gamble", 0.03], ["Exxon Mobil", 0.03], ["Walmart", 0.03]]),
        np.array([["Tesla Inc", 0.50], ["Rivian", 0.20], ["Lucid", 0.15]]),
        np.array([["Apple Inc", 0.15], ["Microsoft Corp", 0.12], ["Amazon.com Inc", 0.10]]),
    ]

    for idx, (ticker, name, current_price, predict_price, std_dev, dividend_yield, expense_ratio) in enumerate(test_data):
        profitability = ProfitabilityData(
            trailing_eps=5.0,
            forward_eps=5.5,
            netIncome_to_common=1000000.0,
            ebitda_margins=0.3,
            operating_margins=0.25,
        )

        beta = 1.0 + (std_dev - 0.15) * 2

        stocks.append(
            StockData(
                ticker_symbol=ticker,
                stock_name=name,
                currency="USD",
                current_price=current_price,
                predict_price=predict_price,
                two_year_file_name="",
                five_year_file_name="",
                is_stock_growing=predict_price > current_price,
                industry="ETF",
                profitability_data=profitability,
                beta=beta,
                standard_deviation=std_dev,
                dividend_yield=dividend_yield,
                top_holdings=top_holdings_list[idx],
                sector_allocation=sector_allocations[idx],
                average_daily_volume=1000000.0,
                assets_under_management=1000000000.0,
                expense_ratio=expense_ratio,
                description="Test ETF",
            )
        )

    return stocks


def _extract_section(result: str, heading: str, next_heading: str | None = None) -> str:
    section = result.split(heading, 1)[1]
    if next_heading is not None:
        section = section.split(next_heading, 1)[0]
    return section


def _sum_costs(section: str) -> float:
    total = 0.0
    for line in section.splitlines():
        if line.strip().startswith("Cost:"):
            total += float(line.split("€", 1)[1])
    return total


def _extract_share_counts(section: str) -> list[int]:
    counts = []
    for line in section.splitlines():
        match = re.search(r":\s*(\d+) shares", line)
        if match:
            counts.append(int(match.group(1)))
    return counts


def test_cvxpy_optimizer_interface_and_budget():
    stocks = create_test_stock_data()
    test_etf_map = {ticker: 0 for ticker in ["TECH", "BLEND", "DIV", "CONC", "LOSS"]}

    with patch.object(optimizer_cvxpy, "__get_etf_map", return_value=test_etf_map):
        result = optimizer_cvxpy.optimize(stocks, budget=50.0)

    assert "📊 *Portfolio Optimization Results*" in result
    assert "⚠️ **Risk-Aware Optimization**" in result
    assert "📈 **Profit-Only Optimization**" in result

    risk_section = _extract_section(result, "⚠️ **Risk-Aware Optimization**", "📈 **Profit-Only Optimization**")
    profit_section = _extract_section(result, "📈 **Profit-Only Optimization**")

    assert _sum_costs(risk_section) <= 50.0 + 1e-6
    assert _sum_costs(profit_section) <= 50.0 + 1e-6

    assert all(count >= 0 for count in _extract_share_counts(risk_section))
    assert all(count >= 0 for count in _extract_share_counts(profit_section))
