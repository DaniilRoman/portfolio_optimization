"""Property-based invariant tests for the optimizer using hypothesis.

Tests three invariants regardless of random inputs:
  1. _build_portfolio only includes allocations with quantity >= 1.
  2. Each allocation's total_cost equals quantity * current_price.
  3. optimize never returns a portfolio whose total cost exceeds the budget.
"""
from unittest.mock import patch

from hypothesis import HealthCheck, given
from hypothesis import settings as h_settings
from hypothesis import strategies as st

from config.configuration import settings
from src.adapter.out.optimization import optimizer
from src.domain.data.data import AnalysisReport, OptimizationResult
from tests.factories import make_stock_data

# --------------------------------------------------------------------------
# Primitive strategies
# --------------------------------------------------------------------------

_price = st.floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False)
_rate = st.floats(min_value=0.0, max_value=0.05, allow_nan=False, allow_infinity=False)


# --------------------------------------------------------------------------
# Composite strategies
# --------------------------------------------------------------------------

@st.composite
def _portfolio_entries(draw: st.DrawFn) -> list[tuple[AnalysisReport, int, float]]:
    """Generate (stock, shares, price) tuples with unique ticker symbols."""
    n = draw(st.integers(min_value=1, max_value=5))
    entries = []
    for i in range(n):
        price = draw(_price)
        stock = make_stock_data(
            ticker_symbol=f"T{i:03d}",
            current_price=price,
            predict_price=draw(_price),
            dividend_yield=draw(_rate),
            expense_ratio=draw(_rate),
        )
        shares = draw(st.integers(min_value=0, max_value=5))
        entries.append((stock, shares, price))
    return entries


@st.composite
def _stocks_list(draw: st.DrawFn) -> list[AnalysisReport]:
    """Generate a list of stocks with unique ticker symbols."""
    n = draw(st.integers(min_value=1, max_value=3))
    return [
        make_stock_data(
            ticker_symbol=f"T{i:03d}",
            current_price=draw(_price),
            predict_price=draw(_price),
            dividend_yield=draw(_rate),
            expense_ratio=draw(_rate),
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# Pure-function invariants: _build_portfolio
# --------------------------------------------------------------------------

@given(_portfolio_entries())
@h_settings(max_examples=60, deadline=None)
def test_build_portfolio_quantities_positive(entries: list[tuple[AnalysisReport, int, float]]) -> None:
    """Allocations must never appear with quantity == 0."""
    stocks = [s for s, _, _ in entries]
    best_individual = [q for _, q, _ in entries]
    prices = [p for _, _, p in entries]
    predicted = [s.forecast.predict_price for s in stocks]
    divs = [s.market.dividend_yield for s in stocks]
    exps = [s.asset.expense_ratio for s in stocks]
    tickers = [s.asset.ticker_symbol for s in stocks]

    portfolio = optimizer._build_portfolio(
        best_individual, stocks, prices, predicted, divs, exps, tickers
    )
    for alloc in portfolio.allocations:
        assert alloc.quantity >= 1


@given(_portfolio_entries())
@h_settings(max_examples=60, deadline=None)
def test_build_portfolio_cost_equals_shares_times_price(entries: list[tuple[AnalysisReport, int, float]]) -> None:
    """total_cost must equal quantity * current_price for every allocation."""
    stocks = [s for s, _, _ in entries]
    best_individual = [q for _, q, _ in entries]
    prices = [p for _, _, p in entries]
    predicted = [s.forecast.predict_price for s in stocks]
    divs = [s.market.dividend_yield for s in stocks]
    exps = [s.asset.expense_ratio for s in stocks]
    tickers = [s.asset.ticker_symbol for s in stocks]

    portfolio = optimizer._build_portfolio(
        best_individual, stocks, prices, predicted, divs, exps, tickers
    )
    price_by_ticker = dict(zip(tickers, prices))
    for alloc in portfolio.allocations:
        expected_cost = alloc.quantity * price_by_ticker[alloc.asset.ticker_symbol]
        assert abs(alloc.total_cost - expected_cost) < 1e-6


# --------------------------------------------------------------------------
# End-to-end invariants: optimize
# --------------------------------------------------------------------------

def _run_optimize(stocks: list[AnalysisReport], budget: float) -> OptimizationResult:
    """Run optimizer with mocked network call and minimal GA settings."""
    etf_map = {s.asset.ticker_symbol: 0 for s in stocks}
    old_pop, old_gen = settings.ga.population, settings.ga.generations
    settings.ga.population = 8
    settings.ga.generations = 3
    try:
        with patch.object(optimizer, "__get_etf_map", return_value=etf_map):
            return optimizer.optimize(stocks, budget=budget)
    finally:
        settings.ga.population = old_pop
        settings.ga.generations = old_gen


@given(
    stocks=_stocks_list(),
    budget=st.floats(min_value=5.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
@h_settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_optimize_never_exceeds_budget(stocks: list[AnalysisReport], budget: float) -> None:
    """sum(allocation.total_cost) <= budget for both risk_aware and profit_only."""
    result = _run_optimize(stocks, budget)
    for portfolio in (result.risk_aware, result.profit_only):
        total = sum(a.total_cost for a in portfolio.allocations)
        assert total <= budget + 1e-6, f"total={total:.4f} exceeds budget={budget:.4f}"


@given(
    stocks=_stocks_list(),
    budget=st.floats(min_value=5.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
@h_settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_optimize_result_has_both_portfolios(stocks: list[AnalysisReport], budget: float) -> None:
    """OptimizationResult always exposes risk_aware and profit_only portfolios."""
    result = _run_optimize(stocks, budget)
    assert result.risk_aware is not None
    assert result.profit_only is not None
