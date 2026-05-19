"""Tests for the HRP optimizer."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pypfopt import HRPOpt

from src.adapter.out.optimization import optimizer_dispatcher, optimizer_hrp
from src.domain.data.data import OptimizationResult, Portfolio
from tests.factories import make_stock_data

# --- shared 3-asset covariance (de Prado-style example) ---
# Std devs: 10%, 20%, 30%; correlations: AB=0.8, AC=0.3, BC=0.1
_TICKERS_3 = ["A", "B", "C"]
_COV_3 = np.array(
    [
        [0.01, 0.016, 0.009],
        [0.016, 0.04, 0.006],
        [0.009, 0.006, 0.09],
    ],
    dtype=float,
)

_EMPTY_MAP = {"A": 0, "B": 0, "C": 0}


def _make_stocks_3(current_prices=(10.0, 20.0, 30.0), predicted_prices=(11.0, 22.0, 33.0)):
    return [
        make_stock_data(
            ticker_symbol=t,
            current_price=cur,
            predict_price=pred,
            standard_deviation=0.10 * (i + 1),
        )
        for i, (t, cur, pred) in enumerate(zip(_TICKERS_3, current_prices, predicted_prices))
    ]


# ---------------------------------------------------------------------------
# Test 1 (acceptance criteria): compute_weights() matches pypfopt HRPOpt reference within 1e-6
# ---------------------------------------------------------------------------


def test_hrp_3asset_matches_pypfopt_reference():
    """compute_weights() must return fractional weights identical to a direct HRPOpt call on the same data."""
    from src.adapter.out.risk.covariance import compute_covariance

    # Build a synthetic price series whose LW covariance approximates _COV_3
    rng = np.random.default_rng(42)
    L = np.linalg.cholesky(_COV_3)
    raw_returns = rng.standard_normal((300, 3)) @ L.T  # 300 daily log-returns
    prices_arr = np.exp(np.cumsum(raw_returns, axis=0)) * np.array([10.0, 20.0, 30.0])
    prices = pd.DataFrame(prices_arr, columns=_TICKERS_3)

    # Our compute_weights() path: computes LW sigma internally, then calls HRPOpt
    our_weights = optimizer_hrp.compute_weights(prices)

    # Reference: same LW sigma + same HRPOpt call
    sigma = compute_covariance(prices)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    cov_df = pd.DataFrame(sigma, index=_TICKERS_3, columns=_TICKERS_3)
    ref_hrp = HRPOpt(returns=log_returns, cov_matrix=cov_df)
    ref_weights = dict(ref_hrp.optimize(linkage_method="single"))

    for ticker in _TICKERS_3:
        assert abs(our_weights[ticker] - ref_weights[ticker]) < 1e-6, (
            f"HRP weight for {ticker}: got {our_weights[ticker]:.8f}, ref {ref_weights[ticker]:.8f}"
        )


# ---------------------------------------------------------------------------
# Test 2: compute_weights() returns valid fractional weights summing to ~1
# ---------------------------------------------------------------------------


def test_compute_weights_sums_to_one():
    n = 50
    rng = np.random.default_rng(42)
    prices_arr = np.cumprod(1 + rng.normal(0, 0.01, (n, 3)), axis=0) * np.array([10.0, 20.0, 30.0])
    prices = pd.DataFrame(prices_arr, columns=_TICKERS_3)

    weights = optimizer_hrp.compute_weights(prices)

    assert isinstance(weights, dict)
    assert set(weights.keys()) == set(_TICKERS_3)
    assert all(w >= -1e-8 for w in weights.values()), "Weights must be non-negative"
    assert abs(sum(weights.values()) - 1.0) < 1e-6, f"Weights sum to {sum(weights.values()):.6f}"


def test_compute_weights_all_positive():
    n = 100
    rng = np.random.default_rng(7)
    prices_arr = np.cumprod(1 + rng.normal(0, 0.01, (n, 3)), axis=0) * np.array([10.0, 20.0, 30.0])
    prices = pd.DataFrame(prices_arr, columns=_TICKERS_3)

    weights = optimizer_hrp.compute_weights(prices)
    assert all(v >= 0 for v in weights.values())


# ---------------------------------------------------------------------------
# Test 3: optimize() returns valid OptimizationResult respecting budget
# ---------------------------------------------------------------------------


def test_hrp_optimize_returns_optimization_result():
    stocks = _make_stocks_3()
    with patch.object(optimizer_hrp, "__get_etf_map", return_value=_EMPTY_MAP):
        result = optimizer_hrp.optimize(stocks, budget=500.0, sigma=_COV_3)

    assert isinstance(result, OptimizationResult)
    assert isinstance(result.risk_aware, Portfolio)
    assert isinstance(result.profit_only, Portfolio)


def test_hrp_optimize_budget_not_exceeded():
    stocks = _make_stocks_3()
    budget = 100.0
    with patch.object(optimizer_hrp, "__get_etf_map", return_value=_EMPTY_MAP):
        result = optimizer_hrp.optimize(stocks, budget=budget, sigma=_COV_3)

    for portfolio in (result.risk_aware, result.profit_only):
        total = sum(a.total_cost for a in portfolio.allocations)
        assert total <= budget + 1e-6, f"Total cost {total:.2f} exceeds budget {budget}"


def test_hrp_optimize_non_negative_quantities():
    stocks = _make_stocks_3()
    with patch.object(optimizer_hrp, "__get_etf_map", return_value=_EMPTY_MAP):
        result = optimizer_hrp.optimize(stocks, budget=200.0, sigma=_COV_3)

    for portfolio in (result.risk_aware, result.profit_only):
        for alloc in portfolio.allocations:
            assert alloc.quantity >= 0


# ---------------------------------------------------------------------------
# Test 4: dispatcher routes OPTIMIZER="hrp" to optimizer_hrp
# ---------------------------------------------------------------------------


def test_dispatcher_routes_hrp(monkeypatch):
    import config.configuration as cfg

    monkeypatch.setattr(cfg.settings, "OPTIMIZER", "hrp")
    monkeypatch.setattr(cfg.settings, "EXPECTED_RETURNS", "mean")

    stocks = _make_stocks_3()
    n = 50
    rng = np.random.default_rng(0)
    prices_arr = np.cumprod(1 + rng.normal(0, 0.01, (n, 3)), axis=0) * 10.0
    price_history = pd.DataFrame(prices_arr, columns=_TICKERS_3)

    with patch.object(optimizer_hrp, "__get_etf_map", return_value=_EMPTY_MAP):
        result = optimizer_dispatcher.optimize(stocks, budget=200, price_history=price_history)

    assert isinstance(result, OptimizationResult)


def test_hrp_raises_without_sigma():
    stocks = _make_stocks_3()
    with patch.object(optimizer_hrp, "__get_etf_map", return_value=_EMPTY_MAP):
        with pytest.raises(ValueError, match="sigma"):
            optimizer_hrp.optimize(stocks, budget=100.0, sigma=None)
