"""Tests for the Black-Litterman expected-returns adapter.

All tests are fully offline (no network calls).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.adapter.out.expected_returns.black_litterman import compute_bl_mu

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_TICKERS = ["A", "B", "C"]

# Simple diagonal covariance (no correlation) so equilibrium prior is easy to reason about
_COV = np.diag([0.04, 0.09, 0.01])  # annualised variances: 20%, 30%, 10% vol

_MARKET_CAPS = {"A": 1_000_000.0, "B": 2_000_000.0, "C": 500_000.0}


# ---------------------------------------------------------------------------
# Test 1 (acceptance criteria): zero views → posterior equals equilibrium prior
# ---------------------------------------------------------------------------


def test_zero_views_returns_equilibrium_prior():
    """With no views the BL posterior must equal the market equilibrium prior (within 1e-9)."""
    from pypfopt.black_litterman import market_implied_prior_returns

    cov_df = pd.DataFrame(_COV, index=_TICKERS, columns=_TICKERS)
    total_cap = sum(_MARKET_CAPS.values())
    mkt_weights = pd.Series({t: _MARKET_CAPS[t] / total_cap for t in _TICKERS})

    risk_aversion = 2.5
    expected_prior = market_implied_prior_returns(mkt_weights, risk_aversion, cov_df)

    bl_result = compute_bl_mu(
        tickers=_TICKERS,
        sigma=_COV,
        market_caps=_MARKET_CAPS,
        views={},  # no views
        uncertainties={},
        risk_aversion=risk_aversion,
    )

    for ticker in _TICKERS:
        assert abs(float(bl_result[ticker]) - float(expected_prior[ticker])) < 1e-9, (
            f"Ticker {ticker}: posterior {bl_result[ticker]:.8f} ≠ prior {expected_prior[ticker]:.8f}"
        )


# ---------------------------------------------------------------------------
# Test 2: single high-confidence view shifts posterior toward the view
# ---------------------------------------------------------------------------


def test_high_confidence_view_moves_posterior_toward_view():
    """A high-confidence absolute view on asset A should pull posterior μ_A toward the view."""
    view_return = 0.30  # 30% annual simple return view on asset A
    view_log = math.log(1 + view_return)  # convert to log return for the input interface

    bl_no_view = compute_bl_mu(
        tickers=_TICKERS,
        sigma=_COV,
        market_caps=_MARKET_CAPS,
        views={},
        uncertainties={},
    )
    bl_with_view = compute_bl_mu(
        tickers=_TICKERS,
        sigma=_COV,
        market_caps=_MARKET_CAPS,
        views={"A": view_log},
        uncertainties={"A": 0.01},  # very low uncertainty → very high confidence
    )

    prior_A = float(bl_no_view["A"])
    posterior_A = float(bl_with_view["A"])

    # Posterior should be between the prior and the view (or closer to view for high confidence)
    assert posterior_A > prior_A or abs(posterior_A - view_return) < abs(prior_A - view_return), (
        f"High-confidence view did not move posterior toward view: "
        f"prior={prior_A:.4f}, posterior={posterior_A:.4f}, view={view_return:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 3: result is a pd.Series with correct index and finite values
# ---------------------------------------------------------------------------


def test_bl_returns_series_with_correct_index():
    views = {"A": 0.10, "B": 0.05}
    uncertainties = {"A": 0.5, "B": 0.8}

    result = compute_bl_mu(
        tickers=_TICKERS,
        sigma=_COV,
        market_caps=_MARKET_CAPS,
        views=views,
        uncertainties=uncertainties,
    )

    assert isinstance(result, pd.Series), f"Expected pd.Series, got {type(result)}"
    for ticker in _TICKERS:
        assert ticker in result.index, f"Ticker {ticker} missing from BL result"
        assert math.isfinite(float(result[ticker])), f"Non-finite return for {ticker}: {result[ticker]}"


# ---------------------------------------------------------------------------
# Test 4: empty tickers → empty series
# ---------------------------------------------------------------------------


def test_empty_tickers_returns_empty_series():
    result = compute_bl_mu(
        tickers=[],
        sigma=np.array([]).reshape(0, 0),
        market_caps={},
        views={},
        uncertainties={},
    )
    assert isinstance(result, pd.Series)
    assert len(result) == 0
