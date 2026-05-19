"""Tests for the multi-factor scoring module.

All tests use fully synthetic 4-asset toy universes with known expected rankings.
"""

from __future__ import annotations

import pandas as pd

from src.adapter.out.factors.multi_factor import (
    _FACTORS,
    _low_vol_scores,
    _momentum_scores,
    _quality_scores,
    _value_scores,
    score,
    screen,
)
from tests.factories import make_stock_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_4_stocks():
    """Return 4 assets with deliberately distinct characteristics for factor sign tests."""
    return [
        make_stock_data(  # HIGH quality, HIGH value, HIGH momentum, LOW vol, SMALL
            ticker_symbol="GOOD",
            current_price=10.0,
            predict_price=11.0,
            trailing_eps=2.0,  # E/P = 0.20
            forward_eps=2.5,  # E/P = 0.25
            operating_margins=0.40,
            ebitda_margins=0.35,
            standard_deviation=0.10,
            assets_under_management=100_000_000.0,
            momentum_12_1=0.20,
        ),
        make_stock_data(  # MEDIUM
            ticker_symbol="MED",
            current_price=20.0,
            predict_price=21.0,
            trailing_eps=1.0,  # E/P = 0.05
            forward_eps=1.2,  # E/P = 0.06
            operating_margins=0.20,
            ebitda_margins=0.18,
            standard_deviation=0.15,
            assets_under_management=500_000_000.0,
            momentum_12_1=0.05,
        ),
        make_stock_data(  # LOW quality, LOW value, LOW momentum, HIGH vol, LARGE
            ticker_symbol="BAD",
            current_price=30.0,
            predict_price=29.0,
            trailing_eps=0.5,  # E/P = 0.017
            forward_eps=0.3,  # E/P = 0.010
            operating_margins=0.05,
            ebitda_margins=0.04,
            standard_deviation=0.35,
            assets_under_management=5_000_000_000.0,
            momentum_12_1=-0.10,
        ),
        make_stock_data(  # ZERO EPS (typical ETF — value factor should not crash)
            ticker_symbol="ETF",
            current_price=15.0,
            predict_price=15.5,
            trailing_eps=0.0,
            forward_eps=0.0,
            operating_margins=0.0,
            ebitda_margins=0.0,
            standard_deviation=0.12,
            assets_under_management=2_000_000_000.0,
            momentum_12_1=0.03,
        ),
    ]


# ---------------------------------------------------------------------------
# Test 1: value factor — high E/P scores higher than low E/P
# ---------------------------------------------------------------------------


def test_value_high_ep_scores_higher():
    reports = _make_4_stocks()
    v_scores = _value_scores(reports)
    ticker_score = dict(zip(["GOOD", "MED", "BAD", "ETF"], v_scores))

    assert ticker_score["GOOD"] > ticker_score["BAD"], (
        f"GOOD E/P ({ticker_score['GOOD']:.4f}) should exceed BAD E/P ({ticker_score['BAD']:.4f})"
    )
    assert ticker_score["ETF"] == 0.0, "ETF with zero EPS should score 0 on value"


# ---------------------------------------------------------------------------
# Test 2: quality factor — high-margin asset scores higher
# ---------------------------------------------------------------------------


def test_quality_high_margin_scores_higher():
    reports = _make_4_stocks()
    q_scores = _quality_scores(reports)
    ticker_score = dict(zip(["GOOD", "MED", "BAD", "ETF"], q_scores))

    assert ticker_score["GOOD"] > ticker_score["MED"] > ticker_score["BAD"], (
        f"Quality order wrong: GOOD={ticker_score['GOOD']:.3f}, "
        f"MED={ticker_score['MED']:.3f}, BAD={ticker_score['BAD']:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 3: momentum factor — higher 12-1 return scores higher
# ---------------------------------------------------------------------------


def test_momentum_higher_return_scores_higher():
    reports = _make_4_stocks()
    m_scores = _momentum_scores(reports)
    ticker_score = dict(zip(["GOOD", "MED", "BAD", "ETF"], m_scores))

    assert ticker_score["GOOD"] > ticker_score["BAD"], (
        f"Momentum: GOOD ({ticker_score['GOOD']:.3f}) should exceed BAD ({ticker_score['BAD']:.3f})"
    )
    assert ticker_score["BAD"] < 0, "Negative 12-1 momentum should be negative"


# ---------------------------------------------------------------------------
# Test 4: low_vol factor — lower std dev → higher score
# ---------------------------------------------------------------------------


def test_low_vol_lower_stddev_scores_higher():
    reports = _make_4_stocks()
    lv_scores = _low_vol_scores(reports)
    ticker_score = dict(zip(["GOOD", "MED", "BAD", "ETF"], lv_scores))

    assert ticker_score["GOOD"] > ticker_score["BAD"], (
        f"Low-vol: GOOD ({ticker_score['GOOD']:.3f}) should exceed BAD ({ticker_score['BAD']:.3f})"
    )


# ---------------------------------------------------------------------------
# Test 5: composite is the equal-weight mean of z-scored factors
# ---------------------------------------------------------------------------


def test_composite_equals_mean_of_zscored_factors():
    reports = _make_4_stocks()
    df = score(reports)

    assert "composite" in df.columns
    computed_composite = df[_FACTORS].mean(axis=1)
    pd.testing.assert_series_equal(df["composite"], computed_composite, check_names=False, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 6: zero EPS (ETF case) does not crash; value score handled gracefully
# ---------------------------------------------------------------------------


def test_zero_eps_does_not_crash():
    reports = _make_4_stocks()
    df = score(reports)

    assert "ETF" in df.index
    assert isinstance(float(df.loc["ETF", "composite"]), float)
    # All factor columns are finite
    for col in _FACTORS:
        val = float(df.loc["ETF", col])
        assert val == val, f"NaN in factor {col} for ETF ticker"


# ---------------------------------------------------------------------------
# Test 7: score() returns DataFrame with correct shape and columns
# ---------------------------------------------------------------------------


def test_score_returns_correct_shape():
    reports = _make_4_stocks()
    df = score(reports)

    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == ["GOOD", "MED", "BAD", "ETF"]
    for col in _FACTORS + ["composite"]:
        assert col in df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Test 8: screen() drops bottom percentile
# ---------------------------------------------------------------------------


def test_screen_drops_bottom_quarter():
    reports = _make_4_stocks()
    screened = screen(reports, bottom_percentile=0.25)

    # With 4 assets and 25% cutoff, at least one should be removed
    assert len(screened) < len(reports), "screen() should remove at least one ticker"
    assert len(screened) >= 1, "screen() should keep at least one ticker"


def test_screen_zero_percentile_returns_all():
    reports = _make_4_stocks()
    screened = screen(reports, bottom_percentile=0.0)
    assert screened is reports  # same object (no filtering)


# ---------------------------------------------------------------------------
# Test 9: score() on empty list returns empty DataFrame
# ---------------------------------------------------------------------------


def test_score_empty_list():
    df = score([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
