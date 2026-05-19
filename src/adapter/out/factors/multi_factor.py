"""Cross-sectional five-factor scoring: value, quality, momentum, low_vol, size."""

from __future__ import annotations

import pandas as pd

from src.domain.data.data import AnalysisReport

_FACTORS = ["value", "quality", "momentum", "low_vol", "size"]


def score(reports: list[AnalysisReport]) -> pd.DataFrame:
    """Score each ticker on five factors, z-score cross-sectionally, and average into 'composite'.

    Args:
        reports: List of AnalysisReport objects (one per ticker in the current universe).

    Returns:
        DataFrame indexed by ticker symbol with columns:
        [value, quality, momentum, low_vol, size, composite].
        All factor columns are cross-sectionally z-scored (mean=0, std=1 across tickers).
        'composite' is the equal-weight mean of the z-scored factors.
    """
    if not reports:
        return pd.DataFrame(columns=_FACTORS + ["composite"])

    tickers = [r.asset.ticker_symbol for r in reports]

    raw: dict[str, list[float]] = {
        "value": _value_scores(reports),
        "quality": _quality_scores(reports),
        "momentum": _momentum_scores(reports),
        "low_vol": _low_vol_scores(reports),
        "size": _size_scores(reports),
    }

    df = pd.DataFrame(raw, index=tickers)

    for col in _FACTORS:
        std = float(df[col].std())
        mean = float(df[col].mean())
        if std > 0:
            df[col] = (df[col] - mean) / std
        else:
            df[col] = 0.0

    df["composite"] = df[_FACTORS].mean(axis=1)
    return df


def screen(reports: list[AnalysisReport], bottom_percentile: float) -> list[AnalysisReport]:
    """Drop the bottom_percentile fraction of tickers by composite factor score.

    Args:
        reports: Full universe of AnalysisReport objects.
        bottom_percentile: Fraction in [0, 1) to drop (e.g. 0.25 drops bottom 25%).

    Returns:
        Filtered list with low-scoring tickers removed.
    """
    if bottom_percentile <= 0 or len(reports) < 2:
        return reports

    factor_df = score(reports)
    cutoff = float(factor_df["composite"].quantile(bottom_percentile))
    return [r for r in reports if factor_df.loc[r.asset.ticker_symbol, "composite"] >= cutoff]


def _value_scores(reports: list[AnalysisReport]) -> list[float]:
    """E/P ratio: higher earnings yield = better value. ETFs with no EPS data score 0."""
    scores = []
    for r in reports:
        price = r.market.current_price
        if price <= 0:
            scores.append(0.0)
            continue
        trailing = r.profitability_data.trailing_eps
        forward = r.profitability_data.forward_eps
        # Use average of trailing and forward E/P; treat negatives as 0
        trailing_ep = max(trailing, 0.0) / price
        forward_ep = max(forward, 0.0) / price
        scores.append((trailing_ep + forward_ep) / 2.0)
    return scores


def _quality_scores(reports: list[AnalysisReport]) -> list[float]:
    """Average of operating margin and EBITDA margin; negative margins treated as 0."""
    return [
        (max(r.profitability_data.operating_margins, 0.0) + max(r.profitability_data.ebitda_margins, 0.0)) / 2.0
        for r in reports
    ]


def _momentum_scores(reports: list[AnalysisReport]) -> list[float]:
    """12-1 month price return stored on AnalysisReport.momentum_12_1."""
    return [r.momentum_12_1 for r in reports]


def _low_vol_scores(reports: list[AnalysisReport]) -> list[float]:
    """Inverse annualised standard deviation: lower volatility = higher score."""
    scores = []
    for r in reports:
        std = r.market.standard_deviation
        scores.append(1.0 / std if std > 0 else 0.0)
    return scores


def _size_scores(reports: list[AnalysisReport]) -> list[float]:
    """Negative AUM: smaller fund / stock = higher size premium."""
    return [-r.asset.assets_under_management for r in reports]
