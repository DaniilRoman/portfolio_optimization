"""Black-Litterman posterior expected returns using pypfopt; blends CAPM equilibrium with GARCH/Prophet views."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pypfopt.black_litterman import BlackLittermanModel, market_implied_prior_returns


def compute_bl_mu(
    tickers: list[str],
    sigma: np.ndarray,
    market_caps: dict[str, float],
    views: dict[str, float],
    uncertainties: dict[str, float],
    risk_aversion: float = 2.5,
) -> pd.Series:
    """Compute Black-Litterman posterior expected returns (simple annual, decimal).

    Args:
        tickers: Ordered list of asset tickers matching sigma's axis order.
        sigma: Annualised covariance matrix (n × n), e.g. from Ledoit-Wolf shrinkage.
        market_caps: Total assets / market cap per ticker (used for equilibrium weights).
        views: Per-ticker annualised log-return views from GARCH/Prophet.
        uncertainties: Per-ticker forecast uncertainty (prediction_uncertainty field).
        risk_aversion: Market risk-aversion coefficient δ for equilibrium prior.

    Returns:
        pd.Series indexed by ticker with posterior expected simple annual returns.
        Falls back to market-implied prior when views is empty.
    """
    if not tickers:
        return pd.Series(dtype=float)

    cov_df = pd.DataFrame(sigma, index=tickers, columns=tickers)
    total_cap = sum(market_caps.get(t, 1.0) for t in tickers)
    mkt_weights = pd.Series({t: market_caps.get(t, 1.0) / total_cap for t in tickers})

    if not views:
        # No views → return equilibrium prior directly
        return market_implied_prior_returns(mkt_weights, risk_aversion, cov_df)

    # Only include views for tickers that are in our universe
    absolute_views = {t: math.exp(v) - 1.0 for t, v in views.items() if t in tickers}

    if not absolute_views:
        return market_implied_prior_returns(mkt_weights, risk_aversion, cov_df)

    # Idzorek confidences: lower uncertainty → higher confidence (clamped to [0.01, 0.99])
    # Must be a 1D numpy array aligned with the views order (same as absolute_views insertion order)
    max_unc = max(uncertainties.values()) if uncertainties else 1.0
    view_conf_list = [
        max(0.01, min(0.99, 1.0 - uncertainties.get(t, max_unc) / max(max_unc, 1e-9))) for t in absolute_views
    ]
    view_confidences = np.array(view_conf_list, dtype=float)

    bl = BlackLittermanModel(
        cov_df,
        pi="market",
        market_caps=mkt_weights,
        absolute_views=absolute_views,
        omega="idzorek",
        view_confidences=view_confidences,
        risk_aversion=risk_aversion,
    )
    return bl.bl_returns()
