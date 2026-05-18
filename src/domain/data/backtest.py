"""BacktestReport: output value object for a walk-forward backtest run."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestReport:
    realised_return: float  # annualised geometric return
    volatility: float  # annualised std of daily returns
    sharpe_ratio: float  # (R_p - R_f) / vol, annualised
    sortino_ratio: float  # mean excess return / downside-deviation, annualised
    max_drawdown: float  # max peak-to-trough decline (positive, e.g. 0.15 = 15%)
    calmar_ratio: float  # annualised return / max drawdown
    cvar_5pct: float  # expected loss in worst 5% of days (positive = loss)
    turnover: float  # mean one-way portfolio turnover per rebalance
    n_rebalances: int
    equity_curve: pd.Series  # daily portfolio value, starting at 1.0
