"""Pandas-based walk-forward P&L simulation engine.

Chosen over vectorbt to avoid a heavy numba dependency — rebalance frequency
is monthly/quarterly so a Python date loop is fast enough in practice.
"""
import numpy as np
import pandas as pd

from src.domain.data.backtest import BacktestReport

_TRADING_DAYS = 252.0


def simulate(
    prices: pd.DataFrame,
    weight_schedule: list[tuple[pd.Timestamp, dict[str, float]]],
    risk_free_rate: pd.Series,
    transaction_cost: float = 0.0005,
    slippage: float = 0.0002,
) -> BacktestReport:
    """Simulate portfolio P&L given a pre-computed weight schedule.

    Args:
        prices: DataFrame with tickers as columns, dates as index (tz-naive).
        weight_schedule: Ordered list of (rebalance_date, {ticker: weight}) pairs.
        risk_free_rate: Daily decimal Rf series (e.g. 0.0001 = 0.01% / day).
        transaction_cost: One-way cost fraction applied to portfolio turnover.
        slippage: One-way slippage fraction applied to portfolio turnover.
    """
    if prices.empty or not weight_schedule:
        return _empty_report()

    returns = prices.pct_change().fillna(0.0)
    schedule_map: dict[pd.Timestamp, dict[str, float]] = dict(weight_schedule)

    portfolio_value = 1.0
    equity_values: list[float] = [1.0]
    equity_dates: list[pd.Timestamp] = [pd.Timestamp(prices.index[0])]
    daily_port_returns: list[float] = []
    turnovers: list[float] = []

    current_weights: dict[str, float] = {}
    n_rebalances = 0
    total_cost_rate = transaction_cost + slippage

    # Apply initial weights on the very first date (no transaction cost — no prior portfolio)
    first_ts = pd.Timestamp(prices.index[0])
    if first_ts in schedule_map:
        current_weights = schedule_map[first_ts]
        turnovers.append(0.0)
        n_rebalances += 1

    for date in prices.index[1:]:
        # Rebalance if this date has a scheduled weight update
        ts = pd.Timestamp(date)
        if ts in schedule_map:
            new_weights = schedule_map[ts]
            all_tickers = set(new_weights) | set(current_weights)
            turnover = (
                sum(abs(new_weights.get(t, 0.0) - current_weights.get(t, 0.0)) for t in all_tickers)
                / 2.0
            )
            portfolio_value *= 1.0 - total_cost_rate * turnover
            current_weights = new_weights
            turnovers.append(turnover)
            n_rebalances += 1

        if not current_weights:
            equity_values.append(portfolio_value)
            equity_dates.append(ts)
            daily_port_returns.append(0.0)
            continue

        row = returns.loc[date]
        port_ret = sum(
            w * float(row.get(t, 0.0))
            for t, w in current_weights.items()
            if t in returns.columns
        )
        portfolio_value *= 1.0 + port_ret
        equity_values.append(portfolio_value)
        equity_dates.append(ts)
        daily_port_returns.append(port_ret)

    equity_curve = pd.Series(equity_values, index=pd.DatetimeIndex(equity_dates))
    ret_series = pd.Series(daily_port_returns, index=pd.DatetimeIndex(equity_dates[1:]))

    return _compute_metrics(ret_series, equity_curve, risk_free_rate, turnovers, n_rebalances)


def _compute_metrics(
    daily_returns: pd.Series,
    equity_curve: pd.Series,
    risk_free_rate: pd.Series,
    turnovers: list[float],
    n_rebalances: int,
) -> BacktestReport:
    n = len(daily_returns)
    if n == 0:
        return _empty_report()

    # Align Rf to simulation dates; forward-fill gaps (weekends / holidays)
    rf_aligned = risk_free_rate.reindex(daily_returns.index).ffill().fillna(0.0)
    daily_rf = rf_aligned / _TRADING_DAYS

    total_return = float(equity_curve.iloc[-1]) - 1.0
    n_years = n / _TRADING_DAYS
    realised_return = (1.0 + total_return) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    vol = float(daily_returns.std()) * (_TRADING_DAYS ** 0.5)

    excess = daily_returns - daily_rf
    sharpe = (float(excess.mean()) / float(daily_returns.std())) * (_TRADING_DAYS ** 0.5) if daily_returns.std() > 0 else 0.0

    neg_rets = daily_returns[daily_returns < 0]
    downside_std = float(neg_rets.std()) * (_TRADING_DAYS ** 0.5) if len(neg_rets) > 1 else 1e-9
    sortino = float(excess.mean()) * _TRADING_DAYS / downside_std

    cumulative = (1.0 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (running_max - cumulative) / running_max.replace(0, np.nan)
    max_dd = float(drawdown.max())

    calmar = realised_return / max_dd if max_dd > 1e-9 else 0.0

    threshold = float(daily_returns.quantile(0.05))
    tail = daily_returns[daily_returns <= threshold]
    cvar = float(-tail.mean()) if not tail.empty else 0.0

    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    return BacktestReport(
        realised_return=realised_return,
        volatility=vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        calmar_ratio=calmar,
        cvar_5pct=cvar,
        turnover=avg_turnover,
        n_rebalances=n_rebalances,
        equity_curve=equity_curve,
    )


def _empty_report() -> BacktestReport:
    return BacktestReport(
        realised_return=0.0,
        volatility=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown=0.0,
        calmar_ratio=0.0,
        cvar_5pct=0.0,
        turnover=0.0,
        n_rebalances=0,
        equity_curve=pd.Series(dtype=float),
    )
