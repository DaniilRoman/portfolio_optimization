"""Unit tests for the pandas walk-forward P&L engine."""

import numpy as np
import pandas as pd

from src.adapter.out.backtest.pandas_engine import simulate
from src.domain.data.backtest import BacktestReport


def _make_prices(n: int = 300, tickers: list[str] | None = None, drift: float = 0.0003) -> pd.DataFrame:
    tickers = tickers or ["A", "B", "C"]
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-02", periods=n)
    data = {}
    for t in tickers:
        log_ret = rng.normal(drift, 0.01, size=n)
        prices = 100.0 * np.exp(np.cumsum(log_ret))
        data[t] = prices
    return pd.DataFrame(data, index=dates)


def _make_rf(n: int = 300, rate: float = 0.05) -> pd.Series:
    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.Series(rate, index=dates, name="Rate")


def _equal_schedule(prices: pd.DataFrame, rebalance_every: int = 21) -> list[tuple[pd.Timestamp, dict[str, float]]]:
    tickers = list(prices.columns)
    w = 1.0 / len(tickers)
    weights = {t: w for t in tickers}
    dates = prices.index[::rebalance_every]
    return [(pd.Timestamp(d), weights) for d in dates]


class TestSimulate:
    def test_returns_backtest_report(self):
        prices = _make_prices()
        report = simulate(prices, _equal_schedule(prices), _make_rf())
        assert isinstance(report, BacktestReport)

    def test_equity_curve_starts_at_one(self):
        prices = _make_prices()
        report = simulate(prices, _equal_schedule(prices), _make_rf())
        assert abs(float(report.equity_curve.iloc[0]) - 1.0) < 1e-9

    def test_equity_curve_length_matches_prices(self):
        prices = _make_prices(n=200)
        report = simulate(prices, _equal_schedule(prices), _make_rf(n=200))
        assert len(report.equity_curve) == len(prices)

    def test_deterministic_output(self):
        prices = _make_prices()
        rf = _make_rf()
        schedule = _equal_schedule(prices)
        r1 = simulate(prices, schedule, rf)
        r2 = simulate(prices, schedule, rf)
        assert r1.sharpe_ratio == r2.sharpe_ratio
        assert r1.max_drawdown == r2.max_drawdown

    def test_volatility_is_positive(self):
        prices = _make_prices()
        report = simulate(prices, _equal_schedule(prices), _make_rf())
        assert report.volatility > 0

    def test_max_drawdown_between_zero_and_one(self):
        prices = _make_prices()
        report = simulate(prices, _equal_schedule(prices), _make_rf())
        assert 0.0 <= report.max_drawdown <= 1.0

    def test_transaction_cost_reduces_terminal_value(self):
        # Alternating all-A / all-B creates 100% turnover each rebalance
        prices = _make_prices(tickers=["A", "B"])
        rf = _make_rf()
        rebalance_idx = list(range(0, len(prices), 50))
        schedule = [
            (prices.index[i], {"A": float(k % 2 == 0), "B": float(k % 2 == 1)}) for k, i in enumerate(rebalance_idx)
        ]
        free = simulate(prices, schedule, rf, transaction_cost=0.0, slippage=0.0)
        costly = simulate(prices, schedule, rf, transaction_cost=0.01, slippage=0.01)
        assert float(costly.equity_curve.iloc[-1]) < float(free.equity_curve.iloc[-1])

    def test_turnover_positive_with_changing_weights(self):
        prices = _make_prices(tickers=["A", "B"])
        # Alternating weights: first rebalance all-A, second all-B
        schedule = [
            (prices.index[10], {"A": 1.0, "B": 0.0}),
            (prices.index[50], {"A": 0.0, "B": 1.0}),
            (prices.index[100], {"A": 1.0, "B": 0.0}),
        ]
        report = simulate(prices, schedule, _make_rf())
        assert report.turnover > 0.0

    def test_empty_prices_returns_empty_report(self):
        report = simulate(pd.DataFrame(), [], pd.Series(dtype=float))
        assert report.n_rebalances == 0
        assert report.equity_curve.empty

    def test_n_rebalances_matches_schedule(self):
        prices = _make_prices(n=300)
        schedule = _equal_schedule(prices, rebalance_every=50)
        report = simulate(prices, schedule, _make_rf(n=300))
        assert report.n_rebalances == len(schedule)

    def test_sharpe_ratio_finite(self):
        prices = _make_prices()
        report = simulate(prices, _equal_schedule(prices), _make_rf())
        assert np.isfinite(report.sharpe_ratio)

    def test_cvar_positive(self):
        prices = _make_prices()
        report = simulate(prices, _equal_schedule(prices), _make_rf())
        assert report.cvar_5pct >= 0.0


class TestMetricSanity:
    def test_upward_trending_prices_positive_return(self):
        prices = _make_prices(drift=0.001)  # strong positive drift
        report = simulate(prices, _equal_schedule(prices), _make_rf())
        assert report.realised_return > 0

    def test_calmar_is_return_over_drawdown(self):
        prices = _make_prices()
        report = simulate(prices, _equal_schedule(prices), _make_rf())
        if report.max_drawdown > 1e-9:
            expected = report.realised_return / report.max_drawdown
            assert abs(report.calmar_ratio - expected) < 1e-9
