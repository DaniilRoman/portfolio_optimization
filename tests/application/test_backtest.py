"""Integration tests for the walk-forward backtest harness."""

import numpy as np
import pandas as pd
import pytest

from src.application.backtest import EqualWeightProvider, run_walk_forward
from src.domain.data.backtest import BacktestReport


def _make_prices(n: int = 756, n_tickers: int = 5, seed: int = 0) -> pd.DataFrame:
    """Generate ~3 years of synthetic daily prices for n_tickers."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    tickers = [f"T{i}" for i in range(n_tickers)]
    data = {}
    for t in tickers:
        log_ret = rng.normal(0.0002, 0.012, size=n)
        data[t] = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(data, index=dates)


def _make_rf(prices: pd.DataFrame, rate: float = 0.04) -> pd.Series:
    return pd.Series(rate, index=prices.index, name="Rate")


class TestRunWalkForward:
    def test_returns_backtest_report(self):
        prices = _make_prices()
        report = run_walk_forward(prices, EqualWeightProvider(), _make_rf(prices))
        assert isinstance(report, BacktestReport)

    def test_deterministic_for_same_input(self):
        prices = _make_prices()
        rf = _make_rf(prices)
        r1 = run_walk_forward(prices, EqualWeightProvider(), rf)
        r2 = run_walk_forward(prices, EqualWeightProvider(), rf)
        assert r1.sharpe_ratio == r2.sharpe_ratio
        assert r1.n_rebalances == r2.n_rebalances

    def test_metrics_within_sane_bounds_3year_5tickers(self):
        prices = _make_prices(n=756, n_tickers=5)
        report = run_walk_forward(prices, EqualWeightProvider(), _make_rf(prices))
        assert report.volatility > 0
        assert 0.0 <= report.max_drawdown <= 1.0
        assert -5.0 < report.sharpe_ratio < 10.0
        assert report.n_rebalances > 0
        assert not report.equity_curve.empty

    def test_equity_curve_starts_at_one(self):
        prices = _make_prices()
        report = run_walk_forward(prices, EqualWeightProvider(), _make_rf(prices))
        assert abs(float(report.equity_curve.iloc[0]) - 1.0) < 1e-9

    def test_raises_if_prices_too_short(self):
        prices = _make_prices(n=100)  # shorter than default train_window_days=252
        with pytest.raises(ValueError, match="shorter than train_window_days"):
            run_walk_forward(prices, EqualWeightProvider(), _make_rf(prices))

    def test_rebalance_count_increases_with_shorter_interval(self):
        prices = _make_prices()
        rf = _make_rf(prices)
        monthly = run_walk_forward(prices, EqualWeightProvider(), rf, rebalance_days=21)
        quarterly = run_walk_forward(prices, EqualWeightProvider(), rf, rebalance_days=63)
        assert monthly.n_rebalances > quarterly.n_rebalances

    def test_zero_transaction_cost_gives_higher_return(self):
        prices = _make_prices()
        rf = _make_rf(prices)
        free = run_walk_forward(prices, EqualWeightProvider(), rf, transaction_cost=0.0, slippage=0.0)
        costly = run_walk_forward(prices, EqualWeightProvider(), rf, transaction_cost=0.01, slippage=0.01)
        assert float(free.equity_curve.iloc[-1]) >= float(costly.equity_curve.iloc[-1])


class TestEqualWeightProvider:
    def test_weights_sum_to_one(self):
        prices = _make_prices(n_tickers=5)
        provider = EqualWeightProvider()
        weights = provider.compute_weights(prices)
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_all_tickers_get_equal_weight(self):
        prices = _make_prices(n_tickers=4)
        provider = EqualWeightProvider()
        weights = provider.compute_weights(prices)
        values = list(weights.values())
        assert all(abs(v - values[0]) < 1e-9 for v in values)

    def test_empty_prices_returns_empty_dict(self):
        provider = EqualWeightProvider()
        result = provider.compute_weights(pd.DataFrame())
        assert result == {}
