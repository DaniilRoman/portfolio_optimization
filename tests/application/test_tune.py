"""Unit tests for the Optuna-driven hyperparameter sweep."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.tune import (
    HRPProvider,
    MinVarianceProvider,
    TuneConfig,
    build_provider,
    run_tune,
    split_train_holdout,
)


def _make_prices(n: int = 800, n_tickers: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n)
    tickers = [f"T{i}" for i in range(n_tickers)]
    data = {}
    for t in tickers:
        log_ret = rng.normal(0.0003, 0.013, size=n)
        data[t] = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(data, index=dates)


def _make_rf(prices: pd.DataFrame, rate: float = 0.04) -> pd.Series:
    return pd.Series(rate, index=prices.index, name="Rate")


class TestHRPProvider:
    def test_weights_sum_to_one(self):
        prices = _make_prices(n=400, n_tickers=5)
        weights = HRPProvider(linkage_method="single").compute_weights(prices)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(0.0 <= w <= 1.0 for w in weights.values())

    def test_different_linkages_yield_different_weights(self):
        prices = _make_prices(n=400, n_tickers=5, seed=1)
        single = HRPProvider("single").compute_weights(prices)
        ward = HRPProvider("ward").compute_weights(prices)
        assert single != ward

    def test_single_ticker_returns_empty(self):
        prices = _make_prices(n=200, n_tickers=1)
        assert HRPProvider("single").compute_weights(prices) == {}


class TestMinVarianceProvider:
    def test_weights_sum_to_one_and_respect_cap(self):
        prices = _make_prices(n=400, n_tickers=5)
        provider = MinVarianceProvider(max_weight=0.30)
        weights = provider.compute_weights(prices)
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(w <= 0.30 + 1e-6 for w in weights.values())


class TestBuildProvider:
    def test_builds_hrp(self):
        provider = build_provider({"optimizer": "hrp", "hrp_linkage": "average"})
        assert isinstance(provider, HRPProvider)
        assert provider.linkage_method == "average"

    def test_builds_min_var(self):
        provider = build_provider({"optimizer": "min_var", "min_var_max_weight": 0.15})
        assert isinstance(provider, MinVarianceProvider)
        assert provider.max_weight == 0.15

    def test_unknown_optimizer_raises(self):
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_provider({"optimizer": "bogus"})


class TestSplitTrainHoldout:
    def test_train_ends_holdout_days_before_end(self):
        prices = _make_prices(n=800)
        train, holdout = split_train_holdout(prices, holdout_days=100, train_window_days=252)
        assert len(train) == 700
        assert len(holdout) == 100 + 252
        assert train.index[-1] < holdout.index[-1]

    def test_holdout_keeps_context_window(self):
        prices = _make_prices(n=800)
        _, holdout = split_train_holdout(prices, holdout_days=100, train_window_days=252)
        assert holdout.index[-1] == prices.index[-1]

    def test_raises_when_too_short(self):
        prices = _make_prices(n=300)
        with pytest.raises(ValueError, match="Not enough data"):
            split_train_holdout(prices, holdout_days=200, train_window_days=252)

    def test_raises_on_non_positive_holdout(self):
        prices = _make_prices(n=800)
        with pytest.raises(ValueError, match="must be positive"):
            split_train_holdout(prices, holdout_days=0)


class TestRunTune:
    def test_runs_and_returns_verdict(self):
        prices = _make_prices(n=1200, n_tickers=4)
        rf = _make_rf(prices)
        cfg = TuneConfig(n_trials=3, holdout_days=200, sampler_seed=7)
        result = run_tune(prices, rf, cfg)

        assert result.best_params  # non-empty
        assert "optimizer" in result.best_params
        assert "train_window_days" in result.best_params
        assert result.verdict in {"ACCEPT", "REJECT"}
        assert result.holdout_report.n_rebalances > 0
        assert result.baseline_holdout_report.n_rebalances > 0

    def test_deterministic_with_seed(self):
        prices = _make_prices(n=1200, n_tickers=4, seed=11)
        rf = _make_rf(prices)
        cfg = TuneConfig(n_trials=4, holdout_days=200, sampler_seed=123)
        a = run_tune(prices, rf, cfg)
        b = run_tune(prices, rf, cfg)
        assert a.best_params == b.best_params
        assert a.best_train_sharpe == pytest.approx(b.best_train_sharpe)
