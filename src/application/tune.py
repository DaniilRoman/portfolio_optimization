"""Hyperparameter sweep over WeightProvider configurations using Optuna.

The sweep wraps each candidate optimiser as a `WeightProvider`, runs
`run_walk_forward` from `src.application.backtest`, and maximises out-of-sample
Sharpe ratio. Results are persisted as an Optuna SQLite study under
`data/tuning/` and validated against an untouched hold-out window plus an
equal-weight baseline before being recommended.

Usage:
    python -m src.application.tune --universe de_etf \\
        --start 2018-01-01 --end 2024-12-31 \\
        --holdout-days 252 --trials 50

The tuning is intended to be run ad-hoc (or via a manual `workflow_dispatch`
GHA job), NOT on the weekly inference cron.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from pypfopt import EfficientFrontier, HRPOpt

from src.adapter.out.risk.covariance import compute_covariance
from src.application.backtest import EqualWeightProvider, run_walk_forward
from src.domain.data.backtest import BacktestReport

_TRAIN_WINDOW_CHOICES = (126, 252, 504)
_REBALANCE_CHOICES = (5, 21, 63)
_HRP_LINKAGES = ("single", "complete", "average", "ward")
_DEFAULT_FAILED_TRIAL_SHARPE = -1e6


class HRPProvider:
    """WeightProvider using Hierarchical Risk Parity with a configurable linkage."""

    def __init__(self, linkage_method: str) -> None:
        self.linkage_method = linkage_method

    def compute_weights(self, prices: pd.DataFrame) -> dict[str, float]:
        if prices.shape[1] < 2:
            return {}
        sigma = compute_covariance(prices)
        tickers = [str(c) for c in prices.columns]
        log_returns = np.log(prices / prices.shift(1)).dropna()
        cov_df = pd.DataFrame(sigma, index=tickers, columns=tickers)
        hrp = HRPOpt(returns=log_returns, cov_matrix=cov_df)
        weights = hrp.optimize(linkage_method=self.linkage_method)
        return {str(k): float(v) for k, v in weights.items()}


class MinVarianceProvider:
    """Long-only minimum-variance portfolio using Ledoit-Wolf Σ."""

    def __init__(self, max_weight: float) -> None:
        self.max_weight = max_weight

    def compute_weights(self, prices: pd.DataFrame) -> dict[str, float]:
        if prices.shape[1] < 2:
            return {}
        sigma = compute_covariance(prices)
        tickers = [str(c) for c in prices.columns]
        cov_df = pd.DataFrame(sigma, index=tickers, columns=tickers)
        mu_ser = pd.Series(0.0, index=tickers)
        try:
            ef = EfficientFrontier(mu_ser, cov_df, weight_bounds=(0.0, self.max_weight))
            ef.min_volatility()
            weights = ef.clean_weights()
        except Exception as exc:
            logging.warning("MinVariance solver failed (%s); falling back to equal weight.", exc)
            equal = 1.0 / len(tickers)
            return {t: equal for t in tickers}
        return {str(k): float(v) for k, v in weights.items()}


@dataclass
class TuneConfig:
    n_trials: int = 50
    holdout_days: int = 252
    sampler_seed: int = 42
    storage: str | None = None  # Optuna storage URL (e.g. sqlite:///data/tuning/foo.db)
    study_name: str = "tune"


@dataclass
class TuneResult:
    study: optuna.Study
    best_params: dict[str, Any]
    best_train_sharpe: float
    holdout_report: BacktestReport
    baseline_holdout_report: BacktestReport
    verdict: str  # "ACCEPT" or "REJECT"


def build_provider(params: dict[str, Any]) -> HRPProvider | MinVarianceProvider:
    """Construct a WeightProvider from a parameter dict (best_params from a study)."""
    optimizer = params["optimizer"]
    if optimizer == "hrp":
        return HRPProvider(linkage_method=params["hrp_linkage"])
    if optimizer == "min_var":
        return MinVarianceProvider(max_weight=params["min_var_max_weight"])
    raise ValueError(f"Unknown optimizer in params: {optimizer!r}")


def split_train_holdout(
    prices: pd.DataFrame,
    holdout_days: int,
    train_window_days: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split price history into a tuning window and a hold-out window.

    The hold-out window keeps `train_window_days` of prior context so that the
    backtest's first rebalance has a full lookback. The tuning window ends
    exactly `holdout_days` before the last available date.
    """
    if holdout_days <= 0:
        raise ValueError(f"holdout_days must be positive; got {holdout_days}")
    if len(prices) <= holdout_days + train_window_days:
        raise ValueError(
            f"Not enough data: need > {holdout_days + train_window_days} rows for "
            f"a {holdout_days}-day holdout with {train_window_days}-day context, got {len(prices)}."
        )
    train = prices.iloc[:-holdout_days]
    holdout = prices.iloc[-(holdout_days + train_window_days) :]
    return train, holdout


def _objective(
    trial: optuna.Trial,
    prices: pd.DataFrame,
    risk_free_rate: pd.Series,
) -> float:
    optimizer = trial.suggest_categorical("optimizer", ["hrp", "min_var"])
    if optimizer == "hrp":
        trial.suggest_categorical("hrp_linkage", list(_HRP_LINKAGES))
    else:
        trial.suggest_float("min_var_max_weight", 0.05, 0.30)

    train_window = int(trial.suggest_categorical("train_window_days", list(_TRAIN_WINDOW_CHOICES)))
    rebalance = int(trial.suggest_categorical("rebalance_days", list(_REBALANCE_CHOICES)))

    provider = build_provider(trial.params)
    try:
        report = run_walk_forward(
            prices,
            provider,
            risk_free_rate,
            train_window_days=train_window,
            rebalance_days=rebalance,
        )
    except Exception as exc:
        logging.warning("Trial %d failed: %s", trial.number, exc)
        return _DEFAULT_FAILED_TRIAL_SHARPE
    return float(report.sharpe_ratio)


def run_tune(
    prices: pd.DataFrame,
    risk_free_rate: pd.Series,
    config: TuneConfig | None = None,
) -> TuneResult:
    """Run an Optuna study, evaluate the best params on the hold-out, and compare to equal-weight.

    Args:
        prices: Full price history (tickers × dates).
        risk_free_rate: Daily decimal Rf series spanning the price history.
        config: Tuning configuration; uses defaults if omitted.

    Returns:
        TuneResult with the study, best params, train-window Sharpe, hold-out Sharpe,
        an equal-weight baseline on the same hold-out, and an ACCEPT / REJECT verdict.
    """
    cfg = config or TuneConfig()
    train_prices, holdout_prices = split_train_holdout(prices, cfg.holdout_days)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.sampler_seed),
        storage=cfg.storage,
        study_name=cfg.study_name,
        load_if_exists=cfg.storage is not None,
    )
    study.optimize(lambda t: _objective(t, train_prices, risk_free_rate), n_trials=cfg.n_trials)

    best_params = dict(study.best_params)
    provider = build_provider(best_params)
    holdout_report = run_walk_forward(
        holdout_prices,
        provider,
        risk_free_rate,
        train_window_days=int(best_params["train_window_days"]),
        rebalance_days=int(best_params["rebalance_days"]),
    )
    baseline_report = run_walk_forward(
        holdout_prices,
        EqualWeightProvider(),
        risk_free_rate,
        train_window_days=int(best_params["train_window_days"]),
        rebalance_days=int(best_params["rebalance_days"]),
    )

    verdict = "ACCEPT" if holdout_report.sharpe_ratio > baseline_report.sharpe_ratio else "REJECT"

    return TuneResult(
        study=study,
        best_params=best_params,
        best_train_sharpe=float(study.best_value),
        holdout_report=holdout_report,
        baseline_holdout_report=baseline_report,
        verdict=verdict,
    )


def _print_summary(result: TuneResult) -> None:
    print("\n=== Best parameters (tuned on train window) ===")
    for k, v in result.best_params.items():
        print(f"  {k}: {v}")
    print(f"  Train Sharpe        : {result.best_train_sharpe:+.3f}")
    print("\n=== Hold-out evaluation ===")
    print(f"  Tuned-config Sharpe : {result.holdout_report.sharpe_ratio:+.3f}")
    print(f"  Equal-weight Sharpe : {result.baseline_holdout_report.sharpe_ratio:+.3f}")
    print(f"  Max drawdown (tuned): {result.holdout_report.max_drawdown:.2%}")
    print(f"  Annualised return   : {result.holdout_report.realised_return:+.2%}")
    print(f"\nVerdict: {result.verdict} (tuned vs. equal-weight on hold-out)")
    if result.verdict == "REJECT":
        print("  → Do NOT commit these params to prod.env; the equal-weight baseline beat the tuned config.")
    else:
        print("  → Safe to consider as a prod.env override; review parameter importance before committing.")


def _load_prices_for_cli(universe: str, start: str, end: str, max_tickers: int) -> pd.DataFrame:
    """Load price history for the CLI; lives here so unit tests can avoid the network."""
    from config.universe import resolve_universe
    from src.adapter.out.download.downloader import download_stock_data

    tickers = resolve_universe(universe, as_of=start)[:max_tickers]
    price_dict: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            info = download_stock_data(ticker, start_date=start, end_date=end)
            price_dict[ticker] = info.historic_data.set_index("ds")["y"]
        except Exception as exc:
            logging.warning("Skipping %s: %s", ticker, exc)
    return pd.DataFrame(price_dict).dropna(how="all")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward Sharpe-maximising hyperparameter sweep")
    parser.add_argument("--universe", default="de_etf")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--holdout-days", type=int, default=252)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--max-tickers", type=int, default=20, help="Cap on universe size for CLI runs")
    parser.add_argument("--storage-dir", default="data/tuning")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from config.configuration import settings
    from src.adapter.out.download.reference_data import fetch_risk_free_rate

    Path(args.storage_dir).mkdir(parents=True, exist_ok=True)
    study_name = f"{args.universe}_{args.start}_{args.end}"
    storage = f"sqlite:///{args.storage_dir}/{study_name}.db"

    print(f"Loading prices for {args.universe} ({args.start} → {args.end}) ...")
    prices = _load_prices_for_cli(args.universe, args.start, args.end, args.max_tickers)
    if len(prices.columns) < 2:
        print("Not enough tickers loaded for a meaningful sweep.")
        return

    rf = fetch_risk_free_rate(args.start, args.end, settings.RISK_FREE_SERIES)

    config = TuneConfig(
        n_trials=args.trials,
        holdout_days=args.holdout_days,
        sampler_seed=args.seed,
        storage=storage,
        study_name=study_name,
    )
    result = run_tune(prices, rf, config)
    _print_summary(result)
    print(f"\nStudy persisted at: {storage}")


if __name__ == "__main__":
    _cli()
