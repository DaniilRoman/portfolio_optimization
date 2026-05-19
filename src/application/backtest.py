"""Walk-forward backtest orchestrator and CLI entry point.

Usage:
    python -m src.application.backtest --universe de_etf --start 2020-01-01 --end 2024-12-31

The harness:
  1. Loads price history for every ticker in the universe from the §4.0.1 cache.
  2. Splits into rolling train/test windows.
  3. Calls weight_provider.compute_weights() on each train window.
  4. Passes the weight schedule to the pandas P&L engine.
  5. Returns (and optionally prints) a BacktestReport.
"""

import argparse
import logging

import pandas as pd

from src.adapter.out.backtest.pandas_engine import simulate
from src.domain.data.backtest import BacktestReport
from src.domain.ports.backtest import WeightProvider


class EqualWeightProvider:
    """Baseline: allocate equally across all available tickers."""

    def compute_weights(self, prices: pd.DataFrame) -> dict[str, float]:
        tickers = [c for c in prices.columns if prices[c].notna().any()]
        if not tickers:
            return {}
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}


def run_walk_forward(
    prices: pd.DataFrame,
    weight_provider: WeightProvider,
    risk_free_rate: pd.Series,
    train_window_days: int = 252,
    rebalance_days: int = 21,
    transaction_cost: float = 0.0005,
    slippage: float = 0.0002,
) -> BacktestReport:
    """Drive a walk-forward backtest.

    Generates a weight schedule by applying weight_provider to successive
    training windows, then delegates P&L simulation to the pandas engine.

    Args:
        prices: DataFrame with tickers as columns, sorted dates as index (tz-naive).
        weight_provider: Any WeightProvider implementation (equal-weight, HRP, BL-MVO…).
        risk_free_rate: Daily decimal Rf series from fetch_risk_free_rate.
        train_window_days: Trading days used for each training window.
        rebalance_days: Trading days between successive rebalances.
        transaction_cost: One-way cost fraction (e.g. 0.0005 = 5 bps).
        slippage: One-way slippage fraction.
    """
    dates = prices.index
    if len(dates) <= train_window_days:
        raise ValueError(f"Price history ({len(dates)} rows) is shorter than train_window_days={train_window_days}.")

    schedule: list[tuple[pd.Timestamp, dict[str, float]]] = []
    i = train_window_days
    while i < len(dates):
        rebalance_date = pd.Timestamp(dates[i])
        train_prices = prices.iloc[i - train_window_days : i]
        weights = weight_provider.compute_weights(train_prices)
        if weights:
            schedule.append((rebalance_date, weights))
        i += rebalance_days

    if not schedule:
        raise ValueError("No rebalance windows could be generated from the available data.")

    eval_prices = prices.loc[schedule[0][0] :]
    logging.info(
        "Walk-forward: %d rebalances, eval window %s → %s",
        len(schedule),
        eval_prices.index[0].date(),
        eval_prices.index[-1].date(),
    )

    return simulate(
        eval_prices,
        schedule,
        risk_free_rate,
        transaction_cost=transaction_cost,
        slippage=slippage,
    )


def _print_report(report: BacktestReport) -> None:
    print(f"  Annualised return : {report.realised_return:+.2%}")
    print(f"  Volatility        : {report.volatility:.2%}")
    print(f"  Sharpe ratio      : {report.sharpe_ratio:.3f}")
    print(f"  Sortino ratio     : {report.sortino_ratio:.3f}")
    print(f"  Max drawdown      : {report.max_drawdown:.2%}")
    print(f"  Calmar ratio      : {report.calmar_ratio:.3f}")
    print(f"  CVaR 5%           : {report.cvar_5pct:.3%}")
    print(f"  Avg turnover      : {report.turnover:.2%}")
    print(f"  Rebalances        : {report.n_rebalances}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward portfolio backtest")
    parser.add_argument("--universe", default="de_etf", help="Universe name from settings.BENCHMARK")
    parser.add_argument("--start", default="2020-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--train-window", type=int, default=252, help="Training window in trading days")
    parser.add_argument("--rebalance-days", type=int, default=21, help="Days between rebalances")
    args = parser.parse_args()

    from config.configuration import settings
    from src.adapter.out.download.reference_data import fetch_risk_free_rate

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Resolve tickers for the universe (point-in-time — avoids survivorship bias)
    from config.universe import resolve_universe

    try:
        tickers = resolve_universe(args.universe, as_of=args.start)[:20]  # cap at 20 for CLI demo
    except FileNotFoundError:
        print(f"Unknown universe '{args.universe}'. Available: sp500, etf, de_etf, ishares, vanguard")
        return
    if not tickers:
        print(f"Universe '{args.universe}' resolved to an empty list for {args.start}.")
        return

    print(f"Loading prices for {len(tickers)} tickers ({args.start} → {args.end}) ...")
    from src.adapter.out.download.downloader import download_stock_data

    price_dict: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            info = download_stock_data(ticker, start_date=args.start, end_date=args.end)
            price_dict[ticker] = info.historic_data.set_index("ds")["y"]
        except Exception as exc:
            logging.warning("Skipping %s: %s", ticker, exc)

    if len(price_dict) < 2:
        print("Not enough tickers loaded for a meaningful backtest.")
        return

    prices = pd.DataFrame(price_dict).dropna(how="all")
    rf = fetch_risk_free_rate(args.start, args.end, settings.RISK_FREE_SERIES)

    report = run_walk_forward(
        prices,
        EqualWeightProvider(),
        rf,
        train_window_days=args.train_window,
        rebalance_days=args.rebalance_days,
    )
    print(f"\nBacktest results — {args.universe} equal-weight ({args.start} → {args.end})")
    _print_report(report)


if __name__ == "__main__":
    _cli()
