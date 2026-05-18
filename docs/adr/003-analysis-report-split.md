# ADR-003: Split StockData into Asset / MarketSnapshot / ForecastSummary / AnalysisReport

**Date:** 2026-05  
**Status:** Accepted

## Context

`StockData` was a 20-field flat dataclass passed whole to every consumer (optimizer, notifier, stats calculator). Each consumer needed only a subset of the fields, but received the entire god-object. Tests had to construct all 20 fields even when only two were under test.

## Decision

Decompose `StockData` into three focused types that represent distinct concerns:

| Type | Fields |
|------|--------|
| `Asset` | ticker_symbol, stock_name, currency, industry, description, expense_ratio, assets_under_management |
| `MarketSnapshot` | current_price, beta, standard_deviation, dividend_yield, average_daily_volume |
| `ForecastSummary` | predict_price, prediction_uncertainty, forecast_volatility, two_year_file_name, five_year_file_name |

`AnalysisReport` composes all three plus the fields that don't fit a single slice (`top_holdings`, `sector_allocation`, `is_stock_growing`, `profitability_data`).

`StockData = AnalysisReport` is kept as an alias so external code referencing the old name continues to work.

`Allocation.asset` is narrowed to `Asset` (identity only); `forecast_volatility` moves onto `Allocation` directly so the notifier can display it without accessing the full `AnalysisReport`.

## Rationale

- **Cheaper test construction**: `make_asset()` and `make_market_snapshot()` have 5–7 fields; tests only override what matters.  
- **Consumer contracts are narrower**: the optimizer signature is `list[AnalysisReport]`, but internally it reads only the sub-objects it needs.  
- **Single allocation identity**: `Allocation.asset: Asset` makes it explicit that an allocation is tied to a stock's identity, not its forecasted state.

## Alternatives considered

- **Flat `AnalysisReport`** (just rename `StockData`): zero consumer changes but misses the decomposition goal; tests remain expensive to build.  
- **Multiple inheritance** (`AnalysisReport(Asset, MarketSnapshot, ForecastSummary)`): Python dataclass MRO makes this brittle; field ordering is unpredictable.

## Consequences

- All consumers access sub-objects: `report.asset.ticker_symbol`, `report.market.current_price`, `report.forecast.predict_price`.  
- The `StockData` alias prevents import breakage in existing code; it should be removed in a future cleanup pass once all call sites are updated.
