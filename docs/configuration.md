# Configuration Reference

All settings are loaded by `config/configuration.py` via `pydantic-settings`.
Priority order (highest wins): shell environment variables → `.env` → `config/profiles/{APP_ENV}.env`.

---

## Top-level settings

| Env var | Type | Default | Description |
|---------|------|---------|-------------|
| `APP_ENV` | `"dev" \| "prod" \| "backtest"` | `"prod"` | Selects the profile overlay from `config/profiles/{APP_ENV}.env`. |
| `TELEGRAM_ENABLED` | `bool` | `true` | When `false`, Telegram calls are skipped (dry-run). Credentials become optional. |
| `TELEGRAM_TOKEN` | `str` | `""` | Telegram bot token. Required when `TELEGRAM_ENABLED=true`. |
| `TELEGRAM_TO` | `str` | `""` | Telegram chat ID to send messages to. Required when `TELEGRAM_ENABLED=true`. |
| `GET_AND_INCREMENT_COUNTER_URL` | `str` | `""` | Google Apps Script endpoint for the ETF ownership counter. |
| `APP_SCRIPT_ID` | `str` | `""` | Google Apps Script ID used to build the stats-sheet URL. |
| `PREDICTER` | `"garch" \| "prophet"` | `"garch"` | Selects the price-forecast backend. |
| `OPTIMIZER` | `"ga" \| "cvxpy"` | `"ga"` | Selects the portfolio-optimization backend. |
| `UNIVERSE` | `"sp500" \| "etf" \| "de_etf" \| "ishares" \| "vanguard"` | `"etf"` | Selects the ETF universe to analyze. |

---

## GA settings (`GA__*`)

Nested under `settings.ga`. Override with env vars prefixed `GA__` (double underscore delimiter).

| Env var | Type | Default | Description |
|---------|------|---------|-------------|
| `GA__POPULATION` | `int` | `120` | Number of individuals per generation. |
| `GA__GENERATIONS` | `int` | `350` | Number of generations to evolve. |
| `GA__TOURNAMENT_SIZE` | `int` | `5` | Tournament selection size. |
| `GA__MUTATION_RATE` | `float` | `0.55` | Probability of mutating an individual. |
| `GA__CROSSOVER_RATE` | `float` | `0.35` | Probability of crossing two individuals. |
| `GA__MUTATION_INDPB` | `float` | `0.4` | Per-gene mutation probability within a mutated individual. |
| `GA__MATE_INDPB` | `float` | `0.1` | Per-gene crossover probability within a mating pair. |
| `GA__MAX_SECTOR_CONCENTRATION` | `float` | `0.40` | Maximum fraction of portfolio value allowed in any single sector. |

---

## CVXPY settings (`CVXPY__*`)

Nested under `settings.cvxpy`. Override with env vars prefixed `CVXPY__`.

| Env var | Type | Default | Description |
|---------|------|---------|-------------|
| `CVXPY__SOLVERS` | `list[str]` | `["ECOS_BB", "OSQP"]` | Solver order; first available solver is used with fallback to the next. |
| `CVXPY__COMPANY_MAX_EXPOSURE` | `float` | `0.10` | Maximum fraction of portfolio value allowed in any single underlying company. |
| `CVXPY__RISK_GAMMA` | `float` | `0.01` | Risk-aversion coefficient scaling the risk penalty in the objective function. |

---

## Profiles

| `APP_ENV` | Universe | Telegram | Purpose |
|-----------|----------|----------|---------|
| `dev` | Small subset | Disabled | Fast local iteration without Telegram noise |
| `prod` | Full universe | Enabled | Live production run |
| `backtest` | Configurable | Disabled | Historical analysis without notifications |

Profile files live at `config/profiles/{APP_ENV}.env` and are merged after `.env`.
Shell env vars always win over both files.

---

## Quick-start examples

```bash
# Dry-run with CVXPY optimizer on iShares universe
APP_ENV=dev OPTIMIZER=cvxpy UNIVERSE=ishares python main.py

# Production GA run on full ETF universe
APP_ENV=prod python main.py

# Test with a tiny population for speed
APP_ENV=dev GA__POPULATION=20 GA__GENERATIONS=10 python main.py
```
