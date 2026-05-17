# Portfolio Optimization

Automated ETF analysis and portfolio optimization tool. Fetches price history, forecasts future prices, scores stocks, optimizes portfolio allocation, and delivers results via Telegram.

## What it does

```
yfinance ──► GARCH/Prophet forecast ──► Analyzer (score + metadata) ──► GA/CVXPY optimizer ──► Telegram
```

1. **Downloader** — fetches up to 5 years of daily price history and ticker metadata from Yahoo Finance
2. **Predictor** — forecasts prices 90 days forward using either GARCH(1,1) or Facebook Prophet
3. **Analyzer** — scores each ETF (growth direction, volatility, profitability, sector weights)
4. **Optimizer** — allocates a fixed budget across the scored ETFs using a genetic algorithm or CVXPY
5. **Notifier** — sends forecast charts and a text recommendation to a Telegram chat

## Quickstart

```bash
# 1. Install dependencies
poetry install

# 2. Create .env with required credentials (see Configuration table below)
cp .env.example .env   # then fill in values

# 3. Run the full pipeline
python main.py
```

## Configuration

All settings are loaded from environment variables or a `.env` file.

| Variable | Required | Default | Description |
|---|---|---|---|
| `TELEGRAM_TO` | yes | — | Telegram chat ID to send results to |
| `TELEGRAM_TOKEN` | yes | — | Telegram bot token |
| `GET_AND_INCREMENT_COUNTER_URL` | yes | — | URL of the counter service (used by stock picker) |
| `APP_SCRIPT_ID` | yes | — | Google Apps Script ID for statistics tracking |
| `PREDICTER` | no | `garch` | Forecast backend: `garch` or `prophet` |
| `OPTIMIZER` | no | `ga` | Allocation backend: `ga` (genetic algorithm) or `cvxpy` |

### GA algorithm hyperparameters (optional)

Override via `GA__<FIELD>` env var (double underscore delimiter).

| Variable | Default | Description |
|---|---|---|
| `GA__POPULATION` | `120` | Population size |
| `GA__GENERATIONS` | `350` | Number of generations |
| `GA__TOURNAMENT_SIZE` | `5` | Tournament selection size |
| `GA__MUTATION_RATE` | `0.55` | Per-individual mutation probability |
| `GA__CROSSOVER_RATE` | `0.35` | Per-individual crossover probability |
| `GA__MAX_SECTOR_CONCENTRATION` | `0.40` | Max portfolio weight in one sector |

### CVXPY hyperparameters (optional)

| Variable | Default | Description |
|---|---|---|
| `CVXPY__COMPANY_MAX_EXPOSURE` | `0.10` | Max weight per individual holding |
| `CVXPY__RISK_GAMMA` | `0.01` | Risk aversion coefficient |

## Choosing algorithms

```bash
# Use GARCH(1,1) forecasting (default — faster, no external model training)
PREDICTER=garch python main.py

# Use Facebook Prophet (slower, captures seasonality)
PREDICTER=prophet python main.py

# Optimize with genetic algorithm (default)
OPTIMIZER=ga python main.py

# Optimize with CVXPY (mean-variance, deterministic)
OPTIMIZER=cvxpy python main.py
```

## Development

```bash
# Run tests
poetry run pytest

# Lint
poetry run ruff check .

# Type check
poetry run mypy src tests
```

<img width="395" alt="Forecast chart example" src="https://github.com/user-attachments/assets/69133dfa-567b-4643-9492-77a044e102c7">
