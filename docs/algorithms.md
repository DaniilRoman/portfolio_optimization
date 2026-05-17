# Algorithms

## Predictors

| | GARCH | Prophet |
|---|---|---|
| **Model** | GARCH(1,1) on log-returns + drift | Additive trend + weekly/yearly seasonality |
| **Backend** | `arch` library | `prophet` (Facebook) |
| **Inputs** | Daily close prices (≥ 60 observations) | Daily close prices (≥ 2 observations) |
| **Outputs** | `yhat`, `yhat_lower`, `yhat_upper`, `volatility_forecast` | Same columns |
| **Volatility estimate** | Conditional variance from the fitted GARCH model | Constant — uncertainty band width proxied from residuals |
| **Trend handling** | Geometric random walk with estimated drift | Piecewise linear trend with automatic changepoint detection |
| **Runtime** | ~1–2 s per ticker | ~10–30 s per ticker (Stan compilation on first call) |
| **When to prefer** | Short horizons, volatility clustering, financial returns | Long horizons, strong seasonality, interpretable decomposition |

### Parameterisation

Both backends share the same `predict_period` argument (default 90 days in `stock_finder.run`). GARCH additionally accepts an `interval_width` keyword (default 0.95) that controls the confidence band width via the normal quantile.

Select the backend via:

```bash
PREDICTER=prophet python main.py
```

---

## Optimizers

| | GA (Genetic Algorithm) | CVXPY |
|---|---|---|
| **Formulation** | Combinatorial — integer share counts | Convex — continuous allocation fractions |
| **Objective** | Weighted score: predicted return, dividend, beta, expense ratio, sector concentration | Risk-adjusted return: maximise Σ(expected return) − γ·portfolio variance |
| **Constraints** | Budget cap, sector concentration ≤ `settings.ga.max_sector_concentration` | Budget cap, per-asset cap ≤ `settings.cvxpy.company_max_exposure` |
| **Integrality** | Native — outputs whole share counts | Relaxed — fractional allocations rounded post-solve |
| **Backend** | `deap` (`eaSimple` + tournament selection) | `cvxpy` with `ECOS_BB` → `OSQP` fallback |
| **Runtime** | ~10–60 s (population × generations evaluations) | < 1 s |
| **Outputs two portfolios?** | Yes — `risk_aware` and `profit_only` differ by objective weights | Yes — solved twice with different risk gamma |
| **When to prefer** | Realistic share-count portfolios, non-convex objectives | Fast iteration, continuous allocations, strict convex constraints |

### GA hyperparameters (`settings.ga.*`)

| Field | Default | Effect |
|---|---|---|
| `population` | 120 | Number of individuals per generation |
| `generations` | 350 | Number of EA iterations |
| `tournament_size` | 5 | Selection pressure |
| `mutation_rate` | 0.55 | Per-individual mutation probability |
| `crossover_rate` | 0.35 | Per-individual crossover probability |
| `mutation_indpb` | 0.4 | Per-gene mutation probability |
| `mate_indpb` | 0.1 | Per-gene crossover probability |
| `max_sector_concentration` | 0.40 | Maximum weight in any single GICS sector |

Override via env vars using double-underscore nesting: `GA__POPULATION=20 GA__GENERATIONS=10 python main.py`

### CVXPY hyperparameters (`settings.cvxpy.*`)

| Field | Default | Effect |
|---|---|---|
| `solvers` | `["ECOS_BB", "OSQP"]` | Tried in order; first solver to succeed wins |
| `company_max_exposure` | 0.10 | Maximum fraction of budget in any single asset |
| `risk_gamma` | 0.01 | Risk-aversion coefficient γ in the objective |

Select the optimizer via:

```bash
OPTIMIZER=cvxpy python main.py
```
