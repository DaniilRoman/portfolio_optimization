"""GARCH(1,1)+drift price forecaster; fits an ARCH model on log-return series and projects forward with confidence bands."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from pandas import DataFrame

from src.domain.data.forecast import Forecast

logger = logging.getLogger(__name__)


@dataclass
class GarchForecastModel:
    fitted_result: Any
    last_price: float
    horizon: int
    alpha: float
    mu_hat: float

    def plot(self, forecast: DataFrame) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))

        if "y" in forecast.columns:
            observed = forecast[forecast["y"].notna()]
            if not observed.empty:
                ax.plot(observed["ds"], observed["y"], label="Observed", color="black", linewidth=1.2)

        ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="C0", linewidth=1.8)
        ax.fill_between(
            forecast["ds"],
            forecast["yhat_lower"],
            forecast["yhat_upper"],
            color="C0",
            alpha=0.18,
            label="Interval",
        )
        ax.set_title("GARCH price forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(loc="best")
        fig.autofmt_xdate()


def _clean_input(data: DataFrame) -> DataFrame:
    cleaned = data.reset_index(drop=True).copy()
    cleaned["ds"] = pd.to_datetime(cleaned["ds"], errors="coerce")
    cleaned["y"] = pd.to_numeric(cleaned["y"], errors="coerce")
    cleaned = cleaned.dropna(subset=["ds", "y"])
    cleaned = cleaned.sort_values("ds").drop_duplicates(subset=["ds"], keep="last")
    if len(cleaned) < 2:
        raise ValueError("Insufficient data for prediction. Need at least 2 data points.")
    if (cleaned["y"] <= 0).any():
        raise ValueError("Price series must contain strictly positive values for log-return forecasting.")
    return cleaned


def _future_business_days(last_date: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    if periods <= 0:
        return pd.DatetimeIndex([])
    return pd.bdate_range(start=last_date + pd.tseries.offsets.BDay(1), periods=periods)


def _fit_garch(returns_pct: pd.Series) -> Any:
    am = arch_model(returns_pct, mean="Zero", vol="Garch", p=1, q=1, dist="normal")  # type: ignore[arg-type]
    return am.fit(disp="off")


def predict(
    data: DataFrame,
    predict_period: int = 30,
    interval_width: float = 0.95,
) -> Forecast:
    """Forecast future prices with a historical-drift mean and GARCH(1,1) volatility.

    Drift μ̂ is the sample mean of daily log returns; volatility is modelled by
    GARCH(1,1) with zero mean (decoupled from the drift estimate). Future bars
    follow geometric Brownian motion: yhat[i] = last_price * exp(μ̂ · i),
    with bands yhat[i] * exp(±z · σ_cum[i]).
    """

    cleaned = _clean_input(data)
    history = cleaned.copy()

    log_prices = np.log(history["y"].astype(float))
    returns = log_prices.diff().dropna()
    returns_pct = returns * 100.0

    mu_hat = float(returns.mean()) if len(returns) else 0.0
    alpha = max(min(1.0 - float(interval_width), 0.999), 1e-6)

    fitted_result = None
    sigma2_per_step_pct = None
    try:
        fitted_result = _fit_garch(returns_pct)
        if predict_period > 0:
            forecasts = fitted_result.forecast(horizon=predict_period, reindex=False)
            sigma2_per_step_pct = np.asarray(forecasts.variance.values[-1], dtype=float)
    except Exception as exc:
        logger.warning("GARCH fit failed, falling back to sample variance: %s", exc)
        fitted_result = None

    if sigma2_per_step_pct is None:
        base_var = float(np.var(returns_pct.to_numpy(), ddof=1)) if len(returns_pct) > 1 else 0.0
        if not np.isfinite(base_var) or base_var <= 0:
            base_var = 1e-6
        sigma2_per_step_pct = np.full(max(predict_period, 1), base_var, dtype=float)

    if predict_period <= 0:
        future_dates = pd.DatetimeIndex([])
    else:
        future_dates = _future_business_days(history["ds"].iloc[-1], predict_period)

    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    last_price = float(history["y"].iloc[-1])

    forecast_rows = []

    for _, row in history.iterrows():
        price = float(row["y"])
        forecast_rows.append(
            {
                "ds": row["ds"],
                "y": price,
                "yhat": price,
                "yhat_lower": price,
                "yhat_upper": price,
                "uncertainty_range": 0.0,
                "volatility_forecast": 0.0,
            }
        )

    cumulative_sigma_pct = np.sqrt(np.cumsum(np.asarray(sigma2_per_step_pct, dtype=float)))
    cumulative_sigma_decimal = cumulative_sigma_pct / 100.0
    step_volatility = np.sqrt(np.asarray(sigma2_per_step_pct, dtype=float)) / 100.0 * np.sqrt(252.0)

    for idx, ds in enumerate(future_dates):
        sigma = float(cumulative_sigma_decimal[min(idx, len(cumulative_sigma_decimal) - 1)])
        drift = float(np.exp(mu_hat * (idx + 1)))
        yhat = last_price * drift
        yhat_lower = yhat * np.exp(-z * sigma)
        yhat_upper = yhat * np.exp(z * sigma)

        forecast_rows.append(
            {
                "ds": ds,
                "y": np.nan,
                "yhat": float(yhat),
                "yhat_lower": float(yhat_lower),
                "yhat_upper": float(yhat_upper),
                "uncertainty_range": float(yhat_upper - yhat_lower),
                "volatility_forecast": float(step_volatility[min(idx, len(step_volatility) - 1)]),
            }
        )

    res = pd.DataFrame(forecast_rows)
    res["ds"] = pd.to_datetime(res["ds"], errors="coerce")
    res = res.sort_values("ds").reset_index(drop=True)

    model = GarchForecastModel(
        fitted_result=fitted_result,
        last_price=last_price,
        horizon=predict_period,
        alpha=alpha,
        mu_hat=mu_hat,
    )
    return Forecast(series=res, model=model)
