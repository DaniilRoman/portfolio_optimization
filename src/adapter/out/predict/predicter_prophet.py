from pandas import DataFrame
from prophet import Prophet
import pandas as pd
from typing import Tuple


def predict(
    data: DataFrame,
    predict_period: int = 30,
    seasonality_mode: str = 'additive',
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 0.1,
    holidays_prior_scale: float = 10.0,
    mcmc_samples: int = 0,
    interval_width: float = 0.95,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = True,
    daily_seasonality: bool = False,
    add_holidays: bool = True,
    **_: object,
) -> Tuple[Prophet, DataFrame]:
    """Predict future prices using Facebook Prophet with financial time series options.

    Returns a Prophet model and a forecast DataFrame with columns:
    ds, yhat, yhat_lower, yhat_upper, trend, weekly, yearly, quarterly, monthly,
    uncertainty_range, volatility_forecast (= 0 for Prophet).
    """
    data = data.reset_index(drop=True).copy()
    data['ds'] = pd.to_datetime(data['ds'], errors='coerce')
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    data = data.dropna(subset=['ds', 'y'])

    if len(data) < 2:
        raise ValueError("Insufficient data for prediction. Need at least 2 data points.")

    prophet = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
        mcmc_samples=mcmc_samples,
        interval_width=interval_width,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=daily_seasonality,
    )

    if add_holidays:
        prophet.add_country_holidays(country_name='US')

    prophet.fit(data)

    future = prophet.make_future_dataframe(periods=predict_period, freq='B')
    forecast = prophet.predict(future)

    res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    res['trend'] = forecast['trend']
    res['weekly'] = forecast['weekly'] if 'weekly' in forecast.columns else 0
    res['yearly'] = forecast['yearly'] if 'yearly' in forecast.columns else 0
    res['quarterly'] = forecast['quarterly'] if 'quarterly' in forecast.columns else 0
    res['monthly'] = forecast['monthly'] if 'monthly' in forecast.columns else 0
    res['uncertainty_range'] = res['yhat_upper'] - res['yhat_lower']
    # Volatility forecast is GARCH-specific; emit a zero column to keep the schema uniform.
    res['volatility_forecast'] = 0.0

    return prophet, res
