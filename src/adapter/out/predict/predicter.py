from pandas import DataFrame
from prophet import Prophet
import pandas as pd
import numpy as np
from typing import Tuple, Optional


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
    add_holidays: bool = True
) -> Tuple[Prophet, DataFrame]:
    """
    Predict future prices using Facebook Prophet with financial time series optimizations.
    
    Args:
        data: DataFrame with columns 'ds' (datetime) and 'y' (numeric)
        predict_period: Number of periods to forecast
        seasonality_mode: 'additive' or 'multiplicative' (default: 'multiplicative' for financial data)
        changepoint_prior_scale: Flexibility of changepoints (higher = more flexible)
        seasonality_prior_scale: Strength of seasonality model
        holidays_prior_scale: Strength of holiday effects
        mcmc_samples: Number of MCMC samples for uncertainty intervals (0 = disable)
        interval_width: Width of uncertainty intervals
        weekly_seasonality: Enable weekly seasonality (default: True for financial data)
        yearly_seasonality: Enable yearly seasonality (default: True for financial data)
        daily_seasonality: Enable daily seasonality (default: False for daily data)
        add_holidays: Add US market holidays (default: True)
    
    Returns:
        Tuple of (fitted Prophet model, forecast DataFrame)
    """
    # Reset index and ensure proper data types
    data = data.reset_index(drop=True).copy()
    data['ds'] = pd.to_datetime(data['ds'], errors='coerce')
    data['y'] = pd.to_numeric(data['y'], errors='coerce')
    
    # Remove any NaN values
    data = data.dropna(subset=['ds', 'y'])
    
    if len(data) < 2:
        raise ValueError("Insufficient data for prediction. Need at least 2 data points.")
    
    # Configure Prophet with financial time series optimizations
    prophet = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
        mcmc_samples=mcmc_samples,
        interval_width=interval_width,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=daily_seasonality
    )
    
    # Add US market holidays for financial data
    if add_holidays:
        prophet.add_country_holidays(country_name='US')
    
    # # Add quarterly seasonality for financial reporting cycles
    # prophet.add_seasonality(
    #     name='quarterly',
    #     period=365.25/4,
    #     fourier_order=5,
    #     prior_scale=seasonality_prior_scale
    # )
    
    # # Add monthly seasonality for economic data releases
    # prophet.add_seasonality(
    #     name='monthly',
    #     period=30.5,
    #     fourier_order=3,
    #     prior_scale=seasonality_prior_scale
    # )
    
    # Fit the model
    prophet.fit(data)
    
    # Create future dataframe
    future = prophet.make_future_dataframe(periods=predict_period, freq='B')
    
    # Make predictions
    forecast = prophet.predict(future)
    
    # Extract relevant columns
    res = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    
    # Add additional useful metrics
    res['trend'] = forecast['trend']
    res['weekly'] = forecast['weekly'] if 'weekly' in forecast.columns else 0
    res['yearly'] = forecast['yearly'] if 'yearly' in forecast.columns else 0
    res['quarterly'] = forecast['quarterly'] if 'quarterly' in forecast.columns else 0
    res['monthly'] = forecast['monthly'] if 'monthly' in forecast.columns else 0
    
    # Calculate prediction intervals
    res['uncertainty_range'] = res['yhat_upper'] - res['yhat_lower']
    
    return prophet, res
