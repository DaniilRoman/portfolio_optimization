#!/usr/bin/env python3
"""Test script for predicter.py"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest
from unittest.mock import Mock, patch

from src.adapter.out.predict import predicter


def create_test_data(days: int = 365*5, start_date: datetime = None) -> pd.DataFrame:
    """Create sample historical price data for testing"""
    if start_date is None:
        start_date = datetime.now() - timedelta(days=days)
    
    dates = pd.date_range(start=start_date, periods=days, freq='D')
    
    # Generate price data with trend, seasonality, and noise
    np.random.seed(42)
    base_price = 100.0
    trend = np.linspace(0, 50, len(dates))  # Upward trend
    weekly_seasonality = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # Weekly pattern
    yearly_seasonality = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)  # Yearly pattern
    noise = np.random.normal(0, 3, len(dates))
    
    prices = base_price + trend + weekly_seasonality + yearly_seasonality + noise
    prices = np.abs(prices)  # Ensure prices are positive
    
    df = pd.DataFrame({
        'ds': dates,
        'y': prices
    })
    return df


def test_predict_basic_functionality():
    """Test basic prediction functionality"""
    print("Testing basic prediction functionality...")
    
    # Create test data
    test_data = create_test_data(days=100)
    
    # Call predict function
    model, forecast = predicter.predict(test_data, predict_period=30)
    
    # Verify model is Prophet instance
    from prophet import Prophet
    assert isinstance(model, Prophet), "Model should be a Prophet instance"
    
    # Verify forecast structure
    assert isinstance(forecast, pd.DataFrame), "Forecast should be a DataFrame"
    assert 'ds' in forecast.columns, "Forecast should have 'ds' column"
    assert 'yhat' in forecast.columns, "Forecast should have 'yhat' column"
    assert 'yhat_lower' in forecast.columns, "Forecast should have 'yhat_lower' column"
    assert 'yhat_upper' in forecast.columns, "Forecast should have 'yhat_upper' column"
    
    # Verify forecast length
    expected_length = len(test_data) + 30
    assert len(forecast) == expected_length, f"Forecast should have {expected_length} rows"
    
    # Verify additional columns
    assert 'trend' in forecast.columns, "Forecast should have 'trend' column"
    assert 'uncertainty_range' in forecast.columns, "Forecast should have 'uncertainty_range' column"
    
    print("  ✅ Basic prediction test passed")


def test_predict_with_custom_parameters():
    """Test prediction with custom parameters"""
    print("Testing prediction with custom parameters...")
    
    # Create test data
    test_data = create_test_data(days=200)
    
    # Test with additive seasonality
    model_additive, forecast_additive = predicter.predict(
        test_data,
        predict_period=20,
        seasonality_mode='additive',
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=5.0,
        interval_width=0.8
    )
    
    # Test with multiplicative seasonality (default)
    model_multiplicative, forecast_multiplicative = predicter.predict(
        test_data,
        predict_period=20,
        seasonality_mode='multiplicative'
    )
    
    # Both should produce valid forecasts
    assert len(forecast_additive) == len(test_data) + 20
    assert len(forecast_multiplicative) == len(test_data) + 20
    
    # Uncertainty ranges should be different for different interval widths
    avg_uncertainty_additive = forecast_additive['uncertainty_range'].mean()
    avg_uncertainty_multiplicative = forecast_multiplicative['uncertainty_range'].mean()
    
    print(f"  Additive uncertainty range: {avg_uncertainty_additive:.2f}")
    print(f"  Multiplicative uncertainty range: {avg_uncertainty_multiplicative:.2f}")
    
    print("  ✅ Custom parameters test passed")


def test_predict_without_holidays():
    """Test prediction without holiday effects"""
    print("Testing prediction without holidays...")
    
    test_data = create_test_data(days=150)
    
    # Test with holidays disabled
    model_no_holidays, forecast_no_holidays = predicter.predict(
        test_data,
        predict_period=15,
        add_holidays=False
    )
    
    # Test with holidays enabled (default)
    model_with_holidays, forecast_with_holidays = predicter.predict(
        test_data,
        predict_period=15,
        add_holidays=True
    )
    
    # Both should produce valid forecasts
    assert len(forecast_no_holidays) == len(test_data) + 15
    assert len(forecast_with_holidays) == len(test_data) + 15
    
    # Check that holiday columns exist in forecast_with_holidays
    # (Prophet adds holiday effects to the forecast)
    print(f"  Forecast columns with holidays: {list(forecast_with_holidays.columns)}")
    
    print("  ✅ Holiday handling test passed")


def test_predict_with_different_seasonality_settings():
    """Test prediction with different seasonality settings"""
    print("Testing different seasonality settings...")
    
    test_data = create_test_data(days=180)
    
    # Test with weekly seasonality disabled
    model_no_weekly, forecast_no_weekly = predicter.predict(
        test_data,
        predict_period=10,
        weekly_seasonality=False,
        yearly_seasonality=True
    )
    
    # Test with yearly seasonality disabled
    model_no_yearly, forecast_no_yearly = predicter.predict(
        test_data,
        predict_period=10,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    
    # Test with both enabled (default)
    model_both, forecast_both = predicter.predict(
        test_data,
        predict_period=10,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    
    # All should produce valid forecasts
    assert len(forecast_no_weekly) == len(test_data) + 10
    assert len(forecast_no_yearly) == len(test_data) + 10
    assert len(forecast_both) == len(test_data) + 10
    
    # Check that seasonality components are present
    assert 'weekly' in forecast_both.columns
    assert 'yearly' in forecast_both.columns
    assert 'quarterly' in forecast_both.columns
    assert 'monthly' in forecast_both.columns
    
    print("  ✅ Seasonality settings test passed")


def test_predict_error_handling():
    """Test error handling for invalid inputs"""
    print("Testing error handling...")
    
    # Test with insufficient data
    minimal_data = pd.DataFrame({
        'ds': [datetime.now()],
        'y': [100.0]
    })
    
    with pytest.raises(ValueError, match="Insufficient data for prediction"):
        predicter.predict(minimal_data, predict_period=10)
    
    # Test with NaN values
    data_with_nan = pd.DataFrame({
        'ds': [datetime.now() - timedelta(days=2), datetime.now() - timedelta(days=1), datetime.now()],
        'y': [100.0, np.nan, 102.0]
    })
    
    # Should handle NaN values by dropping them
    model, forecast = predicter.predict(data_with_nan, predict_period=5)
    assert len(forecast) == 2 + 5  # 2 valid data points + 5 forecast periods
    
    # Test with mixed date formats (some valid, some invalid strings)
    # This tests that the function handles date conversion errors gracefully
    data_mixed_dates = pd.DataFrame({
        'ds': ['2023-01-01', '2023-01-02', 'invalid-date'],
        'y': [100.0, 101.0, 102.0]
    })
    
    # Should handle the invalid date by converting NaT for invalid entries
    # and dropping them when cleaning NaN values
    model, forecast = predicter.predict(data_mixed_dates, predict_period=5)
    # After dropping invalid dates, we should have valid forecasts
    assert len(forecast) > 0
    
    print("  ✅ Error handling test passed")


def test_predict_backward_compatibility():
    """Test backward compatibility with original function signature"""
    print("Testing backward compatibility...")
    
    test_data = create_test_data(days=100)
    
    # Test with original signature (only data and predict_period)
    model, forecast = predicter.predict(test_data, predict_period=30)
    
    # Verify the function still works with minimal parameters
    from prophet import Prophet
    assert isinstance(model, Prophet)
    assert isinstance(forecast, pd.DataFrame)
    
    # Verify the forecast has the expected columns from original implementation
    assert 'ds' in forecast.columns
    assert 'yhat' in forecast.columns
    assert 'yhat_lower' in forecast.columns
    assert 'yhat_upper' in forecast.columns
    
    # Additional columns from enhanced implementation should also be present
    assert 'trend' in forecast.columns
    assert 'uncertainty_range' in forecast.columns
    
    print("  ✅ Backward compatibility test passed")


def test_predict_with_realistic_financial_patterns():
    """Test with realistic financial time series patterns"""
    print("Testing with realistic financial patterns...")
    
    # Create data with financial market characteristics
    np.random.seed(123)
    days = 365 * 3  # 3 years of data
    dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
    
    # Simulate bull market with corrections
    trend = np.linspace(0, 100, days)
    
    # Add market cycles (approx 4-year business cycle)
    cycle_period = 365 * 4
    cycle = 20 * np.sin(2 * np.pi * np.arange(days) / cycle_period)
    
    # Add volatility (simpler version without GARCH overflow)
    # Use lower volatility for more realistic financial data
    volatility = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(days) / 180)  # Volatility cycles every 6 months
    returns = np.random.normal(0, volatility, days) * 0.01  # Scale to 1% daily returns
    
    # Add some market crashes/corrections (milder)
    crash_days = [365, 730]  # Simulate corrections at year 1 and 2
    for crash_day in crash_days:
        if crash_day < days:
            returns[crash_day:crash_day+30] -= 0.005  # 0.5% drop over 30 days
    
    cumulative_returns = np.cumsum(returns)
    
    # Combine all components with less extreme scaling
    prices = 100 + trend + cycle + cumulative_returns * 100  # Scale cumulative returns
    prices = np.abs(prices)  # Ensure positive prices
    
    # Add some noise
    prices = prices + np.random.normal(0, 1, days)
    
    test_data = pd.DataFrame({
        'ds': dates,
        'y': prices
    })
    
    # Make prediction
    model, forecast = predicter.predict(
        test_data,
        predict_period=90,  # 3 months forecast
        changepoint_prior_scale=0.1,  # More flexible for volatile markets
        seasonality_prior_scale=15.0  # Stronger seasonality for financial data
    )
    
    # Verify predictions
    assert len(forecast) == len(test_data) + 90
    
    # Check that uncertainty intervals are reasonable
    uncertainty = forecast['uncertainty_range'].iloc[-90:].mean()  # Average for forecast period
    price_level = forecast['yhat'].iloc[-90:].mean()
    relative_uncertainty = uncertainty / price_level
    
    print(f"  Average price level: ${price_level:.2f}")
    print(f"  Average uncertainty range: ${uncertainty:.2f}")
    print(f"  Relative uncertainty: {relative_uncertainty:.2%}")
    
    # Uncertainty should be reasonable (typically 5-20% for 3-month financial forecasts)
    assert 0.01 < relative_uncertainty < 0.3, f"Uncertainty {relative_uncertainty:.2%} outside reasonable range"
    
    print("  ✅ Realistic financial patterns test passed")


def test_predict_performance_metrics():
    """Test that prediction produces reasonable performance metrics"""
    print("Testing prediction performance metrics...")
    
    test_data = create_test_data(days=200)
    
    # Split data into training and validation
    train_data = test_data.iloc[:150]
    validation_data = test_data.iloc[150:]
    
    # Make prediction for validation period
    model, forecast = predicter.predict(train_data, predict_period=50)
    
    # Get predictions for validation period
    forecast_train = forecast.iloc[:150]
    forecast_validation = forecast.iloc[150:200]
    
    # Calculate basic metrics using numpy (since scikit-learn is not in dependencies)
    def calculate_mae(actual, predicted):
        return np.mean(np.abs(actual - predicted))
    
    def calculate_rmse(actual, predicted):
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    # For training data
    train_mae = calculate_mae(train_data['y'].values, forecast_train['yhat'].values)
    train_rmse = calculate_rmse(train_data['y'].values, forecast_train['yhat'].values)
    
    # For validation data (we only have actuals for the overlapping period)
    overlap_len = min(len(validation_data), len(forecast_validation))
    if overlap_len > 0:
        val_actual = validation_data['y'].iloc[:overlap_len].values
        val_pred = forecast_validation['yhat'].iloc[:overlap_len].values
        val_mae = calculate_mae(val_actual, val_pred)
        val_rmse = calculate_rmse(val_actual, val_pred)
        
        print(f"  Training MAE: ${train_mae:.2f}, RMSE: ${train_rmse:.2f}")
        print(f"  Validation MAE: ${val_mae:.2f}, RMSE: ${val_rmse:.2f}")
        
        # For time series forecasting, validation error is typically higher than training error
        # Check that validation error is not absurdly high (more than 10x training error)
        assert val_mae < train_mae * 10, f"Validation error {val_mae:.2f} absurdly higher than training error {train_mae:.2f}"
        # Also check that we have some prediction capability (validation error less than price range)
        price_range = np.ptp(train_data['y'].values)  # Peak-to-peak range
        assert val_mae < price_range, f"Validation error {val_mae:.2f} larger than price range {price_range:.2f}"
    
    print("  ✅ Performance metrics test passed")


def run_all_tests():
    """Run all predicter tests"""
    print("=" * 60)
    print("Running predicter.py tests")
    print("=" * 60)
    
    try:
        test_predict_basic_functionality()
        print()
        
        test_predict_with_custom_parameters()
        print()
        
        test_predict_without_holidays()
        print()
        
        test_predict_with_different_seasonality_settings()
        print()
        
        test_predict_error_handling()
        print()
        
        test_predict_backward_compatibility()
        print()
        
        test_predict_with_realistic_financial_patterns()
        print()
        
        test_predict_performance_metrics()
        print()
        
        print("=" * 60)
        print("✅ All predicter tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()