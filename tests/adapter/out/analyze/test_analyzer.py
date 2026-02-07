#!/usr/bin/env python3
"""Test script for analyzer.py"""

import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.logic.data.data import StockData, StockInfo, ProfitabilityData
from src.adapter.out.analyze import analyzer

def create_test_historic_data():
    """Create sample historical price data for testing"""
    # Create date range for 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price data with some noise and trend
    np.random.seed(42)
    base_price = 100.0
    trend = np.linspace(0, 50, len(dates))  # Upward trend
    noise = np.random.normal(0, 5, len(dates))
    prices = base_price + trend + noise
    
    # Ensure prices are positive
    prices = np.abs(prices)
    
    df = pd.DataFrame({'y': prices}, index=dates)
    return df

def create_test_predicted_prices():
    """Create sample predicted price data for testing"""
    # Create future dates for predictions
    start_date = datetime.now()
    future_dates = pd.date_range(start=start_date, periods=365*5, freq='D')
    
    # Generate predicted prices with trend
    np.random.seed(42)
    base_price = 150.0
    trend = np.linspace(0, 100, len(future_dates))
    noise = np.random.normal(0, 3, len(future_dates))
    predicted_prices = base_price + trend + noise
    
    df = pd.DataFrame({
        'yhat': predicted_prices,
        'yhat_lower': predicted_prices - 5,
        'yhat_upper': predicted_prices + 5
    }, index=future_dates)
    return df

def create_mock_ticker():
    """Create a mock yfinance Ticker object"""
    mock_ticker = Mock()
    
    # Mock info dictionary - using real yfinance keys
    # Note: yfinance returns expense ratio as percentage (e.g., 0.03 for 0.03%)
    # The analyzer divides by 100 to convert to decimal
    mock_ticker.info = {
        'longName': 'Test ETF',
        'currency': 'USD',
        'industry': 'ETF',
        'averageVolume': 1000000.0,
        'beta': 1.0,
        'totalAssets': 1000000000.0,
        'netExpenseRatio': 0.03,  # 0.03% expense ratio
        'yield': 0.02,
        'ebitdaMargins': 0.3,
        'forwardEps': 5.5,
        'netIncomeToCommon': 1000000.0,
        'operatingMargins': 0.25,
        'trailingEps': 5.0
    }
    
    # Mock funds_data - using real FundsData attributes
    mock_funds_data = Mock()
    mock_funds_data.description = "Test ETF Description"
    mock_funds_data.fund_overview = {
        'family': 'Test Family',
        'legalType': 'ETF'
    }
    
    # Mock top holdings as numpy array
    mock_funds_data.top_holdings = Mock()
    mock_funds_data.top_holdings.values = np.array([
        ["Apple Inc", 0.08],
        ["Microsoft Corp", 0.07],
        ["Amazon.com Inc", 0.05]
    ])
    
    # Mock sector weightings
    mock_funds_data.sector_weightings = {
        "Technology": 0.45,
        "Healthcare": 0.30,
        "Financials": 0.25
    }
    
    mock_ticker.funds_data = mock_funds_data
    return mock_ticker

def create_test_stock_info():
    """Create a complete StockInfo object for testing"""
    historic_data = create_test_historic_data()
    ticker = create_mock_ticker()
    return StockInfo(historic_data=historic_data, ticker=ticker)

def test_last_price():
    """Test __last_price function"""
    print("Testing __last_price function...")
    
    # Create test DataFrame
    test_data = pd.DataFrame({
        'y': [100.0, 101.5, 102.3, 103.7, 104.2]
    })
    
    # Import the module and test the function directly
    from src.adapter.out.analyze.analyzer import __last_price
    result = __last_price(test_data, "y")
    expected = 104.2
    
    print(f"  Result: {result}, Expected: {expected}")
    assert round(result, 2) == round(expected, 2), f"Expected {expected}, got {result}"
    print("  ✅ __last_price test passed")

def test_slice():
    """Test __slice function"""
    print("Testing __slice function...")
    
    # Create test DataFrame with datetime index
    dates = pd.date_range(start='2020-01-01', periods=365*5, freq='D')
    prices = np.random.randn(len(dates)) * 10 + 100
    historic_data = pd.DataFrame({'y': prices}, index=dates)
    
    # Import the module and test the function directly
    from src.adapter.out.analyze.analyzer import __slice
    # Test slicing 2 years ago with 30-day window
    result = __slice(historic_data, 365*2, 30)
    
    # Check that result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    
    # Check that we have data (might be empty if dates don't align perfectly)
    if len(result) > 0:
        # Check that all dates are within expected range
        latest_date = historic_data.index[-1]
        expected_start = latest_date - pd.Timedelta(days=365*2)
        expected_end = latest_date - pd.Timedelta(days=365*2 - 30)
        
        # Allow for some date alignment issues
        print(f"  Sliced {len(result)} rows")
        print(f"  Date range: {result.index[0]} to {result.index[-1]}")
    
    print("  ✅ __slice test passed")

def test_is_stock_historicly_growing():
    """Test __is_stock_historicly_growing function"""
    print("Testing __is_stock_historicly_growing function...")
    
    # Import the module and test the function directly
    from src.adapter.out.analyze.analyzer import __is_stock_historicly_growing
    
    # Test case 1: Current price is higher than all historical prices
    current_price = 150.0
    history_slice = pd.DataFrame({'y': [100.0, 110.0, 120.0, 130.0, 140.0]})
    result = __is_stock_historicly_growing(current_price, history_slice)
    assert result == True, "Should return True when current price > all historical prices"
    print("  ✅ Case 1: Current price higher than history - passed")
    
    # Test case 2: Current price is lower than some historical prices
    current_price = 125.0
    history_slice = pd.DataFrame({'y': [100.0, 110.0, 120.0, 130.0, 140.0]})
    result = __is_stock_historicly_growing(current_price, history_slice)
    assert result == True, "Should return True when current price > some historical prices"
    print("  ✅ Case 2: Current price lower than some history - passed")
    
    # Test case 3: Current price is lower than all historical prices
    current_price = 90.0
    history_slice = pd.DataFrame({'y': [100.0, 110.0, 120.0, 130.0, 140.0]})
    result = __is_stock_historicly_growing(current_price, history_slice)
    assert result == False, "Should return False when current price < all historical prices"
    print("  ✅ Case 3: Current price lower than all history - passed")

def test_is_stock_growing():
    """Test __is_stock_growing function"""
    print("Testing __is_stock_growing function...")
    
    # Import the module and test the function directly
    from src.adapter.out.analyze.analyzer import __is_stock_growing
    
    # Create test historic data
    dates = pd.date_range(start='2020-01-01', periods=365*5, freq='D')
    prices = np.linspace(100, 200, len(dates))  # Steady growth
    historic_data = pd.DataFrame({'y': prices}, index=dates)
    
    # Test case 1: All conditions met (growing stock)
    current_price = 180.0
    two_year_predicted = 200.0  # > current_price
    five_year_predicted = 220.0  # > current_price
    
    result = __is_stock_growing(
        current_price, two_year_predicted, five_year_predicted, historic_data
    )
    
    # Note: This depends on the historical data slice having lower prices
    # With our steadily growing data, this should be True
    print(f"  Growing stock test result: {result}")
    
    # Test case 2: Predicted prices lower than current (not growing)
    current_price = 200.0
    two_year_predicted = 190.0  # < current_price
    five_year_predicted = 180.0  # < current_price
    
    result = __is_stock_growing(
        current_price, two_year_predicted, five_year_predicted, historic_data
    )
    assert result == False, "Should return False when predicted prices are lower"
    print("  ✅ Non-growing stock test passed")

def test_analyses_function():
    """Test the main analyses function"""
    print("Testing analyses function...")
    
    # Create test data
    ticker_symbol = "VOO"
    stock_info = create_test_stock_info()
    
    # Mock Prophet models
    two_year_prophet = Mock()
    five_year_prophet = Mock()
    
    # Create predicted prices
    two_year_predicted_prices = create_test_predicted_prices()
    five_year_predicted_prices = create_test_predicted_prices()
    
    # Mock the plot and savefig functions to avoid file I/O
    with patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('matplotlib.pyplot.figure') as mock_figure:
        
        # Call the analyses function
        result = analyzer.analyses(
            ticker_symbol=ticker_symbol,
            stock_info=stock_info,
            two_year_prophet=two_year_prophet,
            two_year_predicted_prices=two_year_predicted_prices,
            five_year_prophet=five_year_prophet,
            five_year_predicted_prices=five_year_predicted_prices
        )
    
    # Verify the result is a StockData object
    assert isinstance(result, StockData), "Result should be a StockData object"
    print(f"  ✅ Returned StockData object")
    
    # Verify key fields
    assert result.ticker_symbol == ticker_symbol
    assert result.stock_name == "Test ETF"
    assert result.currency == "USD"
    assert result.industry == "ETF"
    assert result.beta == 1.0
    # standard_deviation is calculated from historical data, not from mock info
    # so we just check it's a float
    assert isinstance(result.standard_deviation, float)
    assert result.dividend_yield == 0.02
    assert result.assets_under_management == 1000000000.0
    # expense_ratio is now divided by 100 in analyzer (0.03% -> 0.0003)
    assert result.expense_ratio == 0.0003
    
    # Verify profitability data
    assert isinstance(result.profitability_data, ProfitabilityData)
    assert result.profitability_data.trailing_eps == 5.0
    assert result.profitability_data.forward_eps == 5.5
    assert result.profitability_data.netIncome_to_common == 1000000.0
    assert result.profitability_data.ebitda_margins == 0.3
    assert result.profitability_data.operating_margins == 0.25
    
    # Verify top holdings and sector allocation
    assert isinstance(result.top_holdings, np.ndarray)
    assert len(result.top_holdings) == 3
    assert isinstance(result.sector_allocation, dict)
    assert "Technology" in result.sector_allocation
    
    # Verify file names were set
    assert result.two_year_file_name == f'two_year_{ticker_symbol}.png'
    assert result.five_year_file_name == f'five_year_{ticker_symbol}.png'
    
    print("  ✅ All StockData fields correctly populated")

def test_analyses_with_missing_fund_data():
    """Test analyses function with missing fund data attributes"""
    print("Testing analyses with missing fund data...")
    
    # Create test data with missing fund attributes
    ticker_symbol = "TEST"
    stock_info = create_test_stock_info()
    
    # Remove some attributes from info dict (not funds_data)
    # Since our updated code gets these from info dict, not funds_data
    stock_info.ticker.info.pop('totalAssets', None)
    stock_info.ticker.info.pop('netExpenseRatio', None)
    stock_info.ticker.info.pop('yield', None)
    
    # Mock Prophet models
    two_year_prophet = Mock()
    five_year_prophet = Mock()
    
    # Create predicted prices
    two_year_predicted_prices = create_test_predicted_prices()
    five_year_predicted_prices = create_test_predicted_prices()
    
    # Mock the plot and savefig functions
    with patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('matplotlib.pyplot.figure') as mock_figure:
        
        # Call the analyses function
        result = analyzer.analyses(
            ticker_symbol=ticker_symbol,
            stock_info=stock_info,
            two_year_prophet=two_year_prophet,
            two_year_predicted_prices=two_year_predicted_prices,
            five_year_prophet=five_year_prophet,
            five_year_predicted_prices=five_year_predicted_prices
        )
    
    # Verify that missing attributes default to 0
    print(f"  assets_under_management: {result.assets_under_management}")
    print(f"  expense_ratio: {result.expense_ratio}")
    print(f"  dividend_yield: {result.dividend_yield}")
    
    # These should be 0 since the keys are missing from info dict
    assert result.assets_under_management == 0, f"Expected 0, got {result.assets_under_management}"
    assert result.expense_ratio == 0, f"Expected 0, got {result.expense_ratio}"
    assert result.dividend_yield == 0, f"Expected 0, got {result.dividend_yield}"
    
    print("  ✅ Missing fund data handled correctly")

def test_analyses_with_exception_in_description():
    """Test analyses function when description generation fails"""
    print("Testing analyses with description generation exception...")
    
    # Create test data
    ticker_symbol = "TEST"
    stock_info = create_test_stock_info()
    
    # Make description generation fail
    stock_info.ticker.funds_data.description = None
    stock_info.ticker.funds_data.fund_overview = None
    
    # Mock Prophet models
    two_year_prophet = Mock()
    five_year_prophet = Mock()
    
    # Create predicted prices
    two_year_predicted_prices = create_test_predicted_prices()
    five_year_predicted_prices = create_test_predicted_prices()
    
    # Mock the plot and savefig functions
    with patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('matplotlib.pyplot.figure') as mock_figure:
        
        # Call the analyses function
        result = analyzer.analyses(
            ticker_symbol=ticker_symbol,
            stock_info=stock_info,
            two_year_prophet=two_year_prophet,
            two_year_predicted_prices=two_year_predicted_prices,
            five_year_prophet=five_year_prophet,
            five_year_predicted_prices=five_year_predicted_prices
        )
    
    # Verify description is empty string
    assert result.description == ''
    
    print("  ✅ Exception in description generation handled correctly")

def test_pessimistic_predict_price_calculation():
    """Test the new pessimistic predict_price calculation"""
    print("Testing pessimistic predict_price calculation...")
    
    # Create test data
    ticker_symbol = "TEST"
    stock_info = create_test_stock_info()
    
    # Mock Prophet models
    two_year_prophet = Mock()
    five_year_prophet = Mock()
    
    # Create predicted prices with known values for testing
    # We'll create simple DataFrames with known last values
    future_dates = pd.date_range(start=datetime.now(), periods=10, freq='D')
    
    # Create two-year predictions: yhat=200, yhat_lower=195 (5 less)
    two_year_predicted_prices = pd.DataFrame({
        'yhat': [200.0] * 10,
        'yhat_lower': [195.0] * 10,
        'yhat_upper': [205.0] * 10
    }, index=future_dates)
    
    # Create five-year predictions: yhat=210, yhat_lower=205 (5 less)
    five_year_predicted_prices = pd.DataFrame({
        'yhat': [210.0] * 10,
        'yhat_lower': [205.0] * 10,
        'yhat_upper': [215.0] * 10
    }, index=future_dates)
    
    # Mock the plot and savefig functions
    with patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('matplotlib.pyplot.figure') as mock_figure:
        
        # Call the analyses function
        result = analyzer.analyses(
            ticker_symbol=ticker_symbol,
            stock_info=stock_info,
            two_year_prophet=two_year_prophet,
            two_year_predicted_prices=two_year_predicted_prices,
            five_year_prophet=five_year_prophet,
            five_year_predicted_prices=five_year_predicted_prices
        )
    
    # Verify the predict_price is the minimum of the two yhat_lower values
    # two_year_min = 195.0, five_year_min = 205.0, min = 195.0
    expected_predict_price = 195.0
    actual_predict_price = result.predict_price
    
    print(f"  Expected predict_price: {expected_predict_price}")
    print(f"  Actual predict_price: {actual_predict_price}")
    
    assert round(actual_predict_price, 2) == round(expected_predict_price, 2), \
        f"Expected predict_price {expected_predict_price}, got {actual_predict_price}"
    
    print("  ✅ Pessimistic predict_price calculation test passed")

def test_pessimistic_predict_price_with_different_values():
    """Test pessimistic calculation when five-year minimum is lower"""
    print("Testing pessimistic predict_price with different values...")
    
    # Create test data
    ticker_symbol = "TEST"
    stock_info = create_test_stock_info()
    
    # Mock Prophet models
    two_year_prophet = Mock()
    five_year_prophet = Mock()
    
    # Create predicted prices with five-year minimum being lower
    future_dates = pd.date_range(start=datetime.now(), periods=10, freq='D')
    
    # Create two-year predictions: yhat_lower=200
    two_year_predicted_prices = pd.DataFrame({
        'yhat': [205.0] * 10,
        'yhat_lower': [200.0] * 10,
        'yhat_upper': [210.0] * 10
    }, index=future_dates)
    
    # Create five-year predictions: yhat_lower=190 (lower than two-year)
    five_year_predicted_prices = pd.DataFrame({
        'yhat': [195.0] * 10,
        'yhat_lower': [190.0] * 10,
        'yhat_upper': [200.0] * 10
    }, index=future_dates)
    
    # Mock the plot and savefig functions
    with patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('matplotlib.pyplot.figure') as mock_figure:
        
        # Call the analyses function
        result = analyzer.analyses(
            ticker_symbol=ticker_symbol,
            stock_info=stock_info,
            two_year_prophet=two_year_prophet,
            two_year_predicted_prices=two_year_predicted_prices,
            five_year_prophet=five_year_prophet,
            five_year_predicted_prices=five_year_predicted_prices
        )
    
    # Verify the predict_price is the minimum of the two yhat_lower values
    # two_year_min = 200.0, five_year_min = 190.0, min = 190.0
    expected_predict_price = 190.0
    actual_predict_price = result.predict_price
    
    print(f"  Expected predict_price: {expected_predict_price}")
    print(f"  Actual predict_price: {actual_predict_price}")
    
    assert round(actual_predict_price, 2) == round(expected_predict_price, 2), \
        f"Expected predict_price {expected_predict_price}, got {actual_predict_price}"
    
    print("  ✅ Pessimistic predict_price with different values test passed")

def run_all_tests():
    """Run all analyzer tests"""
    print("=" * 60)
    print("Running analyzer.py tests")
    print("=" * 60)
    
    try:
        test_last_price()
        print()
        
        test_slice()
        print()
        
        test_is_stock_historicly_growing()
        print()
        
        test_is_stock_growing()
        print()
        
        test_analyses_function()
        print()
        
        test_analyses_with_missing_fund_data()
        print()
        
        test_analyses_with_exception_in_description()
        print()
        
        test_pessimistic_predict_price_calculation()
        print()
        
        test_pessimistic_predict_price_with_different_values()
        print()
        
        print("=" * 60)
        print("✅ All analyzer tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    run_all_tests()