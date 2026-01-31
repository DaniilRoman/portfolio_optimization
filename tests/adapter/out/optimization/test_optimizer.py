#!/usr/bin/env python3
"""Test script for the refactored optimizer.py"""

import sys  # Added missing import
from src.logic.data.data import StockData, ProfitabilityData
import numpy as np
from src.adapter.out.optimization import optimizer
from unittest.mock import Mock, patch

def create_test_stock_data():
    """Create sample StockData objects for testing"""
    stocks = []
    
    # Create 5 sample ETFs with different prices and predicted values
    test_data = [
        ("VOO", "Vanguard S&P 500 ETF", 450.25, 470.50),
        ("QQQ", "Invesco QQQ Trust", 420.75, 440.20),
        ("VTI", "Vanguard Total Stock Market ETF", 250.30, 260.80),
        ("ARKK", "ARK Innovation ETF", 55.80, 60.25),
        ("SPY", "SPDR S&P 500 ETF Trust", 510.40, 525.75),
    ]
    
    for ticker, name, current_price, predict_price in test_data:
        profitability = ProfitabilityData(
            trailing_eps=5.0,
            forward_eps=5.5,
            netIncome_to_common=1000000.0,
            ebitda_margins=0.3,
            operating_margins=0.25
        )
        
        stock = StockData(
            ticker_symbol=ticker,
            stock_name=name,
            currency="USD",
            current_price=current_price,
            predict_price=predict_price,
            two_year_file_name="",
            five_year_file_name="",
            is_stock_growing=True,
            industry="ETF",
            profitability_data=profitability,
            beta=1.0,
            standard_deviation=0.15,
            dividend_yield=0.02,
            top_holdings=np.array([]),
            sector_allocation={},
            average_daily_volume=1000000.0,
            assets_under_management=1000000000.0,
            expense_ratio=0.03,
            description="Test ETF"
        )
        stocks.append(stock)
    
    return stocks

def test_optimizer():
    """Test the refactored optimize function"""
    print("Testing refactored optimizer...")
    
    # Create test data
    stocks = create_test_stock_data()
    
    # Mock the __get_etf_map function to return test ownership data
    test_etf_map = {
        "VOO": 5,  # Already own 5 shares of VOO
        "QQQ": 2,  # Already own 2 shares of QQQ
        "VTI": 0,  # Don't own VTI
        "ARKK": 1, # Own 1 share of ARKK
        "SPY": 10  # Own 10 shares of SPY
    }
    
    with patch.object(optimizer, '__get_etf_map', return_value=test_etf_map):
        # Test with default budget (50 EUR)
        print("\nTest 1: Default budget (50 EUR)")
        result = optimizer.optimize(stocks, budget=50.0)
        print(result)
        
        # Test with larger budget (150 EUR)
        print("\nTest 2: Larger budget (150 EUR)")
        result = optimizer.optimize(stocks, budget=150.0, max_per_etf_budget=200.0)
        print(result)
        
        # Test with very small budget (10 EUR) - should find cheaper ETFs
        print("\nTest 3: Small budget (10 EUR)")
        result = optimizer.optimize(stocks, budget=10.0, max_per_etf_budget=20.0)
        print(result)
        
        print("\n✅ Optimizer tests completed successfully!")
        # No return value for pytest compliance

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    from deap import base, creator, tools, algorithms
    import requests
    print("✅ All imports successful")
    # No return value for pytest compliance

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Refactored Portfolio Optimizer")
    print("=" * 60)
    
    # Test imports first
    try:
        test_imports()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)
    
    # Run optimizer tests
    try:
        test_optimizer()
        print("\n" + "=" * 60)
        print("All tests passed! ✅")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("Tests failed! ❌")
        print("=" * 60)
        sys.exit(1)