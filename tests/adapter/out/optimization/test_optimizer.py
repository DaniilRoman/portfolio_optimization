#!/usr/bin/env python3
"""Test script for the refactored optimizer.py"""

from src.logic.data.data import StockData, ProfitabilityData
import numpy as np
from src.adapter.out.optimization import optimizer
from unittest.mock import Mock, patch

def create_test_stock_data():
    """Create sample StockData objects for testing"""
    stocks = []
    
    # Create 5 sample ETFs with different prices and predicted values
    test_data = [
        ("VOO", "Vanguard S&P 500 ETF", 40.25, 41.50),
        ("QQQ", "Invesco QQQ Trust", 20.75, 22.20),
        ("VTI", "Vanguard Total Stock Market ETF", 25.30, 26.80),
        ("ARKK", "ARK Innovation ETF", 55.80, 60.25),
        ("SPY", "SPDR S&P 500 ETF Trust", 10.40, 11.75),
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
    print("Testing refactored optimizer with expense_ratio and dividend_yield...")
    
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
        print("============================================")
        print(result)
        print("============================================")
        
        print("\n✅ Optimizer tests completed successfully!")
        # Updated assertion to check for new output format
        assert "Net Profit" in result
        assert "Dividend Income" in result
        assert "Expenses" in result
