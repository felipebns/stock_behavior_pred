import pytest
import pandas as pd
import numpy as np
from services.backtesting.return_calculator import ReturnCalculator

def test_daily_portfolio_return_calculation():
    """Ensure that ReturnCalculator correctly handles gross returns, transaction costs, and cash interest."""
    calc = ReturnCalculator(transaction_cost=0.01, slippage=0.01, annual_rf_rate=0.0) # No risk-free to simplify
    
    # 2 days of trading 2 stocks
    df = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
        'ticker': ['A', 'B', 'A', 'B'],
        'next_return': [0.10, -0.05, 0.0, 0.0]
    })
    
    # Day 1: Bought A (Weight 0.5), Didn't buy B (Weight 0)
    # Day 2: Sold A (Weight 0.0), Bought B (Weight 0.5)
    positions = np.array([0.5, 0.0, 0.0, 0.5])
    
    returns = calc.calculate(positions, df)
    
    # Let's break down the expected math:
    # Day 1:
    # We moved from 0 to [0.5, 0] --> Cost = 0.5 * (0.01 + 0.01) = 0.01
    # Gross Return = A(0.5 * 0.10) + B(0 * -0.05) = 0.05
    # Net Return = 0.05 (Gross) - 0.01 (Cost) = 0.04
    
    # Day 2:
    # We moved from [0.5, 0] to [0.0, 0.5]
    # Cost = A(abs(0.0-0.5)*0.02) + B(abs(0.5-0.0)*0.02) = 0.01 + 0.01 = 0.02
    # Gross Return = 0.0
    # Net Return = 0.0 - 0.02 = -0.02
    
    np.testing.assert_array_almost_equal(returns, [0.04, -0.02])
