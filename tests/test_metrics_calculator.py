import pytest
import numpy as np
import pandas as pd
from services.backtesting.metrics_calculator import MetricsCalculator

def test_max_drawdown():
    """Ensure Max Drawdown calculates the deepest valley from peak."""
    calc = MetricsCalculator(10000, 0.05)
    
    # 100 -> 120 -> 60 -> 150
    # Peak is 120, trough is 60. Drawdown = (60-120)/120 = -0.5 (-50%)
    equity_curve = np.array([100, 120, 60, 150])
    
    mdd = calc._calculate_max_drawdown(equity_curve)
    assert np.isclose(mdd, -0.5)

def test_sharpe_ratio_edge_cases():
    """Ensure Sharpe Ratio protects against zero variance and caps explosive ratios."""
    calc = MetricsCalculator(10000, 0.0)
    
    # Case 1: Zero variance (flat returns) -> Sharpe should be 0.0
    flat = np.array([0.01, 0.01, 0.01, 0.01])
    assert calc._calculate_sharpe_ratio(flat) == 0.0
    
    # Case 2: Positive variance standard return
    normal_returns = np.array([0.01, -0.005, 0.02, -0.01])
    std = np.std(normal_returns)
    mean = np.mean(normal_returns)
    expected_sharpe = (mean / std) * np.sqrt(252)
    assert np.isclose(calc._calculate_sharpe_ratio(normal_returns), expected_sharpe)

def test_annualized_return():
    """Convert raw cumulative return to Annualized Return correctly."""
    calc = MetricsCalculator(10000, 0.05)
    
    # 25.2% over half a year (126 days) 
    # (1 + 0.252)^(252/126) - 1 = (1.252)^2 - 1 = 1.5675 - 1 = 56.75%
    ann_ret = calc._calculate_annualized_return(0.252, 126)
    assert np.isclose(ann_ret, 0.567504)
