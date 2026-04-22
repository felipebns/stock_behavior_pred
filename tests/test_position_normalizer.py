import pytest
import pandas as pd
import numpy as np
from services.backtesting.position_normalizer import PositionNormalizer

def test_full_deployment_success():
    """Test standard normalization where entire portfolio is divided among selected stocks."""
    df = pd.DataFrame({'date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01']})
    positions = np.array([1, 1, 0, 0]) # Model chose 2 actions
    probs = np.array([0.65, 0.70, 0.40, 0.30]) # Best signal is 0.70 (> 0.50 threshold)
    
    normalizer = PositionNormalizer(purchase_threshold=0.50, allocation_mode="full_deployment")
    res = normalizer.normalize(positions, probs, df)
    
    # Active count is 2. Both should receive 50% weight (1.0 / 2)
    np.testing.assert_array_almost_equal(res, [0.50, 0.50, 0.0, 0.0])

def test_cash_allocation_success():
    """Test cash_allocation where positions only take their fraction of total universe, leaving rest in cash."""
    df = pd.DataFrame({'date': ['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01']})
    positions = np.array([1, 1, 0, 0]) # Model chose 2 actions
    probs = np.array([0.65, 0.70, 0.40, 0.30]) # Best signal is 0.70
    
    normalizer = PositionNormalizer(purchase_threshold=0.50, allocation_mode="cash_allocation")
    res = normalizer.normalize(positions, probs, df)
    
    # Total assets is 4. Weight should be 25% (1.0 / 4). Wait, remaining 50% is unallocated (CASH)
    np.testing.assert_array_almost_equal(res, [0.25, 0.25, 0.0, 0.0])

def test_confidence_gate_fallback_to_cash():
    """Test the 'Global Gate' logic: if best signal doesn't beat threshold, allocate 0 to ALL."""
    df = pd.DataFrame({'date': ['2023-01-01', '2023-01-01']})
    positions = np.array([1, 1]) 
    probs = np.array([0.45, 0.49]) # Max is 0.49. Fails the 0.50 threshold!
    
    normalizer = PositionNormalizer(purchase_threshold=0.50, allocation_mode="full_deployment")
    res = normalizer.normalize(positions, probs, df)
    
    # Even if positions were requested, global gate cuts them to zero (100% Cash Fallback)
    np.testing.assert_array_almost_equal(res, [0.0, 0.0])
