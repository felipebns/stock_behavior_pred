import pytest
import pandas as pd
import numpy as np
from services.backtesting.allocation_manager import AllocationManager

def test_top_k_selection():
    """Verify that only the top K probability positions are kept, others zeroed."""
    df = pd.DataFrame({
        'date': ['2023-01-01'] * 5,
        'ticker': ['A', 'B', 'C', 'D', 'E']
    })
    positions = np.array([1, 1, 1, 1, 1])
    probs = np.array([0.9, 0.4, 0.8, 0.5, 0.6]) # Rank: A (1), B (5), C (2), D (4), E (3)
    
    manager = AllocationManager(df, position_selection="top_3")
    filtered = manager._apply_position_selection(positions, probs)
    
    # Needs to keep Top 3 only (A, C, E)
    np.testing.assert_array_equal(filtered, [1, 0, 1, 0, 1])

def test_probability_weighting():
    """Verify that probabilities are normalized to sum to 1.0 logic per day."""
    df = pd.DataFrame({
        'date': ['2023-01-01'] * 3,
        'ticker': ['A', 'B', 'C']
    })
    probs = np.array([0.5, 0.3, 0.2])
    
    manager = AllocationManager(df)
    weights = manager._apply_probability_weights(probs)
    
    # 0.5 + 0.3 + 0.2 = 1.0 (Sum=1.0)
    np.testing.assert_array_almost_equal(weights, [0.5, 0.3, 0.2])
