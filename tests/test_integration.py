import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from services.pipeline.pipeline import Pipeline
from services.algorithms.logistic_regression import LogisticRegressionAlgorithm
from services.stock.stock import Stock

class MockStock(Stock):
    def __init__(self):
        super().__init__(['A', 'B'], '2020-01-01', '2021-01-01')
        
    def fetch(self) -> pd.DataFrame:
        """Generate 300 days of mock market data for 2 tickers to survive SMA200 dropna."""
        np.random.seed(42) # Deterministic
        dates = pd.date_range(start='2020-01-01', periods=300, freq='B')
        data = []
        for t in ['A', 'B']:
            returns = np.random.normal(0.0005, 0.015, size=len(dates))
            prices = 100 * np.cumprod(1 + returns)
            for i, d in enumerate(dates):
                data.append({
                    'date': d,
                    'ticker': t,
                    'open': prices[i] * 0.99,
                    'high': prices[i] * 1.01,
                    'low': prices[i] * 0.98,
                    'close': prices[i],
                    'adj_close': prices[i],
                    'volume': 1000 + i * 10
                })
        return pd.DataFrame(data)

def test_pipeline_integration(tmp_path):
    """
    Test End-to-End Integration of Pipeline and Backtesting without real API connectivity.
    Uses lightweight windows and single small Model to guarantee structural integrity.
    """
    mock_stock = MockStock()
    
    # Very light model configuration
    models = [LogisticRegressionAlgorithm(max_iter=10)]
    
    # tmp_path is a pytest fixture providing an isolated temporary directory
    pipeline = Pipeline(
        stock=mock_stock,
        algorithms=models,
        output_dir=str(tmp_path),
        test_size=0.20,  # 20% of remaining ~100 days = ~20 days for Test Set
        wfv_train_window=40,  # Train fold size
        wfv_test_window=10,   # Test fold size
        probability_thresholds=[0.50], # Just one threshold to be fast
        parallelization={
            "algorithm_selection": 1,
            "fold_evaluation": 1,
            "threshold_testing": 1
        }
    )
    
    # Run the pipeline (save=False to just get the Backtest outputs without writing JSON/Plots in tests)
    # Wait, save=True is better to prove writers don't crash
    results = pipeline.run(save=True)
    
    # Validate Backtest Results returned properly
    assert isinstance(results, dict)
    
    # Ensure Benchmarks were calculated
    assert "Buy & Hold (benchmark)" in results
    assert "Fixed Income 5.00% (benchmark)" in results
    
    # Ensure ML logic created at least one proper threshold test key
    ml_keys = [k for k in results.keys() if k.startswith("ML ")]
    assert len(ml_keys) > 0, "Pipeline failed to produce ML strategy outputs."
    
    # Validate structure of the return
    sample_res = results["Buy & Hold (benchmark)"]
    assert "total_return" in sample_res
    assert "sharpe_ratio" in sample_res
    assert "max_drawdown" in sample_res
    assert "positions_by_ticker" in sample_res
