import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class SimpleReversalStrategy(BaseStrategy):
    """
    Simple mean reversion benchmark strategy.
    
    Buy when price < 20-day SMA (simple benchmark, not using ML).
    No model predictions - just technical analysis.
    
    Represents: Classic technical analysis approach
    Use for: Technical vs ML comparison
    """
    
    def __init__(self, sma_period: int = 20):
        super().__init__("simple_reversal")
        self.sma_period = sma_period
    
    def apply(self, df: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> np.ndarray:
        """
        Buy when price < SMA (ignores model probabilities).
        
        Args:
            df: DataFrame with OHLCV data
            probabilities: Ignored
            threshold: Ignored
            
        Returns:
            Binary signal (1 when price < SMA, 0 otherwise)
        """
        close = df['Close'] if 'Close' in df.columns else df['close']
        sma = close.rolling(window=self.sma_period).mean()
        
        signals = (close < sma).astype(int)
        # Handle initial NaN values
        signals = signals.fillna(0).astype(int)
        
        return signals.values
