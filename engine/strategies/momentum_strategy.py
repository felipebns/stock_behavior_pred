import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.
    
    Modifies buy signals based on price momentum:
    - Strong uptrend (positive momentum): Increase confidence, buy more aggressively
    - Downtrend (negative momentum): Decrease confidence, only buy on very high probability
    
    Logic:
        If model probability > threshold AND recent price momentum is positive:
            → Buy with full position
        If model probability > threshold BUT momentum is negative:
            → Only buy if probability > threshold + momentum_adjustment
    """
    
    def __init__(self, momentum_period: int = 20, momentum_threshold: float = 0.01):
        super().__init__("momentum")
        self.momentum_period = momentum_period
        self.momentum_threshold = momentum_threshold
    
    def apply(self, df: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply momentum filter to buy signals.
        
        - Positive momentum: Accept model signal at base threshold
        - Negative momentum: Require higher probability (threshold + adjustment)
        """
        close = df['close'] if 'close' in df.columns else df.get('Close', df['close'])
        momentum = self._calculate_momentum(close, self.momentum_period)
        
        # Adjust threshold based on momentum
        # If momentum is negative, require higher probability to trade
        adjusted_threshold = threshold - (momentum * 2)  # Momentum boosts confidence
        
        # Buy signal: probability > adjusted threshold
        signals = (probabilities > adjusted_threshold).astype(int)
        
        return signals
