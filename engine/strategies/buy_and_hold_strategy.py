import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """
    Buy & Hold benchmark strategy.
    
    Buy on day 1 and hold forever.
    No selling, no timing.
    
    Represents: Passive index investment
    Use for: Baseline comparison
    """
    
    def __init__(self):
        super().__init__("buy_and_hold")
    
    def apply(self, df: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> np.ndarray:
        """
        Always buy (hold position).
        
        Args:
            df: DataFrame (not used)
            probabilities: Ignored
            threshold: Ignored
            
        Returns:
            Always 1 (always buy/hold)
        """
        signals = np.ones(len(df)).astype(int)
        return signals
