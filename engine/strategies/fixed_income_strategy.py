import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class FixedIncomeStrategy(BaseStrategy):
    """
    Fixed Income (cash) benchmark strategy.
    
    Stay 100% in cash/bonds earning risk-free rate.
    No stock exposure.
    
    Represents: Conservative (no equity risk)
    Use for: Downside reference point
    """
    
    def __init__(self):
        super().__init__("fixed_income")
    
    def apply(self, df: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> np.ndarray:
        """
        Never buy (stay in cash).
        
        Args:
            df: DataFrame (not used)
            probabilities: Ignored
            threshold: Ignored
            
        Returns:
            Always 0 (never buy, stay in cash)
        """
        signals = np.zeros(len(df)).astype(int)
        return signals
