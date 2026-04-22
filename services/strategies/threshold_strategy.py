import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class ThresholdStrategy(BaseStrategy):
    """
    Simple threshold-based strategy (baseline).
    
    Buy when model probability exceeds a given threshold.
    No additional technical filters - just raw model confidence.
    
    Useful as: baseline comparison, simple reference point
    """
    
    def __init__(self):
        super().__init__("threshold")
    
    def apply(self, df: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> np.ndarray:
        """
        Buy when probability > threshold.
        
        Args:
            df: DataFrame (not used, but required by interface)
            probabilities: Model probabilities (0-1)
            threshold: Probability threshold
            
        Returns:
            Binary signal (0 or 1)
        """
        signals = (probabilities > threshold).astype(int)
        return signals
