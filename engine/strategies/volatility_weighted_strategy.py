import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class VolatilityWeightedStrategy(BaseStrategy):
    """
    Volatility-weighted position sizing strategy.
    
    Adjusts position size based on market volatility:
    - Low volatility (calm market): Buy with full position when model signals
    - High volatility (risky market): Reduce position size (take smaller bets)
    
    Logic:
        Position size = Model probability × (1 - volatility/max_volatility)
        
    This prevents large losses in high-risk environments while staying
    fully invested when markets are calm and predictable.
    """
    
    def __init__(self, volatility_period: int = 20, volatility_percentile: float = 0.75):
        super().__init__("volatility_weighted")
        self.volatility_period = volatility_period
        self.volatility_percentile = volatility_percentile
    
    def apply(self, df: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply volatility weighting to position sizes.
        
        Returns:
            Position signal adjusted for volatility (0-1 scale)
            - 1.0: Full buy when vol is low
            - 0.5: Half position when vol is high
            - 0.0: No position when vol is extreme or probability is low
        """
        close = df['close'] if 'close' in df.columns else df.get('Close', df['close'])
        volatility = self._calculate_volatility(close, self.volatility_period)
        
        # Get historical volatility percentile
        vol_percentile = np.nanpercentile(volatility, self.volatility_percentile * 100)
        
        # Normalize volatility to 0-1 range for weighting
        vol_weight = 1 - (volatility / (vol_percentile + 1e-6))
        vol_weight = np.clip(vol_weight, 0, 1)
        
        # Basic buy signal (probability > threshold)
        buy_signal = (probabilities > threshold).astype(float)
        
        # Weight the signal by volatility adjustment
        weighted_signals = buy_signal * vol_weight
        
        # Convert back to binary (0 or 1) based on weighted value
        signals = (weighted_signals > 0.5).astype(int)
        
        return signals
