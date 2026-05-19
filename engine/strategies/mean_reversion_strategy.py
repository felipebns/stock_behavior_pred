import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion-based trading strategy.
    
    Exploits the fact that prices tend to revert to their average:
    - When price is far BELOW SMA: Market undervalued, increase buy confidence
    - When price is far ABOVE SMA: Market overvalued, decrease buy confidence
    
    Logic:
        If model says "BUY" AND price is below its 50-day average:
            → Strong buy signal (prices tend to recover)
        If model says "BUY" BUT price is way above its 50-day average:
            → Weak buy signal (prices may pull back)
    """
    
    def __init__(self, sma_period: int = 50, distance_threshold: float = 0.05):
        super().__init__("mean_reversion")
        self.sma_period = sma_period
        self.distance_threshold = distance_threshold
    
    def apply(self, df: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply mean reversion filter to buy signals.
        
        - Price below SMA: Accept signal at base threshold (undervalued)
        - Price above SMA: Require higher probability (overvalued, less attractive)
        """
        close = df['close'] if 'close' in df.columns else df.get('Close', df['close'])
        distance_from_sma = self._calculate_sma_distance(close, self.sma_period)
        
        # Adjust threshold based on mean reversion
        # Negative distance = price below SMA = boost buying confidence
        # Positive distance = price above SMA = require higher conviction
        adjusted_threshold = threshold + (distance_from_sma * 2)
        
        # Buy signal: probability > adjusted threshold
        signals = (probabilities > adjusted_threshold).astype(int)
        
        return signals
