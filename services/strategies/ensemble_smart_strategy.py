import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy


class EnsembleSmartStrategy(BaseStrategy):
    """
    Smart ensemble strategy combining multiple technical indicators.
    
    This is the RECOMMENDED strategy - combines:
    1. Momentum (uptrend confirmation)
    2. Mean reversion (valuation levels)
    3. Volatility weighting (risk adjustment)
    4. RSI (overbought/oversold)
    
    Buy signal is STRONGER when multiple conditions align:
    - Model predicts high probability (main signal)
    - Price has positive momentum (trend confirmation)
    - Price is below SMA (valuation discount)
    - Volatility is not extreme (risk-adjusted)
    - RSI not overbought (room to run)
    
    This creates a "consensus" trading system where you only buy when
    the model AND market conditions ALL agree.
    """
    
    def __init__(self, momentum_period: int = 20, sma_period: int = 50, 
                 volatility_period: int = 20, rsi_period: int = 14):
        super().__init__("ensemble_smart")
        self.momentum_period = momentum_period
        self.sma_period = sma_period
        self.volatility_period = volatility_period
        self.rsi_period = rsi_period
    
    def apply(self, df: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply multi-factor trading strategy.
        
        Combines 4 factors into a composite score:
        - Momentum: positive trend = +1, negative = -1
        - Mean reversion: below SMA = +1, above = -1
        - Volatility: low vol = +1, high vol = -1
        - RSI: oversold (<40) = +1, overbought (>60) = -1, neutral = 0
        
        Buy decision:
        - Base (model): probability > threshold
        - Confirmation: Need at least 2 of 4 factors to agree
        - Strength: More factors = more confident
        """
        close = df['close'] if 'close' in df.columns else df.get('Close', df['close'])
        
        # Calculate all technical indicators
        momentum = self._calculate_momentum(close, self.momentum_period)
        distance_from_sma = self._calculate_sma_distance(close, self.sma_period)
        volatility = self._calculate_volatility(close, self.volatility_period)
        rsi = self._calculate_rsi(close, self.rsi_period)
        
        # Normalize volatility to 0-1
        vol_percentile = np.nanpercentile(volatility[~np.isnan(volatility)], 75)
        vol_normalized = np.clip(volatility / (vol_percentile + 1e-6), 0, 1)
        
        # Combine signals into composite score
        # Each factor ranges from -1 (unfavorable) to +1 (favorable)
        
        # 1. Momentum factor: positive momentum is favorable
        momentum_signal = np.sign(momentum)  # -1, 0, or 1
        
        # 2. Mean reversion factor: below SMA is favorable (undervalued)
        mean_reversion_signal = -np.sign(distance_from_sma)  # Below SMA = favorable = +1
        
        # 3. Volatility factor: low volatility is favorable
        volatility_signal = 1 - (2 * vol_normalized)  # High vol = -1, low vol = +1
        
        # 4. RSI factor: oversold (<40) is favorable, overbought (>60) is unfavorable
        rsi_signal = np.zeros_like(rsi)
        rsi_signal[rsi < 40] = 1      # Oversold (bullish)
        rsi_signal[rsi > 60] = -1     # Overbought (bearish)
        # Between 40-60 remains 0 (neutral)
        
        # Composite factor score (-4 to +4 scale)
        factor_score = momentum_signal + mean_reversion_signal + volatility_signal + rsi_signal
        
        # Base model signal
        base_signal = (probabilities > threshold).astype(float)
        
        # Smart ensemble: buy when model agrees AND at least 1.5 factors confirm (2+ out of 4)
        # This creates a "consensus" approach
        factor_confirmation = (factor_score > 1).astype(float)
        
        # Final signal: model probability × factor confirmation
        final_signals = base_signal * factor_confirmation
        
        # Convert to binary
        signals = (final_signals > 0).astype(int)
        
        return signals
