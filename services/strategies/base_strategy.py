from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    A strategy takes model probabilities (buy signals) and modifies trading decisions
    based on additional market conditions (momentum, volatility, mean reversion, etc).
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def apply(self, df: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply strategy to model predictions.
        
        Args:
            df: DataFrame with OHLCV data (must have 'Close', 'Volume', etc.)
            probabilities: Model probability predictions (0-1)
            threshold: Probability threshold for initial buy signal
            
        Returns:
            Modified position signal (0=sell, 1=buy)
        """
        pass
    
    def _calculate_momentum(self, close_prices: pd.Series, period: int = 20) -> np.ndarray:
        """Calculate price momentum (returns)."""
        returns = close_prices.pct_change(period)
        return returns.fillna(0).values
    
    def _calculate_volatility(self, close_prices: pd.Series, period: int = 20) -> np.ndarray:
        """Calculate rolling volatility (standard deviation of returns)."""
        returns = close_prices.pct_change()
        volatility = returns.rolling(window=period).std()
        return volatility.fillna(volatility.mean()).values
    
    def _calculate_rsi(self, close_prices: pd.Series, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index (0-100, >70=overbought, <30=oversold)."""
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values
    
    def _calculate_sma_distance(self, close_prices: pd.Series, period: int = 50) -> np.ndarray:
        """
        Calculate distance from SMA (mean reversion indicator).
        Positive: price above SMA (overbought)
        Negative: price below SMA (oversold)
        """
        sma = close_prices.rolling(window=period).mean()
        distance = (close_prices - sma) / sma
        return distance.fillna(0).values
