import pandas as pd
import numpy as np
from typing import Dict


class MetricsCalculator:
    """Calculates performance metrics for a strategy."""
    
    def __init__(self, initial_capital: float, annual_rf_rate: float):
        self.initial_capital = initial_capital
        self.daily_rf_rate = (1 + annual_rf_rate) ** (1/252) - 1
    
    def calculate(self, strategy_returns: np.ndarray, positions: np.ndarray, 
                  test_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics."""
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1
        total_return = cumulative_returns[-1]
        equity_curve = self.initial_capital * (1 + cumulative_returns)
        
        return {
            "total_return": float(total_return),
            "annualized_return": self._calculate_annualized_return(total_return, len(strategy_returns)),
            "sharpe_ratio": self._calculate_sharpe_ratio(strategy_returns),
            "max_drawdown": self._calculate_max_drawdown(equity_curve),
            "active_hit_rate": self._calculate_hit_rate(positions, test_df),
            "final_equity": float(equity_curve[-1]),
            "equity_curve": equity_curve,
            "daily_returns": strategy_returns,
            "positions": positions
        }
    
    def _calculate_annualized_return(self, total_return: float, n_days: int) -> float:
        """Convert total return to annualized return."""
        years = n_days / 252
        if years > 0:
            return float((1 + total_return) ** (1 / years) - 1)
        return 0.0
    
    def _calculate_sharpe_ratio(self, strategy_returns: np.ndarray) -> float:
        """Calculate Sharpe ratio with edge case protection."""
        std_returns = np.std(strategy_returns)
        mean_returns = np.mean(strategy_returns)
        
        if std_returns > 1e-8:
            sharpe = (mean_returns / std_returns) * np.sqrt(252)
            return float(0.0 if sharpe > 10 else sharpe)
        return 0.0
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown from peak."""
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return float(np.min(drawdown))
    
    def _calculate_hit_rate(self, positions: np.ndarray, test_df: pd.DataFrame) -> float:
        """Calculate hit rate: % of days with positive returns when active."""
        next_returns = test_df["next_return"].fillna(0).values
        gross_returns = positions * next_returns
        
        is_active = (positions > 0)
        active_days = np.sum(is_active)
        
        if active_days > 0:
            return float(np.sum(gross_returns[is_active] > 0) / active_days)
        return 0.0
