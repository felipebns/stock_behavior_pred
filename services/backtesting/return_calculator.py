import pandas as pd
import numpy as np


class ReturnCalculator:
    """Calculates portfolio returns from positions."""
    
    def __init__(self, transaction_cost: float, slippage: float, annual_rf_rate: float):
        self.tc = transaction_cost
        self.slippage = slippage
        self.daily_rf_rate = (1 + annual_rf_rate) ** (1/252) - 1
    
    def calculate(self, positions: np.ndarray, test_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate daily portfolio returns.
        
        daily_return = Σ(position_i × return_i) - costs + cash_interest
        """
        next_returns = test_df["next_return"].fillna(0).values
        
        # Gross returns
        gross_returns = positions * next_returns
        
        # Trading costs
        test_df_copy = test_df.copy()
        test_df_copy["_pos"] = positions
        position_changes = test_df_copy.groupby("ticker")["_pos"].diff()
        position_changes = position_changes.fillna(test_df_copy["_pos"])
        costs = np.abs(position_changes.values) * (self.tc + self.slippage)
        
        # Aggregate by date (no division by n_assets - that was the bug!)
        net_returns_flat = gross_returns - costs
        test_df_copy["_net_ret"] = net_returns_flat
        daily_returns = test_df_copy.groupby("date")["_net_ret"].sum()
        
        # Cash interest on undeployed capital
        total_deployed = test_df_copy.groupby("date")["_pos"].sum()
        cash_level = 1.0 - total_deployed
        daily_cash_interest = cash_level * self.daily_rf_rate
        
        final_returns = daily_returns + daily_cash_interest
        return final_returns.sort_index().values
