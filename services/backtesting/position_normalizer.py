import pandas as pd
import numpy as np
from services.log.logger_config import get_logger

logger = get_logger()

class PositionNormalizer:
    """Normalizes position vectors to capital fractions.
    
    Pipeline:
    1. GATE: Check if best_signal meets purchase_threshold (confidence gate)
       -> If NO -> 100% CASH (no position opened)
       -> If YES -> continue to normalization
    
    2. NORMALIZE: Apply allocation mode rules
       - cash_allocation: Deploy only among selected, rest CASH
       - full_deployment: Deploy 100% equally among selected
    """
    
    def __init__(self, purchase_threshold: float = 0.50, 
                 allocation_mode: str = "full_deployment"):
        self.purchase_threshold = purchase_threshold
        self.allocation_mode = allocation_mode
    
    def normalize(self, positions: np.ndarray, probabilities: np.ndarray, test_df: pd.DataFrame, 
                  allocation_mode: str = None,
                  purchase_threshold: float = None) -> np.ndarray:
        alloc_mode = allocation_mode if allocation_mode is not None else self.allocation_mode
        threshold = purchase_threshold if purchase_threshold is not None else self.purchase_threshold
        
        is_binary = np.all((positions == 0) | (positions == 1))
        
        if is_binary:
            return self._normalize_binary_positions(
                positions, probabilities, test_df, alloc_mode, threshold
            )
        
        return self._normalize_weighted_positions(
            positions, probabilities, test_df, alloc_mode, threshold
        )
    
    def _normalize_binary_positions(self, positions: np.ndarray, 
                                    probabilities: np.ndarray,
                                    test_df: pd.DataFrame,
                                    allocation_mode: str,
                                    purchase_threshold: float) -> np.ndarray:
        df = pd.DataFrame({
            'date': test_df['date'].values,
            'pos': positions,
            'prob': probabilities
        })
        
        # Vectorized grouping metrics
        grouped = df.groupby('date')
        active_counts = grouped['pos'].transform('sum')
        best_signals = grouped['prob'].transform('max')
        total_assets = grouped['date'].transform('count')
        
        normalized = np.zeros_like(positions, dtype=float)
        
        # Valid conditions (Confidence Gate passing & Activity check)
        valid_mask = (best_signals >= purchase_threshold) & (active_counts > 0)
        
        if allocation_mode == "cash_allocation":
            weight = 1.0 / total_assets
        else:
            # Full deployment uses inverse of active signals count
            weight = 1.0 / np.where(active_counts > 0, active_counts, 1)
            
        # Apply weights where validity passes
        normalized[valid_mask] = positions[valid_mask] * weight[valid_mask]
        
        return normalized
    
    def _normalize_weighted_positions(self, positions: np.ndarray,
                                      probabilities: np.ndarray,
                                      test_df: pd.DataFrame,
                                      allocation_mode: str,
                                      purchase_threshold: float) -> np.ndarray:
        df = pd.DataFrame({
            'date': test_df['date'].values,
            'pos': positions,
            'prob': probabilities
        })
        
        # Vectorized grouping metrics
        grouped = df.groupby('date')
        totals = grouped['pos'].transform('sum')
        best_signals = grouped['prob'].transform('max')
        
        normalized = np.zeros_like(positions, dtype=float)
        
        # Valid conditions (Confidence Gate passing & Activity check)
        valid_mask = (best_signals >= purchase_threshold) & (totals > 0)
        
        if allocation_mode == "cash_allocation":
            normalized[valid_mask] = positions[valid_mask]
        else:
            normalized[valid_mask] = positions[valid_mask] / totals[valid_mask]
            
        return normalized
