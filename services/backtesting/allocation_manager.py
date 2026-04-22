from services.backtesting.position_normalizer import PositionNormalizer
from services.log.logger_config import get_logger
import pandas as pd
import numpy as np

logger = get_logger()


class AllocationManager:
    """
    Capital allocation orchestrator.
    
    Consolidates the three-step allocation pipeline while maintaining
    the exact logic of the original implementation:
    1. Top-K asset selection
    2. Probability weighting
    3. Position normalization (with confidence gate)
    
    Separates allocation logic from backtesting engine for better
    testability and maintainability.
    """
    
    def __init__(self, test_df: pd.DataFrame, 
                 position_selection: str = "top_5",
                 position_sizing: str = "equal_weight",
                 allocation_mode: str = "full_deployment",
                 purchase_threshold: float = 0.50):
        """
        Initialize allocation manager.
        
        Args:
            test_df: DataFrame with date and ticker columns
            position_selection: "all", "top_1", "top_5", etc.
            position_sizing: "equal_weight" or "probability_weighted"
            allocation_mode: "full_deployment" or "cash_allocation"
            purchase_threshold: Minimum confidence to invest (confidence gate)
        """
        self.test_df = test_df
        self.position_selection = position_selection
        self.position_sizing = position_sizing
        self.allocation_mode = allocation_mode
        self.purchase_threshold = purchase_threshold
    
    def allocate(self, positions: np.ndarray, probabilities: np.ndarray,
                 strategy_name: str) -> np.ndarray:
        """
        Execute complete allocation pipeline.
        
        Pipeline (same logic as original):
        1. Apply top-K filtering (skip for benchmarks)
        2. Apply probability weighting if enabled
        3. Normalize positions (apply allocation mode + confidence gate)
        
        Args:
            positions: Binary positions (0/1) from strategy
            probabilities: Model probabilities for weighting
            strategy_name: Strategy name (skip allocation for benchmarks)
        
        Returns:
            Allocated and normalized positions
        """
        n_selected_before = np.sum(positions > 0)
        
        # STEP 1: Top-K filtering (skip for benchmarks)
        if strategy_name not in ["fixed_income", "buy_and_hold"]:
            positions = self._apply_position_selection(positions, probabilities)
            n_selected_after_topk = np.sum(positions > 0)
            logger.debug(f"[{strategy_name}] Top-K Filter ({self.position_selection}): "
                        f"{n_selected_before} → {n_selected_after_topk} selected")
        
        # STEP 2: Probability weighting (skip for benchmarks)
        if self.position_sizing == "probability_weighted" and \
            strategy_name not in ["fixed_income", "buy_and_hold", "simple_reversal"]:
            # Weight the binary positions by their probabilities
            positions = positions * probabilities
            positions = self._apply_probability_weights(positions)
            logger.debug(f"[{strategy_name}] Probability weighting applied (weighted mode)")
        
        # STEP 3: Normalize positions (apply allocation mode + confidence gate)
        positions = self._normalize_positions(positions, probabilities)
        n_final = np.sum(positions > 0)
        logger.debug(f"[{strategy_name}] Normalization complete: "
                    f"allocation_mode={self.allocation_mode}, "
                    f"purchase_threshold={self.purchase_threshold}, "
                    f"final_active={n_final}")
        
        return positions
    
    def _apply_position_selection(self, positions: np.ndarray, 
                                   probabilities: np.ndarray) -> np.ndarray:
        """Filter positions keeping only top-K by probability."""
        if self.position_selection == "all":
            return positions
        
        if not self.position_selection.startswith("top_"):
            return positions
        
        try:
            n_top = int(self.position_selection.split("_")[1])
        except (IndexError, ValueError):
            return positions
        
        filtered_positions = positions.copy()
        
        # Fast vectorized top-K selection using pandas groupby
        df_temp = pd.DataFrame({
            'date': self.test_df['date'].values,
            'pos': filtered_positions,
            'prob': probabilities,
            'idx': np.arange(len(filtered_positions))
        })
        
        # Only consider active positions
        active = df_temp[df_temp['pos'] > 0]
        
        if not active.empty:
            # Find the top K for each date
            top_k = active.groupby('date', group_keys=False).apply(
                lambda x: x.nlargest(n_top, 'prob')
            )
            
            # Reset all to 0, then set only the top K to 1
            filtered_positions.fill(0)
            if not top_k.empty:
                filtered_positions[top_k['idx'].values] = 1
                
        return filtered_positions
    
    def _apply_probability_weights(self, probabilities: np.ndarray) -> np.ndarray:
        """Normalize probabilities per date."""
        weights = np.zeros_like(probabilities)
        
        # Fast vectorized probability weighting
        df_temp = pd.DataFrame({
            'date': self.test_df['date'].values,
            'prob': probabilities
        })
        
        # Only sum positive probabilities per date
        df_temp['pos_prob'] = np.where(df_temp['prob'] > 0, df_temp['prob'], 0)
        daily_sums = df_temp.groupby('date')['pos_prob'].transform('sum')
        
        # Calculate weights (avoiding division by zero)
        mask = (daily_sums > 0) & (df_temp['pos_prob'] > 0)
        weights[mask] = df_temp['pos_prob'][mask] / daily_sums[mask]
        
        return weights
    
    def _normalize_positions(self, positions: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        """Normalize positions based on allocation mode.
        
        This delegates to PositionNormalizer to maintain exact original logic.
        The confidence gate is applied INSIDE the normalization, which is the
        critical difference from the broken refactored version.
        
        Args:
            positions: Position array (binary or weighted)
            probabilities: Array of probability values
        
        Returns:
            Normalized positions respecting allocation mode and confidence gate
        """
        
        normalizer = PositionNormalizer()
        return normalizer.normalize(
            positions, 
            probabilities,
            self.test_df,
            allocation_mode=self.allocation_mode,
            purchase_threshold=self.purchase_threshold
        )
