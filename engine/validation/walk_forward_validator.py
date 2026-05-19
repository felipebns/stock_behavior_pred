import pandas as pd
import numpy as np
import copy
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, List

class WalkForwardValidator:
    """Walk-Forward Validation for model selection without data leakage.
    
    Uses GROWING WINDOW approach:
    - Train on [0:train_end], test on [train_end:test_end]
    - Each iteration rolls BOTH boundaries forward by test_window
    - No overlap or reuse of test data in training
    - Models strictly isolated per fold using deepcopy
    - Uses Information Coefficient (IC) as selection metric
    """
    
    def __init__(self, train_window: int = 1000, test_window: int = 250):
        """Initialize validator with window sizes (in dates, not rows).
        
        Args:
            train_window: Number of unique dates to train on
            test_window: Number of unique dates to test on
        """
        self.train_window = train_window
        self.test_window = test_window
    
    def validate(
        self,
        df: pd.DataFrame,
        algorithm,
        features: list[str],
        target_col: str
    ) -> Tuple[pd.DataFrame, np.ndarray, List[float], float, float]:
        """Perform walk-forward validation with parallel fold evaluation.
        
        Uses ROLLING WINDOW approach with FIXED train size:
        Example with train_window=750, test_window=250:
            Iter 0: train[0:750],      test[750:1000]
            Iter 1: train[250:1000],   test[1000:1250]
            Iter 2: train[500:1250],   test[1250:1500]
        
        Each fold trains on exactly train_window unique dates (fair model comparison).
        
        Returns:
            Tuple containing:
            - wfv_df: DataFrame with out-of-sample test data
            - wfv_predictions: Array with out-of-sample predictions (0/1)
            - fold_ics: List of Information Coefficients for each fold
            - mean_ic: Mean IC across folds (PRIMARY SELECTION METRIC)
            - std_ic: Std dev of IC across folds
        """
        # Prepare fold boundaries
        dates = df["date"].sort_values().unique()
        n_dates = len(dates)
        
        fold_specs = []
        train_end = self.train_window
        test_end = self.train_window + self.test_window
        
        while test_end <= n_dates:
            train_start = max(0, train_end - self.train_window)  # Fixed window size
            split_date_start = dates[train_end]
            if test_end < n_dates:
                split_date_end = dates[test_end]
                test_df_mask = (df["date"] >= split_date_start) & (df["date"] < split_date_end)
            else:
                test_df_mask = (df["date"] >= split_date_start)
            
            # Fixed-size rolling window: only use most recent train_window dates
            train_df = df[(df["date"] >= dates[train_start]) & (df["date"] < split_date_start)]
            test_df = df[test_df_mask]
            
            fold_specs.append({
                'fold_idx': len(fold_specs),
                'train_df': train_df,
                'test_df': test_df,
            })
            
            train_end = test_end
            if test_end == n_dates:
                break
            test_end = min(train_end + self.test_window, n_dates)
        
        # Parallel fold evaluation
        all_predictions = []
        all_probs = []
        all_test_indices = []
        fold_ics = []
        
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            fold_futures = {
                executor.submit(
                    self._evaluate_fold,
                    algorithm,
                    spec['train_df'],
                    spec['test_df'],
                    features,
                    target_col
                ): spec['fold_idx']
                for spec in fold_specs
            }
            
            # Collect results as they complete (non-blocking)
            fold_results = {}
            for future in concurrent.futures.as_completed(fold_futures):
                fold_idx = fold_futures[future]
                predictions, probs, ic, test_indices = future.result()
                fold_results[fold_idx] = (predictions, probs, ic, test_indices)
        
        # Reconstruct in order
        for fold_idx in sorted(fold_results.keys()):
            predictions, probs, ic, test_indices = fold_results[fold_idx]
            all_predictions.extend(predictions)
            all_probs.extend(probs)
            all_test_indices.extend(test_indices)
            fold_ics.append(ic)
        
        # Reconstruct aggregated OOS dataframe and predictions
        wfv_df = df.loc[all_test_indices].copy()
        wfv_predictions = np.array(all_predictions)
        wfv_probs = np.array(all_probs)
        
        mean_ic = float(np.mean(fold_ics)) if fold_ics else 0.0
        std_ic = float(np.std(fold_ics)) if fold_ics else 0.0
        
        return wfv_df, wfv_predictions, wfv_probs, fold_ics, mean_ic, std_ic
    
    @staticmethod
    def _evaluate_fold(algorithm, train_df: pd.DataFrame, test_df: pd.DataFrame,
                      features: list[str], target_col: str) -> Tuple:
        """Evaluate algorithm on a single fold.
        
        Static method to be thread-safe for ThreadPoolExecutor.
        
        Returns:
            (predictions, probabilities, ic, test_indices)
        """
        
        # Deep clone algorithm to ensure isolation
        fold_algo = copy.deepcopy(algorithm)
        
        # Train on historical data
        fold_algo.fit(train_df, pd.DataFrame(), features, target_col)
        
        # Predict on blind test period
        predictions = fold_algo.predict(test_df, features)
        y_prob = fold_algo.predict_proba(test_df, features)
        
        # Calculate Information Coefficient
        actual_returns = test_df["next_return"].fillna(0).to_numpy()
        ic = float(np.corrcoef(y_prob, actual_returns)[0, 1]) if len(y_prob) > 1 else 0.0
        ic = 0.0 if np.isnan(ic) else ic
        
        test_indices = test_df.index.tolist()
        
        return (predictions, y_prob, ic, test_indices)
