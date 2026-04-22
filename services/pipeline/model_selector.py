import time
import copy
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from concurrent.futures import ProcessPoolExecutor

from services.validation.walk_forward_validator import WalkForwardValidator
from services.algorithms.base import Algorithm
from services.log.logger_config import get_logger


class ModelSelector:
    """Selects best ML model using walk-forward validation with parallel evaluation."""
    
    def __init__(self, wfv_train_window: int, wfv_test_window: int, 
                 max_workers: int = 2, fold_workers: int = 4):
        self.wfv_train_window = wfv_train_window
        self.wfv_test_window = wfv_test_window
        self.max_workers = max_workers
        self.fold_workers = fold_workers
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        self.results = {}
        self.logger = get_logger()
    
    def select(self, algorithms: list[Algorithm], train_df: pd.DataFrame, 
               features: list[str], target_col: str) -> Tuple[Algorithm, str, float]:
        """Select best algorithm using parallel walk-forward validation.
        
        Returns:
            (best_algorithm, algorithm_name, best_score)
        """
        self.logger.info(f"Evaluating {len(algorithms)} algorithms in parallel (max_workers={self.max_workers})...")
        
        results_data = {}
        
        # Parallel evaluation using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._evaluate_algorithm,
                    algo,
                    train_df,
                    features,
                    target_col,
                    self.wfv_train_window,
                    self.wfv_test_window
                ): algo.name()
                for algo in algorithms
            }
            
            # Collect results as they complete
            for future in futures:
                try:
                    algo_name, fold_ics, mean_ic, std_ic, score, elapsed, acc, f1, auc = future.result()
                    
                    results_data[algo_name] = {
                        'fold_ics': fold_ics,
                        'mean_ic': mean_ic,
                        'std_ic': std_ic,
                        'score': score,
                        'elapsed': elapsed,
                        'accuracy': acc,
                        'f1_score': f1,
                        'auc': auc
                    }
                    
                    self._print_wfv_results(algo_name, fold_ics, mean_ic, std_ic, score, elapsed, acc, f1, auc)
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating algorithm: {e}", exc_info=True)
                    raise
        
        # Select best model
        self.best_model_name = max(results_data, key=lambda k: results_data[k]['score'])
        self.best_score = results_data[self.best_model_name]['score']
        
        # Get the actual model object
        self.best_model = self._get_algorithm_by_name(algorithms, self.best_model_name)
        
        # Store results
        self.results = results_data
        
        return self.best_model, self.best_model_name, self.best_score
    
    @staticmethod
    def _evaluate_algorithm(algorithm, train_df: pd.DataFrame, 
                           feature_cols: list, target_col: str,
                           train_window: int, test_window: int) -> Tuple:
        """Evaluate a single algorithm via walk-forward validation.
        
        Static method to be pickle-able for ProcessPoolExecutor.
        """
        start_time = time.time()
        
        validator = WalkForwardValidator(train_window, test_window)
        algo_copy = copy.deepcopy(algorithm)
        
        wfv_df, predictions, probs, fold_ics, mean_ic, std_ic = validator.validate(
            train_df, algo_copy, feature_cols, target_col
        )
        
        # Calculate extra ML metrics on aggregated out-of-sample predictions
        y_true = wfv_df[target_col].to_numpy()
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        try:
            acc = float(accuracy_score(y_true, predictions))
        except:
            acc = 0.0
            
        try:
            f1 = float(f1_score(y_true, predictions, zero_division=0))
        except:
            f1 = 0.0
            
        try:
            auc = float(roc_auc_score(y_true, probs))
        except:
            auc = 0.5
        
        elapsed = time.time() - start_time
        score = mean_ic - (0.5 * std_ic)
        
        return (algo_copy.name(), fold_ics, mean_ic, std_ic, score, elapsed, acc, f1, auc)
    
    def _get_algorithm_by_name(self, algorithms: list[Algorithm], name: str) -> Algorithm:
        """Get algorithm object by name."""
        for algo in algorithms:
            if algo.name() == name:
                return algo
        raise ValueError(f"Algorithm '{name}' not found")
    
    def _print_wfv_results(self, name: str, fold_ics: list, mean_ic: float, 
                          std_ic: float, score: float, elapsed: float,
                          acc: float, f1: float, auc: float) -> None:
        """Print walk-forward validation results."""
        print(f"\n[Phase 1: Selection] WFV for {name}...")
        print(f"  -> Fold ICs:     {[round(ic, 4) for ic in fold_ics]}")
        print(f"  -> Mean IC:      {mean_ic:.6f} ± {std_ic:.6f}")
        print(f"  -> Robust Score: {score:.6f}")
        print(f"  -> ML Metrics:   Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
        print(f"  -> Time:         {elapsed:.2f}s")
