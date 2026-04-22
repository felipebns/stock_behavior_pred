import pandas as pd
import numpy as np
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import spearmanr

from services.algorithms.base import Algorithm


class MetricsEvaluator:
    """Evaluates ML model performance on test set."""
    
    def evaluate(self, algorithm: Algorithm, test_df: pd.DataFrame, 
                 features: list[str], target_col: str) -> Dict[str, float]:
        """Calculate pure ML classification metrics."""
        y_true = test_df[target_col].to_numpy()
        y_pred = algorithm.predict(test_df, features)
        y_prob = algorithm.predict_proba(test_df, features)
        actual_returns = test_df["next_return"].fillna(0).to_numpy()
        
        metrics = {
            "accuracy": self._safe_metric(lambda: accuracy_score(y_true, y_pred)),
            "precision": self._safe_metric(lambda: precision_score(y_true, y_pred, zero_division=0)),
            "recall": self._safe_metric(lambda: recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": self._safe_metric(lambda: f1_score(y_true, y_pred, zero_division=0)),
            "auc": self._safe_metric(lambda: roc_auc_score(y_true, y_prob), default=0.5),
            "information_coefficient": self._calculate_ic(y_prob, actual_returns),
            "spearman_correlation": self._calculate_spearman(y_prob, actual_returns)
        }
        
        self._print_metrics(algorithm.name(), metrics)
        return metrics
    
    def _safe_metric(self, func, default=0.0) -> float:
        """Safely calculate metric with error handling."""
        try:
            result = func()
            return float(result) if not np.isnan(result) else default
        except:
            return default
    
    def _calculate_ic(self, y_prob: np.ndarray, actual_returns: np.ndarray) -> float:
        """Calculate Information Coefficient."""
        if len(y_prob) <= 1:
            return 0.0
        ic = float(np.corrcoef(y_prob, actual_returns)[0, 1])
        return 0.0 if np.isnan(ic) else ic
    
    def _calculate_spearman(self, y_prob: np.ndarray, actual_returns: np.ndarray) -> float:
        """Calculate Spearman rank correlation."""
        try:
            corr, _ = spearmanr(y_prob, actual_returns)
            return float(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def _print_metrics(self, name: str, metrics: Dict) -> None:
        """Print evaluation metrics."""
        print(f"[{name}] Acc: {metrics['accuracy']:.4f} | Prec: {metrics['precision']:.4f} | "
              f"Rec: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f} | "
              f"AUC: {metrics['auc']:.4f} | IC: {metrics['information_coefficient']:.4f} | "
              f"Spearman: {metrics['spearman_correlation']:.4f}")
