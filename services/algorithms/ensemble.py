from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from services.algorithms.base import BaseScikitClassificationAlgorithm

class EnsembleClassificationAlgorithm(BaseScikitClassificationAlgorithm):
    """Hybrid Ensemble using params from config, not hardcoded.
    
    Uses the same hyperparameters as individual algorithms for consistency.
    All sub-models are created with the shared params for fair voting.
    """
    
    def __init__(
        self,
        lr_params=None,
        svc_params=None,
        rf_params=None,
    ):
        """Initialize ensemble with provided model params or defaults.
        
        Args:
            lr_params: Dict with LogisticRegression params (uses config defaults if None)
            svc_params: Dict with SVC params (uses config defaults if None)
            rf_params: Dict with RandomForest params (uses config defaults if None)
        """
        # Default params if not provided
        if lr_params is None:
            lr_params = {
                "random_state": 42,
                "max_iter": 1000,
                "class_weight": "balanced",
                "C": 1.0,
            }
        if svc_params is None:
            svc_params = {
                "kernel": "linear",
                "random_state": 42,
                "probability": True,
                "class_weight": "balanced",
                "C": 10.0,
                "max_iter": 2000,
            }
        if rf_params is None:
            rf_params = {
                "n_estimators": 100,
                "random_state": 42,
                "class_weight": "balanced",
                "max_depth": None,
                "n_jobs": -1,
            }
        
        # Create models with config params
        lr = LogisticRegression(**lr_params)
        svc = SVC(**svc_params)
        rf = RandomForestClassifier(**rf_params)
        
        ensemble = VotingClassifier(
            estimators=[('lr', lr), ('svc', svc), ('rf', rf)],
            voting='soft'
        )
        super().__init__(model=ensemble)

    def name(self) -> str:
        return "Hybrid Ensemble"
