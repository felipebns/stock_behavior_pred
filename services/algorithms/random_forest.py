from sklearn.ensemble import RandomForestClassifier
from services.algorithms.base import BaseScikitClassificationAlgorithm

class RandomForestAlgorithm(BaseScikitClassificationAlgorithm):
    def __init__(self, n_estimators=100, random_state=42, class_weight="balanced", 
                 max_depth=None, n_jobs=-1):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
            max_depth=max_depth,
            n_jobs=n_jobs
        )
        super().__init__(model=model)

    def name(self) -> str:
        return "Random Forest"
