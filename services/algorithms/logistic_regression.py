from sklearn.linear_model import LogisticRegression
from services.algorithms.base import BaseScikitClassificationAlgorithm

class LogisticRegressionAlgorithm(BaseScikitClassificationAlgorithm):
    def __init__(self, random_state=42, max_iter=1000, class_weight="balanced", C=1.0):
        model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            class_weight=class_weight,
            C=C
        )
        super().__init__(model=model)

    def name(self) -> str:
        return "Logistic Regression"
