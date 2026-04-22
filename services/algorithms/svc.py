from sklearn.svm import SVC
from services.algorithms.base import BaseScikitClassificationAlgorithm

class SVCAlgorithm(BaseScikitClassificationAlgorithm):
    def __init__(self, kernel="linear", random_state=42, probability=True, 
                 class_weight="balanced", C=1.0, max_iter=2000):
        model = SVC(
            kernel=kernel,
            random_state=random_state,
            probability=probability,
            class_weight=class_weight,
            C=C,
            max_iter=max_iter
        )
        super().__init__(model=model)

    def name(self) -> str:
        return "Support Vector Classifier"
