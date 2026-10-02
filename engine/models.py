"""Classifiers the walk-forward can fit: name → (label shown in the app, factory(params, n_rows))."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _lightgbm(params: dict, n_rows: int):
    """min_child_samples grows with the training rows: share × rows, never below the floor."""
    params = dict(params)
    share, floor = params.pop("min_child_share"), params.pop("min_child_floor")
    return LGBMClassifier(**params, min_child_samples=max(floor, round(share * n_rows)))


class RowwiseLogisticRegression(LogisticRegression):
    """Scores each row with its own elementwise sum instead of a matrix product: BLAS rounds a row differently depending
    on how many rows come with it, and a day's prediction must not change with the days predicted alongside it."""

    def decision_function(self, X):
        return (np.asarray(X, dtype=self.coef_.dtype) * self.coef_[0]).sum(axis=1) + self.intercept_[0]


def _sklearn(estimator, scale: bool = False):
    """sklearn models don't accept NaN: missing features get the training rows' median (and, for scale, the training
    rows' mean and spread) — fit inside the pipeline, so only on the training window."""
    def make(params: dict, n_rows: int):
        steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
        return make_pipeline(*steps, *([StandardScaler()] if scale else []), estimator(**params))
    return make


MODELS = {
    "lightgbm": ("LightGBM", _lightgbm),
    "logistic": ("Logística regularizada", _sklearn(RowwiseLogisticRegression, scale=True)),
    "random_forest": ("Random Forest", _sklearn(RandomForestClassifier)),
    "extra_trees": ("Extra Trees", _sklearn(ExtraTreesClassifier)),
    "naive_bayes": ("Naive Bayes gaussiano", _sklearn(GaussianNB)),
}


def importance(model, n_features: int) -> np.ndarray:
    """Normalized importance: tree gain/impurity, |coefficient| for the logistic, zeros when the model has neither."""
    final = model[-1] if hasattr(model, "steps") else model
    if hasattr(final, "feature_importances_"):
        values = np.asarray(final.feature_importances_, dtype=float)
    elif hasattr(final, "coef_"):
        values = np.abs(np.asarray(final.coef_, dtype=float)).ravel()
    else:
        values = np.zeros(n_features)
    return values / values.sum() if values.sum() > 0 else values
