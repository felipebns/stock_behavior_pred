from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class Algorithm(ABC):
    @abstractmethod
    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str], target_col: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def predict_proba(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def feature_profile(self) -> str:
        raise NotImplementedError

class BaseScikitClassificationAlgorithm(Algorithm):
    def __init__(self, model):
        self.model = make_pipeline(StandardScaler(), model)

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str], target_col: str) -> None:
        X_train = train_df[features].values
        y_train = train_df[target_col].values
        
        if valid_df is not None and not valid_df.empty:
            X_valid = valid_df[features].values
            y_valid = valid_df[target_col].values
            
            X_all = np.vstack((X_train, X_valid))
            y_all = np.concatenate((y_train, y_valid))
        else:
            X_all, y_all = X_train, y_train
            
        self.model.fit(X_all, y_all)

    def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        X = df[features].values
        return self.model.predict(X)
        
    def predict_proba(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        X = df[features].values
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.decision_function(X) # fallback for SVC without probability

    def feature_profile(self) -> str:
        return "classification_indicators"
