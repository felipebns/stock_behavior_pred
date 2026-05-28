from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

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
