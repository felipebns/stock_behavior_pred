import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Optional

from engine.algorithms.base import Algorithm


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out).squeeze(-1)


class LSTMAlgorithm(Algorithm):
    def __init__(
        self,
        lookback: int = 20,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 64,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        self.lookback = max(1, int(lookback))
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.device = device
        self.seed = int(seed)

        self.model: Optional[LSTMClassifier] = None
        self.scaler = StandardScaler()
        self._train_prefix: Optional[pd.DataFrame] = None

    def name(self) -> str:
        return "LSTM"

    def fit(self, train_df: pd.DataFrame, valid_df: pd.DataFrame, features: list[str], target_col: str) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.scaler.fit(train_df[features].values)
        self._train_prefix = self._build_train_prefix(train_df)

        X_train, y_train, _ = self._build_sequences(train_df, features, target_col)
        if X_train.size == 0:
            raise ValueError("No training sequences were generated for LSTM")

        device = self._resolve_device()
        self.model = LSTMClassifier(
            input_size=len(features),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(device)

        dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        probs = self.predict_proba(df, features)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LSTM model is not trained. Call fit() first.")

        X, _, _ = self._build_sequences(df, features, target_col=None, prefix_df=self._train_prefix)
        if X.size == 0:
            return np.array([], dtype=float)

        device = self._resolve_device()
        self.model.eval()

        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                logits = self.model(batch_x)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def _build_train_prefix(self, train_df: pd.DataFrame) -> pd.DataFrame:
        if self.lookback <= 1:
            return train_df.iloc[0:0].copy()
        return train_df.groupby("ticker").tail(self.lookback - 1)

    def _build_sequences(
        self,
        target_df: pd.DataFrame,
        features: list[str],
        target_col: Optional[str],
        prefix_df: Optional[pd.DataFrame] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray], list[int]]:
        sequences: list[np.ndarray] = []
        targets: list[float] = []
        indices: list[int] = []

        lookback = self.lookback
        n_features = len(features)

        for ticker in target_df["ticker"].unique():
            ticker_df = target_df[target_df["ticker"] == ticker]
            if ticker_df.empty:
                continue

            prefix_tail = None
            if prefix_df is not None:
                prefix_tail = prefix_df[prefix_df["ticker"] == ticker].tail(lookback - 1)

            if prefix_tail is not None and not prefix_tail.empty:
                combined = pd.concat([prefix_tail, ticker_df], axis=0)
                prefix_len = len(prefix_tail)
            else:
                combined = ticker_df
                prefix_len = 0

            combined_scaled = self.scaler.transform(combined[features].values)

            for i in range(len(ticker_df)):
                end_idx = prefix_len + i
                start_idx = end_idx - (lookback - 1)

                if start_idx < 0:
                    pad_len = -start_idx
                    seq = combined_scaled[0 : end_idx + 1]
                    pad = np.zeros((pad_len, n_features), dtype=np.float32)
                    seq = np.vstack([pad, seq])
                else:
                    seq = combined_scaled[start_idx : end_idx + 1]

                sequences.append(seq.astype(np.float32))

                if target_col is not None:
                    targets.append(float(ticker_df.iloc[i][target_col]))
                indices.append(int(ticker_df.index[i]))

        if not sequences:
            empty_x = np.empty((0, lookback, n_features), dtype=np.float32)
            return empty_x, np.array([], dtype=np.float32) if target_col else None, []

        X = np.stack(sequences, axis=0)
        y = np.array(targets, dtype=np.float32) if target_col else None
        return X, y, indices

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
