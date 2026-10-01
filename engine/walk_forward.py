"""Walk-forward: after each close, predict the next day with a model fit only on labels already known; decision
dates run in parallel."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from lightgbm import LGBMClassifier

MODELS = {"lightgbm": LGBMClassifier}


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    importance: pd.DataFrame


class WalkForward:
    """Row s's label needs the open of s+2, so a fit at decision t uses exactly the rows dated t-X-1 .. t-2; that model
    also predicts the next retrain_every-1 decisions, all after t."""

    def __init__(self, train_window: int, params: dict, retrain_every: int = 1, min_child_share: float = 0.004,
                 min_child_floor: int = 20, n_jobs: int = 1, model_class=LGBMClassifier):
        self.train_window = train_window
        self.params = params
        self.retrain_every = retrain_every
        self.min_child_share = min_child_share
        self.min_child_floor = min_child_floor
        self.n_jobs = n_jobs
        self.model_class = model_class

    def decisions(self, row_pos: np.ndarray, calendar: pd.DatetimeIndex, start=None) -> np.ndarray:
        first = row_pos[0] + self.train_window + 1
        begin = first if start is None else calendar.searchsorted(pd.Timestamp(start))
        if begin < first:
            room = begin - row_pos[0] - 1
            if room < 1:
                raise ValueError(f"não há pregões de dados antes de {pd.Timestamp(start).date()} para treinar")
            raise ValueError(f"a janela de {self.train_window} pregões precisa de {self.train_window + 1} pregões de dados "
                             f"antes de {pd.Timestamp(start).date()}; o máximo é {room}")
        decisions = np.unique(row_pos)
        decisions = decisions[decisions >= begin]
        if len(decisions) == 0:
            raise ValueError("nenhuma data de decisão depois do início pedido")
        return decisions

    def run(self, dataset: pd.DataFrame, features: list[str], calendar: pd.DatetimeIndex, start=None,
            progress=None) -> WalkForwardResult:
        row_pos = calendar.get_indexer(dataset.index.get_level_values("date"))
        X = dataset[features].to_numpy(dtype=np.float32)
        y = dataset["label"].to_numpy(dtype=float)
        decisions = self.decisions(row_pos, calendar, start)
        fits = [decisions[i:i + self.retrain_every] for i in range(0, len(decisions), self.retrain_every)]
        blocks = np.array_split(np.arange(len(fits)), min(len(fits), 4 * effective_n_jobs(self.n_jobs)))
        jobs = (delayed(_fit_block)(X, y, row_pos, [fits[i] for i in block], self.train_window, self.params,
                                    self.min_child_share, self.min_child_floor, self.model_class) for block in blocks)
        results = []
        for done, result in enumerate(Parallel(n_jobs=self.n_jobs, return_as="generator")(jobs), start=1):
            results.append(result)
            if progress is not None:
                progress(done / len(blocks))
        rows = dataset[np.isin(row_pos, decisions)]
        predictions = pd.DataFrame({
            "date": rows.index.get_level_values("date"),
            "ticker": rows.index.get_level_values("ticker"),
            "prob": np.concatenate([probs for probs, _ in results]),
            "fwd_ret": rows["fwd_ret"].to_numpy(),
            "label": rows["label"].to_numpy(),
        })
        importance = pd.DataFrame(np.vstack([gains for _, gains in results]),
                                  index=pd.DatetimeIndex(calendar[[group[0] for group in fits]], name="date"),
                                  columns=features)
        return WalkForwardResult(predictions, importance)


def _fit(X, y, params, min_child_share, min_child_floor, model_class):
    """A function giving P(up) for new rows, and the normalized gains. A window with a single class (say X = 1 on a
    day every stock fell) predicts that class's frequency: a classifier fit on one class would put it in the wrong
    column."""
    if len(np.unique(y)) < 2:
        constant = float(y.mean()) if len(y) else 0.5
        return (lambda rows: np.full(len(rows), constant)), np.zeros(X.shape[1])
    model = model_class(**params, min_child_samples=max(min_child_floor, round(min_child_share * len(y))))
    model.fit(X, y)
    gain = np.asarray(model.feature_importances_, dtype=float)
    return (lambda rows: model.predict_proba(rows)[:, 1]), (gain / gain.sum() if gain.sum() > 0 else gain)


def _fit_block(X, y, row_pos, fits, window, params, min_child_share, min_child_floor, model_class):
    probs, gains = [], []
    for group in fits:
        lo = np.searchsorted(row_pos, group[0] - window - 1, side="left")
        hi = np.searchsorted(row_pos, group[0] - 2, side="right")
        known = ~np.isnan(y[lo:hi])
        predict, gain = _fit(X[lo:hi][known], y[lo:hi][known].astype(int), params, min_child_share, min_child_floor,
                             model_class)
        for t in group:
            first, last = np.searchsorted(row_pos, t, side="left"), np.searchsorted(row_pos, t, side="right")
            probs.append(predict(X[first:last]))
        gains.append(gain)
    return np.concatenate(probs), np.vstack(gains)
