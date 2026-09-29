"""Walk-forward: a decision every `horizon` sessions, each predicted by a model fit only on labels already known;
decision dates run in parallel."""
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
    """Row s's label needs the open of s+1+horizon, so a fit at decision t uses exactly the rows dated
    t-horizon-X .. t-horizon-1; that model also predicts the next retrain_every-1 decisions, all after t."""

    def __init__(self, train_window: int, params: dict, horizon: int = 1, retrain_every: int = 1,
                 min_child_share: float = 0.004, min_child_floor: int = 20, n_jobs: int = 1, model_class=LGBMClassifier):
        self.train_window = train_window
        self.params = params
        self.horizon = horizon
        self.retrain_every = retrain_every
        self.min_child_share = min_child_share
        self.min_child_floor = min_child_floor
        self.n_jobs = n_jobs
        self.model_class = model_class

    def decisions(self, row_pos: np.ndarray, calendar: pd.DatetimeIndex, start=None) -> np.ndarray:
        first = row_pos[0] + self.train_window + self.horizon
        begin = first if start is None else calendar.searchsorted(pd.Timestamp(start))
        if begin < first:
            raise ValueError(f"a janela de {self.train_window} pregões com horizonte de {self.horizon} precisa de "
                             f"{self.train_window + self.horizon} pregões de dados antes de {pd.Timestamp(start).date()}; "
                             f"o máximo é {begin - row_pos[0] - self.horizon}")
        with_rows = np.unique(row_pos)
        schedule = np.arange(begin, with_rows[-1] + 1, self.horizon)
        decisions = schedule[np.isin(schedule, with_rows)]
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
        jobs = (delayed(_fit_block)(X, y, row_pos, [fits[i] for i in block], self.train_window, self.horizon,
                                    self.params, self.min_child_share, self.min_child_floor, self.model_class)
                for block in blocks)
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


def _fit_block(X, y, row_pos, fits, window, horizon, params, min_child_share, min_child_floor, model_class):
    probs, gains = [], []
    for group in fits:
        lo = np.searchsorted(row_pos, group[0] - horizon - window, side="left")
        hi = np.searchsorted(row_pos, group[0] - horizon - 1, side="right")
        known = ~np.isnan(y[lo:hi])
        model = model_class(**params, min_child_samples=max(min_child_floor, round(min_child_share * known.sum())))
        model.fit(X[lo:hi][known], y[lo:hi][known].astype(int))
        for t in group:
            first, last = np.searchsorted(row_pos, t, side="left"), np.searchsorted(row_pos, t, side="right")
            probs.append(model.predict_proba(X[first:last])[:, 1])
        gain = np.asarray(model.feature_importances_, dtype=float)
        gains.append(gain / gain.sum() if gain.sum() > 0 else gain)
    return np.concatenate(probs), np.vstack(gains)
