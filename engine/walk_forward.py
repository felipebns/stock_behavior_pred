"""Walk-forward: after each close, predict the next day's open-to-close move with a model fit only on labels already
known; decision dates run in parallel."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs

from engine.models import importance

SCOPES = {"pooled": "um modelo para todas", "per_stock": "um modelo por ação"}


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    importance: pd.DataFrame


class WalkForward:
    """Row s's label (open to close of s+1) is known at the close of s+1, so a fit at decision t uses exactly the rows
    dated t-X .. t-1; that fit also predicts the next retrain_every-1 decisions, all after t. "pooled" fits one model on
    every stock's rows; "per_stock" fits one model per stock on its own rows."""

    def __init__(self, train_window: int, make_model, params: dict, retrain_every: int = 1, scope: str = "pooled",
                 min_stock_rows: int = 63, n_jobs: int = 1):
        self.train_window = train_window
        self.make_model = make_model
        self.params = params
        self.retrain_every = retrain_every
        self.scope = scope
        self.min_stock_rows = min_stock_rows
        self.n_jobs = n_jobs

    def decisions(self, row_pos: np.ndarray, calendar: pd.DatetimeIndex, start=None) -> np.ndarray:
        first = row_pos[0] + self.train_window
        begin = first if start is None else calendar.searchsorted(pd.Timestamp(start))
        if begin < first:
            room = begin - row_pos[0]
            if room < 1:
                raise ValueError(f"não há pregões de dados antes de {pd.Timestamp(start).date()} para treinar")
            raise ValueError(f"a janela de {self.train_window} pregões precisa de {self.train_window} pregões de dados "
                             f"antes de {pd.Timestamp(start).date()}; o máximo é {room}")
        decisions = np.unique(row_pos)
        decisions = decisions[decisions >= begin]
        if len(decisions) == 0:
            raise ValueError("nenhuma data de decisão depois do início pedido")
        return decisions

    def run(self, dataset: pd.DataFrame, features: list[str], calendar: pd.DatetimeIndex, start=None,
            progress=None) -> WalkForwardResult:
        row_pos = calendar.get_indexer(dataset.index.get_level_values("date"))
        stocks = pd.factorize(dataset.index.get_level_values("ticker"))[0]
        X = dataset[features].to_numpy(dtype=np.float32)
        y = dataset["label"].to_numpy(dtype=float)
        decisions = self.decisions(row_pos, calendar, start)
        fits = [decisions[i:i + self.retrain_every] for i in range(0, len(decisions), self.retrain_every)]
        blocks = np.array_split(np.arange(len(fits)), min(len(fits), 4 * effective_n_jobs(self.n_jobs)))
        jobs = (delayed(_fit_block)(X, y, row_pos, stocks, [fits[i] for i in block], self.train_window, self.make_model,
                                    self.params, self.scope, self.min_stock_rows) for block in blocks)
        prob = np.full(len(dataset), np.nan)
        gains = []
        for done, (rows, probs, block_gains) in enumerate(Parallel(n_jobs=self.n_jobs, return_as="generator")(jobs), start=1):
            prob[rows] = probs
            gains.append(block_gains)
            if progress is not None:
                progress(done / len(blocks))
        decided = np.flatnonzero(np.isin(row_pos, decisions))
        predictions = pd.DataFrame({
            "date": dataset.index.get_level_values("date")[decided],
            "ticker": dataset.index.get_level_values("ticker")[decided],
            "prob": prob[decided],
            "fwd_ret": dataset["fwd_ret"].to_numpy()[decided],
            "label": y[decided],
        })
        importance_ = pd.DataFrame(np.vstack(gains), index=pd.DatetimeIndex(calendar[[group[0] for group in fits]], name="date"),
                                   columns=features)
        return WalkForwardResult(predictions, importance_)


def _fit(X, y, make_model, params, min_rows=1):
    """P(up) for new rows and the normalized importance. Fewer than min_rows examples or a single class predict the
    up-frequency of those examples (0.5 with none): a classifier fit on one class would put it in the wrong column."""
    if len(y) < min_rows or len(np.unique(y)) < 2:
        constant = float(y.mean()) if len(y) else 0.5
        return (lambda rows: np.full(len(rows), constant)), np.zeros(X.shape[1])
    model = make_model(params, len(y)).fit(X, y.astype(int))
    return (lambda rows: model.predict_proba(rows)[:, 1]), importance(model, X.shape[1])


def _fit_block(X, y, row_pos, stocks, fits, window, make_model, params, scope, min_stock_rows):
    rows_out, probs_out, gains = [], [], []
    for group in fits:
        train = np.arange(np.searchsorted(row_pos, group[0] - window, side="left"),
                          np.searchsorted(row_pos, group[0] - 1, side="right"))
        train = train[~np.isnan(y[train])]
        target = np.arange(np.searchsorted(row_pos, group[0], side="left"), np.searchsorted(row_pos, group[-1], side="right"))
        if scope == "pooled":
            predict, gain = _fit(X[train], y[train], make_model, params)
            rows_out.append(target)
            probs_out.append(predict(X[target]))
            gains.append(gain)
            continue
        order = train[np.argsort(stocks[train], kind="stable")]
        sorted_stocks = stocks[order]
        stock_gains = []
        for stock in np.unique(stocks[target]):
            mine = order[np.searchsorted(sorted_stocks, stock, side="left"):np.searchsorted(sorted_stocks, stock, side="right")]
            predict, gain = _fit(X[mine], y[mine], make_model, params, min_stock_rows)
            rows = target[stocks[target] == stock]
            rows_out.append(rows)
            probs_out.append(predict(X[rows]))
            stock_gains.append(gain)
        gains.append(np.mean(stock_gains, axis=0))
    return np.concatenate(rows_out), np.concatenate(probs_out), np.vstack(gains)
