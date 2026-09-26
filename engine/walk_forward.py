"""Daily retraining on a rolling window, predicting only the next decision date."""
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    importance: pd.DataFrame


def walk_forward(dataset: pd.DataFrame, features: list[str], calendar: pd.DatetimeIndex, train_window: int,
                 model_factory: Callable, start=None, end=None, progress_every: int = 0, log=print) -> WalkForwardResult:
    """Row s's label needs the open of s+2, so the fit for decision t uses exactly the rows dated t-X-1 .. t-2."""
    row_pos = calendar.get_indexer(dataset.index.get_level_values("date"))
    X = dataset[features].to_numpy(dtype=np.float32)
    y = dataset["label"].to_numpy(dtype=float)
    decisions = np.unique(row_pos)
    decisions = decisions[decisions - train_window - 1 >= row_pos[0]]
    if start is not None:
        decisions = decisions[calendar[decisions] >= pd.Timestamp(start)]
    if end is not None:
        decisions = decisions[calendar[decisions] <= pd.Timestamp(end)]
    if len(decisions) == 0:
        raise ValueError("Nenhuma data de decisão com janela de treino completa no intervalo pedido.")

    predictions, importance = [], []
    started = time.perf_counter()
    for n, k in enumerate(decisions, start=1):
        lo = np.searchsorted(row_pos, k - train_window - 1, side="left")
        hi = np.searchsorted(row_pos, k - 2, side="right")
        known = ~np.isnan(y[lo:hi])
        model = model_factory()
        model.fit(X[lo:hi][known], y[lo:hi][known].astype(int))

        first, last = np.searchsorted(row_pos, k, side="left"), np.searchsorted(row_pos, k, side="right")
        block = dataset.iloc[first:last]
        predictions.append(pd.DataFrame({
            "date": block.index.get_level_values("date"),
            "ticker": block.index.get_level_values("ticker"),
            "prob": model.predict_proba(X[first:last])[:, 1],
            "fwd_ret": block["fwd_ret"].to_numpy(),
            "label": block["label"].to_numpy(),
        }))
        gain = np.asarray(model.feature_importances_, dtype=float)
        importance.append(gain / gain.sum() if gain.sum() > 0 else gain)

        if progress_every and n % progress_every == 0:
            elapsed = time.perf_counter() - started
            remaining = elapsed / n * (len(decisions) - n)
            log(f"walk-forward {n}/{len(decisions)} ({calendar[k].date()}) · {elapsed / 60:.1f} min · faltam ~{remaining / 60:.1f} min")

    return WalkForwardResult(
        predictions=pd.concat(predictions, ignore_index=True),
        importance=pd.DataFrame(importance, index=pd.DatetimeIndex(calendar[decisions], name="date"), columns=features),
    )
