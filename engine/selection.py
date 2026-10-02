"""Walk-forward choice of run and threshold: every `step` days, pick the candidate with the best net Sharpe over the
previous `lookback` days and follow it until the next pick. Nothing is refit: candidates are out-of-sample already."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.metrics import TRADING_DAYS

CHOICE_COLUMNS = ["start", "candidate", "lookback_sharpe", "days"]


@dataclass
class SelectionResult:
    daily: pd.DataFrame
    choices: pd.DataFrame


def walk_forward_selection(candidates: dict[str, pd.DataFrame], lookback: int, step: int) -> SelectionResult:
    """`candidates`: name → Portfolio daily frame, all on the same days. Row i is the day traded after decision i and is
    realized at that day's close, so the pick made at decision i sees rows i-lookback .. i-1 only. An undefined Sharpe
    (no variation, e.g. all cash) counts as 0; ties go to the first candidate."""
    names = list(candidates)
    index = candidates[names[0]].index
    if any(not frame.index.equals(index) for frame in candidates.values()):
        raise ValueError("os candidatos precisam cobrir os mesmos dias")
    excess = pd.DataFrame({name: frame["net"] - frame["rf"] for name, frame in candidates.items()})
    pieces, choices = [], []
    for start in range(lookback, len(index), step):
        past = excess.iloc[start - lookback:start]
        sharpe = (past.mean() / past.std(ddof=1) * np.sqrt(TRADING_DAYS)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        best = sharpe.idxmax()
        end = min(start + step, len(index))
        pieces.append(candidates[best].iloc[start:end].assign(candidate=best))
        choices.append({"start": index[start], "candidate": best, "lookback_sharpe": float(sharpe[best]), "days": end - start})
    daily = pd.concat(pieces) if pieces else candidates[names[0]].iloc[:0].assign(candidate=pd.Series(dtype=object))
    return SelectionResult(daily, pd.DataFrame(choices, columns=CHOICE_COLUMNS))
