"""Performance and model-quality metrics for daily data (252 trading days per year)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TRADING_DAYS = 252


def drawdown(returns: pd.Series) -> pd.Series:
    """Distance from the running peak; the initial capital (1.0) is the first peak."""
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax().clip(lower=1.0) - 1.0


def performance_metrics(returns: pd.Series, rf: pd.Series, invested: pd.Series | None = None) -> dict[str, float]:
    excess = returns - rf
    spread = excess.std(ddof=1)
    active = returns[invested.astype(bool)] if invested is not None else returns
    return {
        "total_return": float((1.0 + returns).prod() - 1.0),
        "volatility": float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": float(excess.mean() / spread * np.sqrt(TRADING_DAYS)) if spread > 0 else float("nan"),
        "max_drawdown": float(drawdown(returns).min()),
        "hit_ratio": float((active > 0).mean()) if len(active) else float("nan"),
    }


def daily_ic(predictions: pd.DataFrame) -> pd.Series:
    """Spearman correlation between predicted probability and realized return, per decision date (empty if none is known)."""
    realized = predictions.dropna(subset=["fwd_ret"])
    if realized.empty:
        return pd.Series(dtype=float)
    return realized.groupby("date")[["prob", "fwd_ret"]].apply(lambda day: day["prob"].corr(day["fwd_ret"], method="spearman"))


def model_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """NaN where there is nothing to measure: no known outcome yet, or a single class."""
    known = predictions.dropna(subset=["label"])
    label, prob = known["label"].astype(int), known["prob"]
    return {
        "auc": float(roc_auc_score(label, prob)) if label.nunique() == 2 else float("nan"),
        "accuracy": float(((prob > 0.5).astype(int) == label).mean()) if len(known) else float("nan"),
        "ic": float(daily_ic(known).mean()) if len(known) else float("nan"),
    }


def monthly_auc(predictions: pd.DataFrame) -> pd.Series:
    known = predictions.dropna(subset=["label"])

    def auc(period: pd.DataFrame) -> float:
        return float(roc_auc_score(period["label"], period["prob"])) if period["label"].nunique() == 2 else float("nan")

    if known.empty:
        return pd.Series(dtype=float)
    return known.groupby(pd.Grouper(key="date", freq="ME"))[["label", "prob"]].apply(auc)


def _ranked(predictions: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known outcome, with the return relative to the day's median and the probability rank and decile that day."""
    known = predictions.dropna(subset=["label"])
    by_date = known.groupby("date")
    rank = by_date["prob"].rank(method="first")
    size = by_date["prob"].transform("size")
    return known.assign(
        relative=known["fwd_ret"] - by_date["fwd_ret"].transform("median"),
        rank=rank, size=size, decile=np.ceil(10 * rank / size).astype(int),
    )


def deciles(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per probability decile (1 = lowest that day): mean prob, share that beat the median and return relative to it —
    averaged within each date first, then across dates."""
    per_day = _ranked(predictions).groupby(["date", "decile"])[["prob", "label", "relative"]].mean()
    return per_day.groupby(level="decile").mean()


def decile_spread(predictions: pd.DataFrame) -> pd.Series:
    """Per decision date: mean return of the top 10% by probability minus the bottom 10% (at least one stock each)."""
    ranked = _ranked(predictions)
    side = np.maximum(1, ranked["size"] // 10)
    top = ranked[ranked["rank"] > ranked["size"] - side].groupby("date")["fwd_ret"].mean()
    bottom = ranked[ranked["rank"] <= side].groupby("date")["fwd_ret"].mean()
    return top - bottom
