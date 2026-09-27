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
    """Spearman correlation between predicted probability and realized return, per decision date."""
    realized = predictions.dropna(subset=["fwd_ret"])
    return realized.groupby("date")[["prob", "fwd_ret"]].apply(lambda day: day["prob"].corr(day["fwd_ret"], method="spearman"))


def model_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    known = predictions.dropna(subset=["label"])
    label, prob = known["label"].astype(int), known["prob"]
    return {
        "auc": float(roc_auc_score(label, prob)),
        "accuracy": float(((prob > 0.5).astype(int) == label).mean()),
        "ic": float(daily_ic(known).mean()),
    }


def monthly_auc(predictions: pd.DataFrame) -> pd.Series:
    known = predictions.dropna(subset=["label"])

    def auc(period: pd.DataFrame) -> float:
        return float(roc_auc_score(period["label"], period["prob"])) if period["label"].nunique() == 2 else float("nan")

    return known.groupby(pd.Grouper(key="date", freq="ME"))[["label", "prob"]].apply(auc)
