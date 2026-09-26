"""Performance and model-quality metrics for daily data (252 trading days per year)."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

TRADING_DAYS = 252


def drawdown(returns: pd.Series) -> pd.Series:
    """Distance from the running peak; the initial capital (1.0) is the first peak."""
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax().clip(lower=1.0) - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown(returns).min())


def _annualized(mean: float, spread: float) -> float:
    return float(mean / spread * np.sqrt(TRADING_DAYS)) if spread > 0 else float("nan")


def performance_metrics(returns: pd.Series, rf: pd.Series, benchmark: pd.Series | None = None,
                        invested: pd.Series | None = None) -> dict[str, float]:
    total = float((1.0 + returns).prod() - 1.0)
    cagr = float((1.0 + total) ** (TRADING_DAYS / len(returns)) - 1.0)
    excess = returns - rf
    worst = max_drawdown(returns)
    active_days = returns[invested.astype(bool)] if invested is not None else returns
    out = {
        "total_return": total,
        "cagr": cagr,
        "volatility": float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": _annualized(excess.mean(), excess.std(ddof=1)),
        "sortino": _annualized(excess.mean(), float(np.sqrt((excess.clip(upper=0.0) ** 2).mean()))),
        "max_drawdown": worst,
        "calmar": cagr / abs(worst) if worst < 0 else float("nan"),
        "hit_ratio": float((active_days > 0).mean()) if len(active_days) else float("nan"),
    }
    if benchmark is not None:
        bench_excess = benchmark - rf
        beta = float(np.cov(excess, bench_excess, ddof=1)[0, 1] / bench_excess.var(ddof=1))
        active = returns - benchmark
        tracking = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
        out.update({
            "beta": beta,
            "alpha": float((excess.mean() - beta * bench_excess.mean()) * TRADING_DAYS),
            "tracking_error": tracking,
            "information_ratio": float(active.mean() * TRADING_DAYS / tracking) if tracking > 0 else float("nan"),
        })
    return out


def daily_ic(predictions: pd.DataFrame) -> pd.Series:
    """Spearman correlation between predicted probability and realized return, per decision date."""
    realized = predictions.dropna(subset=["fwd_ret"])
    return realized.groupby("date")[["prob", "fwd_ret"]].apply(lambda day: day["prob"].corr(day["fwd_ret"], method="spearman"))


def model_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    known = predictions.dropna(subset=["label"])
    label, prob = known["label"].astype(int), known["prob"]
    ic = daily_ic(known).dropna()
    ic_spread = ic.std(ddof=1)
    return {
        "auc": float(roc_auc_score(label, prob)),
        "accuracy": float(((prob > 0.5).astype(int) == label).mean()),
        "brier": float(brier_score_loss(label, prob)),
        "log_loss": float(log_loss(label, prob)),
        "base_rate": float(label.mean()),
        "ic_mean": float(ic.mean()),
        "ic_tstat": float(ic.mean() / ic_spread * np.sqrt(len(ic))) if ic_spread > 0 else float("nan"),
        "n_predictions": int(len(known)),
    }


def rolling_auc(predictions: pd.DataFrame, freq: str = "ME") -> pd.Series:
    known = predictions.dropna(subset=["label"])

    def auc(period: pd.DataFrame) -> float:
        return float(roc_auc_score(period["label"], period["prob"])) if period["label"].nunique() == 2 else float("nan")

    return known.groupby(pd.Grouper(key="date", freq=freq))[["label", "prob"]].apply(auc)


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    known = predictions.dropna(subset=["label"])
    bucket = pd.qcut(known["prob"], q=bins, duplicates="drop")
    table = known.groupby(bucket, observed=True).agg(prob_mean=("prob", "mean"), up_rate=("label", "mean"), n=("label", "size"))
    return table.reset_index(drop=True)


def position_hit_ratio(positions: pd.DataFrame) -> float:
    realized = positions["fwd_ret"].dropna()
    return float((realized > 0).mean()) if len(realized) else float("nan")


def annual_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).groupby(returns.index.year).prod() - 1.0
